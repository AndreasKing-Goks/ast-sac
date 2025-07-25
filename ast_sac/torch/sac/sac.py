from collections import OrderedDict, namedtuple
from typing import Tuple

import numpy as np
import torch
import torch.optim as optim
from ast_sac.core.loss import LossFunction, LossStatistics
from torch import nn

import ast_sac.torch.utils.pytorch_util as ptu
from ast_sac.core.eval_util import create_stats_ordered_dict
from ast_sac.torch.core.torch_rl_algorithm import TorchTrainer
from ast_sac.core.logging import add_prefix
import gtimer as gt

SACLosses = namedtuple(
    'SACLosses',
    'policy_loss qf1_loss qf2_loss alpha_loss',
)

class SACTrainer(TorchTrainer, LossFunction):
    def __init__(
            self,
            env,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
            
            action_reg_coeff =None,
            clip_val=np.inf
            
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy is None:
                # Use heuristic value from SAC paper
                self.target_entropy = -np.prod(
                    self.env.action_space.shape).item()
            else:
                self.target_entropy = target_entropy
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.eval_statistics = OrderedDict()
        
        self.action_reg_coeff = action_reg_coeff
        self.clip_val = clip_val

    def train_from_torch(self, batch):
        gt.blank_stamp()
        losses, stats = self.compute_loss(
            batch,
            skip_statistics=not self._need_to_update_eval_statistics,
        )
        # ========= DEBUG CHECK =========
        # for loss_name, loss_value in losses._asdict().items():
        #     if torch.isnan(loss_value).any():
        #         print(f"NaN detected in loss: {loss_name}")
        #     if torch.isinf(loss_value).any():
        #         print(f"Inf detected in loss: {loss_name}")
        # ===============================
        '''
        Update networks
        '''
        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            losses.alpha_loss.backward()
            self.alpha_optimizer.step()

        self.policy_optimizer.zero_grad()
        losses.policy_loss.backward()
        self.policy_optimizer.step()

        self.qf1_optimizer.zero_grad()
        losses.qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        losses.qf2_loss.backward()
        self.qf2_optimizer.step()

        self._n_train_steps_total += 1

        self.try_update_target_networks()
        if self._need_to_update_eval_statistics:
            self.eval_statistics = stats
            # Compute statistics using only one batch per epoch
            self._need_to_update_eval_statistics = False
        gt.stamp('sac training', unique=False)

    def try_update_target_networks(self):
        if self._n_train_steps_total % self.target_update_period == 0:
            self.update_target_networks()

    def update_target_networks(self):
        ptu.soft_update_from_to(
            self.qf1, self.target_qf1, self.soft_target_tau
        )
        ptu.soft_update_from_to(
            self.qf2, self.target_qf2, self.soft_target_tau
        )

    def compute_loss(
        self,
        batch,
        skip_statistics=False,
        debug=False,
    ) -> Tuple[SACLosses, LossStatistics]:
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Policy and Alpha Loss
        """
         # ========= DEBUG CHECK =========
        if debug:
            print("obs:", torch.isnan(obs).any(), torch.isinf(obs).any())
            print("actions:", torch.isnan(actions).any(), torch.isinf(actions).any())
        # ===============================
        dist = self.policy(obs)
        new_obs_actions, log_pi = dist.rsample_and_logprob()
        log_pi = log_pi.unsqueeze(-1)
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        # ========= DEBUG CHECK =========
        if debug:
            print("new_obs_actions:", torch.isnan(new_obs_actions).any(), torch.isinf(new_obs_actions).any())
            print("log_pi:", torch.isnan(log_pi).any(), torch.isinf(log_pi).any())
        # ===============================
        
        # Q values given observation and new_obs_actions
        q_new_actions = torch.min(
            self.qf1(obs, new_obs_actions),
            self.qf2(obs, new_obs_actions),
        )
        
        # ========= DEBUG CHECK =========
        if debug:
            print("q_new_actions:", torch.isnan(q_new_actions).any(), torch.isinf(q_new_actions).any())
        # ===============================
        
        policy_loss = (alpha*log_pi - q_new_actions).mean()
        
        ## IMPLEMENT ACTION REGULARIZATION IN POLICY LOSS
        # If do action regularization
        if self.action_reg_coeff:
            action_reg = (new_obs_actions ** 2).mean()
            policy_loss += self.action_reg_coeff * action_reg

        """
        QF Loss
        """
        # Q values given observation and actions from batches
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        # ========= DEBUG CHECK =========
        if debug:
            print('q1_pred nan:', torch.isnan(q1_pred).any().item(), 'inf:', torch.isinf(q1_pred).any().item())
            print('q2_pred nan:', torch.isnan(q2_pred).any().item(), 'inf:', torch.isinf(q2_pred).any().item())
            print("q1_pred range:", q1_pred.min().item(), q1_pred.max().item())
            print("q2_pred range:", q2_pred.min().item(), q2_pred.max().item())
        # ===============================
        next_dist = self.policy(next_obs)
        new_next_actions, new_log_pi = next_dist.rsample_and_logprob()
        new_log_pi = new_log_pi.unsqueeze(-1)
        target_q_values = torch.min(
            self.target_qf1(next_obs, new_next_actions),
            self.target_qf2(next_obs, new_next_actions),
        ) - alpha * new_log_pi

        # ========= DEBUG CHECK =========
        if debug:
            print('target_q_values nan:', torch.isnan(target_q_values).any().item(), 'inf:', torch.isinf(target_q_values).any().item())
            print("target_q_values range:", target_q_values.min().item(), target_q_values.max().item())
        # ===============================
        
        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        
        ## IMPLEMENT Q_TARGET CLIPPING
        # Default is infinite
        q_target = torch.clamp(q_target, min=-self.clip_val, max=self.clip_val)
        
        # ========= DEBUG CHECK =========
        if debug:
            print("rewards", rewards)
            print("target_q_values", target_q_values)
            print("q_target", q_target)
            print('q_target nan:', torch.isnan(q_target).any().item(), 'inf:', torch.isinf(q_target).any().item())
        # ===============================
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        # ========= DEBUG CHECK =========
        if debug:
            print("qf1_loss", qf1_loss.item())
            print("qf2_loss", qf2_loss.item())
            print("policy_loss", policy_loss.item())
            print('############################################################')
        # ===============================
        """
        Save some statistics for eval
        """
        eval_statistics = OrderedDict()
        if not skip_statistics:
            eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            policy_statistics = add_prefix(dist.get_diagnostics(), "policy/")
            eval_statistics.update(policy_statistics)
            if self.use_automatic_entropy_tuning:
                eval_statistics['Alpha'] = alpha.item()
                eval_statistics['Alpha Loss'] = alpha_loss.item()

        loss = SACLosses(
            policy_loss=policy_loss,
            qf1_loss=qf1_loss,
            qf2_loss=qf2_loss,
            alpha_loss=alpha_loss,
        )

        return loss, eval_statistics

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return stats

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]

    @property
    def optimizers(self):
        return [
            self.alpha_optimizer,
            self.qf1_optimizer,
            self.qf2_optimizer,
            self.policy_optimizer,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
        )