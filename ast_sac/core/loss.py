import abc
from collections import OrderedDict

LossStatistic = OrderedDict

class LossFunction(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def compute_loss(self, batch, skip_statistics=False, **kwargs):
        '''
        Returns loss and statistics given a batch of data.
        
        param:
        - batch: Data to compute loss of
        - skip_statistics: Whether statistics should be calculated. If True, then
            an empty dict is returned for the statistics.

        returns: 
        - (loss, stats) tuple.
        '''
        pass