import pandas as pd
import matplotlib.pyplot as plt


class ProgressPlotter:
    def __init__(self, csv_path):
        """
        Initialize the class by loading a CSV file into a pandas DataFrame.
        """
        self.df = pd.read_csv(csv_path)
        self.columns = self.df.columns.tolist()
        # print(f"Loaded CSV with columns: {self.columns}")

    def plot_column(self, column_name):
        """
        Plot a specific column by name.
        """
        if column_name not in self.df.columns:
            print(f"Column '{column_name}' not found in CSV.")
            return
        plt.figure(figsize=(10, 4))
        plt.plot(self.df[column_name], label=column_name)
        plt.title(f"Plot of column: {column_name}")
        plt.xlabel("Index")
        plt.ylabel(column_name)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        # plt.show()
        
    def plot_columns(self, column_names):
        """
        Plot multiple columns together on the same figure.

        Parameters:
        - column_names: List of column names to plot
        """
        missing = [col for col in column_names if col not in self.df.columns]
        if missing:
            print(f"These columns were not found in the CSV: {missing}")
            return

        plt.figure(figsize=(12, 6))
        for col in column_names:
            plt.plot(self.df[col], label=col)
        plt.title("Plot of multiple columns")
        plt.xlabel("Index")
        plt.ylabel("Values")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        # plt.show()

    def show_plot(self):
        plt.show()
    
    def list_columns(self):
        """
        Print a list of all columns in the CSV.
        """
        print("Available columns:")
        for col in self.columns:
            print(f"- {col}")
