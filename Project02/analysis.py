'''analysis.py
Run statistical analyses and plot Numpy ndarray data
Trey Tuscai
CS 251 Data Analysis Visualization
Spring 2023
'''
import numpy as np
import matplotlib.pyplot as plt


class Analysis:
    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        self.data = data

        # Make plot font sizes legible
        plt.rcParams.update({'font.size': 18})

    def set_data(self, data):
        self.data = data
        pass

    def min(self, headers, rows=[]):
        selected_data = self.data.select_data(headers, rows)
        mins = np.min(selected_data, axis=0)
        return mins

    def max(self, headers, rows=[]):
        selected_data = self.data.select_data(headers, rows)
        maxs = np.max(selected_data, axis=0)
        return maxs

    def range(self, headers, rows=[]):
        mins = self.min(headers, rows)
        maxs = self.max(headers, rows)
        return mins, maxs

    def mean(self, headers, rows=[]):
        selected_data = self.data.select_data(headers, rows)
        num_rows, num_cols = selected_data.shape
        means = np.sum(selected_data, axis=0) / num_rows
        return means

    def var(self, headers, rows=[]):
        selected_data = self.data.select_data(headers, rows)
        deviations = selected_data - self.mean(headers, rows)
        vars = np.sum(deviations**2, axis=0) / (selected_data.shape[0] - 1)
        return vars

    def std(self, headers, rows=[]):
        variance = self.var(headers, rows)
        return np.sqrt(variance)

    def show(self):
        plt.show()

    def scatter(self, ind_var, dep_var, title):
        x = self.data.select_data([ind_var]).ravel()
        y = self.data.select_data([dep_var]).ravel()

        plt.figure(figsize = (7,7))
        plt.scatter(x, y,)
        plt.xlabel(ind_var)
        plt.ylabel(dep_var)
        plt.title(title)

        return x, y

    def pair_plot(self, data_vars, fig_sz=(12, 12), title=''):
        fig, axes = plt.subplots(len(data_vars), len(data_vars), figsize=fig_sz)
        for i in range(len(data_vars)):
            for j in range(len(data_vars)): 
                x = self.data.select_data([data_vars[j]])
                y = self.data.select_data([data_vars[i]])   
                axes[i, j].scatter(x, y)
                if j == 0:
                    axes[i, j].set_ylabel(data_vars[i])
                if i == len(data_vars) - 1:
                    axes[i, j].set_xlabel(data_vars[j])
                if i != len(data_vars) - 1:
                    axes[i, j].set_xticks([])
                if j != 0:
                    axes[i, j].set_yticks([])
        fig.suptitle(title)
        plt.tight_layout()
        return fig, axes


