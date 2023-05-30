'''linear_regression.py
Subclass of Analysis that performs linear regression on data
YOUR NAME HERE
CS251 Data Analysis Visualization
Spring 2023
'''
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

import analysis


class LinearRegression(analysis.Analysis):
    '''
    Perform and store linear regression and related analyses
    '''

    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        super().__init__(data)

        # ind_vars: Python list of strings.
        #   1+ Independent variables (predictors) entered in the regression.
        self.ind_vars = None
        # dep_var: string. Dependent variable predicted by the regression.
        self.dep_var = None

        # A: ndarray. shape=(num_data_samps, num_ind_vars)
        #   Matrix for independent (predictor) variables in linear regression
        self.A = None

        # y: ndarray. shape=(num_data_samps, 1)
        #   Vector for dependent variable predictions from linear regression
        self.y = None

        # R2: float. R^2 statistic
        self.R2 = None

        # Mean SEE. float. Measure of quality of fit
        self.mse = None

        # slope: ndarray. shape=(num_ind_vars, 1)
        #   Regression slope(s)
        self.slope = None
        # intercept: float. Regression intercept
        self.intercept = None
        # residuals: ndarray. shape=(num_data_samps, 1)
        #   Residuals from regression fit
        self.residuals = None

        # p: int. Polynomial degree of regression model (Week 2)
        self.p = 1

        #condition number
        self.cond_X = None

    def linear_regression(self, ind_vars, dep_var):
        self.ind_vars = ind_vars
        self.dep_var = dep_var
        self.A = self.data.select_data(ind_vars)
        self.y = self.data.select_data([dep_var])
        Ahat = np.hstack((np.ones((self.A.shape[0],1)),self.A))
        c, _, _, _ = scipy.linalg.lstsq( Ahat, self.y )
        self.slope = c[1:]
        self.intercept = float(c[0])
        self.mse = self.compute_mse()
        y_pred = self.predict(X = self.A)
        self.residuals = self.compute_residuals(y_pred)
        self.R2 = self.r_squared(y_pred)
        pass

    def linear_regression_normal(self, ind_vars, dep_var):
        self.ind_vars = ind_vars
        self.dep_var = dep_var
        self.A = self.data.select_data(ind_vars)
        self.y = self.data.select_data([dep_var])
        Ahat = np.hstack((np.ones((self.A.shape[0],1)),self.A))
        ATA = Ahat.T @ Ahat
        self.cond_X = np.linalg.cond(ATA)
        ATA_inv = np.linalg.inv(ATA)
        c = ATA_inv @ Ahat.T @ self.y
        self.slope = c[1:]
        self.intercept = float(c[0])
        self.mse = self.compute_mse()
        y_pred = self.predict(X = self.A)
        self.residuals = self.compute_residuals(y_pred)
        self.R2 = self.r_squared(y_pred)
        pass

    def get_cond_x(self):
        return self.cond_X

    def predict(self, X=None):
        if X is None:
            X = self.A
        if self.p > 1:
            X = self.make_polynomial_matrix(X, self.p)
        y_pred = X @ self.slope + self.intercept
        return y_pred

    def r_squared(self, y_pred):
        SS_res = np.sum((self.y - y_pred) ** 2)
        SS_tot = np.sum((self.y - np.mean(self.y)) ** 2)
        R2 = 1 - (SS_res / SS_tot)
        return R2

    def compute_residuals(self, y_pred):
        self.residuals = self.y - y_pred
        return self.residuals

    def compute_mse(self):
        self.compute_residuals(self.predict())
        mse = np.mean(self.residuals**2)
        return mse

    def scatter(self, ind_var, dep_var, title):
        x, y = analysis.Analysis.scatter(self, ind_var, dep_var, title)
        x_min = min(x)
        x_max = max(x)
        x_line = np.linspace(x_min, x_max, 100).reshape(-1, 1)
        if self.p > 1:
            x_poly = self.make_polynomial_matrix(x_line, self.p)
            r_poly = np.dot(x_poly, self.slope) + self.intercept
            plt.plot(x_line, r_poly, color='red')
        else:
            r_line = x_line * self.slope[0] + self.intercept
            plt.plot(x_line, r_line, color='red')
        plt.scatter(x, y)
        plt.title(f'{title} (R^2 = {self.R2:.2f})')
        plt.show()
        pass

    def pair_plot(self, data_vars, fig_sz=(15, 15), hists_on_diag=True):
        fig, axes = analysis.Analysis.pair_plot(self, data_vars, fig_sz=fig_sz, title='')
        for i in range(len(data_vars)):
            for j in range(len(data_vars)):
                if hists_on_diag and i == j:
                    numVars = len(data_vars)
                    if i != 0 and i!=0:
                        axes[i, j].remove()
                    axes[i, j] = fig.add_subplot(numVars, numVars, i*numVars+j+1)
                    if j < numVars-1:
                        axes[i, j].set_xticks([])
                    else:
                        axes[i, j].set_xlabel(data_vars[i])
                    axes[i, j].set_yticks([])
                    axes[i, j].hist(self.data.select_data([data_vars[i]]))
                    continue
                x = self.data.select_data([data_vars[j]])
                x_min = min(x)
                x_max = max(x)
                x_line = np.linspace(x_min, x_max, 100)
                self.linear_regression([data_vars[j]], data_vars[i])
                r_squared = self.R2
                r_line = self.intercept + self.slope[0] * x_line
                axes[i, j].plot(x_line, r_line, color='red')
                axes[i, j].set_title(f'R^2 = {r_squared:.2f}')
        pass

    def make_polynomial_matrix(self, A, p):
        powers = np.arange(1, p+1)
        A_reshape = A.reshape(-1, 1)
        poly_matrix = np.power(np.broadcast_to(A_reshape, (A_reshape.shape[0], p)), np.broadcast_to(powers, (A_reshape.shape[0], p)))  
        return poly_matrix

    def poly_regression(self, ind_var, dep_var, p):
        self.ind_vars = ind_var
        self.dep_var = dep_var
        self.p = p
        self.A = self.data.select_data(ind_var)
        self.y = self.data.select_data([dep_var])
        Ahat = np.hstack((np.ones((self.A.shape[0],1)), self.make_polynomial_matrix(self.A, self.p)))
        c, _, _, _ = scipy.linalg.lstsq( Ahat, self.y )
        self.slope = c[1:]
        self.intercept = float(c[0])
        self.mse = self.compute_mse()
        y_pred = self.predict(X = self.A)
        self.residuals = self.compute_residuals(y_pred)
        self.R2 = self.r_squared(y_pred)
        pass

    def get_fitted_slope(self):
        return self.slope

    def get_fitted_intercept(self):
        return self.intercept

    def initialize(self, ind_vars, dep_var, slope, intercept, p):
        self.ind_vars = ind_vars
        self.dep_var = dep_var
        self.slope = slope
        self.intercept = intercept
        self.p = p
        self.poly_matrix = self.make_polynomial_matrix(self.data.get_all_data(), self.p)
        pass
