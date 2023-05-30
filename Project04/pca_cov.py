'''pca_cov.py
Performs principal component analysis using the covariance matrix approach
YOUR NAME HERE
CS 251/2 Data Analysis Visualization
Spring 2023
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class PCA_COV:
    '''
    Perform and store principal component analysis results
    '''

    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: pandas DataFrame. shape=(num_samps, num_vars)
            Contains all the data samples and variables in a dataset. Should be set as an instance variable.
        '''
        self.data = data

        # vars: Python list. len(vars) = num_selected_vars
        #   String variable names selected from the DataFrame to run PCA on.
        #   num_selected_vars <= num_vars
        self.vars = None

        # A: ndarray. shape=(num_samps, num_selected_vars)
        #   Matrix of data selected for PCA
        self.A = None

        # normalized: boolean.
        #   Whether data matrix (A) is normalized by self.pca
        self.normalized = None

        # A_proj: ndarray. shape=(num_samps, num_pcs_to_keep)
        #   Matrix of PCA projected data
        self.A_proj = None

        # e_vals: ndarray. shape=(num_pcs,)
        #   Full set of eigenvalues (ordered large-to-small)
        self.e_vals = None
        # e_vecs: ndarray. shape=(num_selected_vars, num_pcs)
        #   Full set of eigenvectors, corresponding to eigenvalues ordered large-to-small
        self.e_vecs = None

        # prop_var: Python list. len(prop_var) = num_pcs
        #   Proportion variance accounted for by the PCs (ordered large-to-small)
        self.prop_var = None

        # cum_var: Python list. len(cum_var) = num_pcs
        #   Cumulative proportion variance accounted for by the PCs (ordered large-to-small)
        self.cum_var = None

    def get_prop_var(self):
        '''(No changes should be needed)'''
        return self.prop_var

    def get_cum_var(self):
        '''(No changes should be needed)'''
        return self.cum_var

    def get_eigenvalues(self):
        '''(No changes should be needed)'''
        return self.e_vals

    def get_eigenvectors(self):
        '''(No changes should be needed)'''
        return self.e_vecs

    def covariance_matrix(self, data):
        data_centered = data - np.mean(data, axis=0)
        cov_matrix = np.dot(data_centered.T, data_centered) / (data.shape[0] - 1)
        return cov_matrix

    def compute_prop_var(self, e_vals):
        total_var = np.sum(e_vals)
        prop_var = [eig_val / total_var for eig_val in e_vals]
        self.prop_var = prop_var
        return prop_var

    def compute_cum_var(self, prop_var):
        cum_var = []
        for i in range(len(prop_var)):
            cum_var.append(sum(prop_var[:i+1]))
        self.cum_var = cum_var
        return cum_var

    def pca(self, vars, normalize=False):
        self.vars = vars
        selected_data = self.data[self.vars].values
        if normalize:
            self.normalized = True
            self.norm_info = {"mins": np.min(selected_data, axis=0),
                              "maxs": np.max(selected_data, axis=0)}
            self.A = (selected_data - self.norm_info["mins"]) / (self.norm_info["maxs"] - self.norm_info["mins"])
        else:
            self.A = selected_data
        cov_matrix = self.covariance_matrix(self.A)
        self.e_vals, self.e_vecs = np.linalg.eig(cov_matrix)
        self.prop_var = self.compute_prop_var(self.e_vals)
        self.cum_var = self.compute_cum_var(self.prop_var)
        pass

    def elbow_plot(self, num_pcs_to_keep=None):
        if num_pcs_to_keep is None:
            num_pcs_to_keep = len(self.cum_var)
        else:
            num_pcs_to_keep = min(num_pcs_to_keep, len(self.cum_var))
        plt.plot(range(1, num_pcs_to_keep + 1), self.cum_var[:num_pcs_to_keep], marker='o', markersize=8)
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Variance (%)')
        pass

    def pca_project(self, pcs_to_keep):
        e_vecs_to_keep = self.e_vecs[:, pcs_to_keep]
        pca_proj = self.A.dot(e_vecs_to_keep)
        self.A_proj = pca_proj
        return pca_proj

    def pca_then_project_back(self, top_k):
        pcs_to_keep = np.arange(top_k)
        pca_proj = self.pca_project(pcs_to_keep)
        print(pca_proj)
        data_proj = pca_proj @ self.e_vecs[:, :top_k].T
        if self.normalized:
            data_proj = data_proj * (self.norm_info["maxs"] - self.norm_info["mins"]) + self.norm_info["mins"]
        return data_proj
    
    def pca_project_image(self, image, pcs_to_keep):
        e_vecs_to_keep = self.e_vecs[:, pcs_to_keep]
        image_centered = image - np.mean(self.data, axis=0)
        image_proj = np.dot(image_centered, e_vecs_to_keep)
        return image_proj
