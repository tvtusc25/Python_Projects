'''transformation.py
Perform projections, translations, rotations, and scaling operations on Numpy ndarray data.
Trey Tuscai
CS 251 Data Analysis Visualization
Spring 2023
'''
import numpy as np
import matplotlib.pyplot as plt
import palettable
import analysis
import data


class Transformation(analysis.Analysis):

    def __init__(self, orig_dataset, data=None):
        super().__init__(data)
        self.orig_dataset = orig_dataset
        pass

    def project(self, headers):
        projected_data = self.orig_dataset.select_data(headers)
        projected_header2col = {}
        for i, header in enumerate(headers):
                projected_header2col[header.strip()] = i
        self.data = data.Data(data=projected_data, headers=headers, header2col=projected_header2col)
        pass

    def get_data_homogeneous(self):
        proj_data = self.data.get_all_data()
        ones = np.ones((proj_data.shape[0], 1))
        return np.hstack((proj_data, ones))

    def translation_matrix(self, magnitudes):
        num_proj_vars = self.data.get_num_dims()
        translation_mat = np.eye(num_proj_vars+1)
        translation_mat[:num_proj_vars, num_proj_vars] = magnitudes
        return translation_mat

    def scale_matrix(self, magnitudes):
        num_proj_vars = self.data.get_num_dims()
        scaling_matrix = np.eye(num_proj_vars + 1)
        scaling_matrix[np.arange(num_proj_vars), np.arange(num_proj_vars)] = [magnitudes]
        return scaling_matrix

    def translate(self, magnitudes):
        translation_mat = self.translation_matrix(magnitudes)
        proj_data_homogeneous = self.get_data_homogeneous()
        translated_data = np.dot(translation_mat, proj_data_homogeneous.T).T[:, :-1]
        self.data = data.Data(data=translated_data, headers=self.data.get_headers(), header2col=self.data.get_mappings())
        return translated_data

    def scale(self, magnitudes):
        scale_mat = self.scale_matrix(magnitudes)
        proj_data_homogeneous = self.get_data_homogeneous()
        scaled_data = np.dot(scale_mat, proj_data_homogeneous.T).T[:, :-1]
        self.data = data.Data(data=scaled_data, headers=self.data.get_headers(), header2col=self.data.get_mappings())
        return scaled_data

    def transform(self, C):
        proj_data_homogeneous = self.get_data_homogeneous()
        transformed_data = np.dot(proj_data_homogeneous, C.T)[:, :-1]
        self.data = data.Data(data=transformed_data, headers=self.data.get_headers(), header2col=self.data.get_mappings())
        return transformed_data

    def normalize_together(self):
        global_min = np.min(self.data.get_all_data(), axis = None)
        global_max = np.max(self.data.get_all_data(), axis = None)
        translation_mat= self.translation_matrix(-global_min)
        scale_mat = self.scale_matrix(1 / (global_max - global_min))
        C = np.dot(scale_mat, translation_mat)
        normalized_together_data = self.transform(C)
        self.data = data.Data(data=normalized_together_data, headers=self.data.get_headers(), header2col=self.data.get_mappings())
        return normalized_together_data

    def normalize_separately(self):
        global_min = self.min(self.data.get_headers())
        global_max = self.max(self.data.get_headers())
        translation_mat = self.translation_matrix(-global_min)
        scale_mat = self.scale_matrix(1 / (global_max - global_min))
        C = np.dot(scale_mat, translation_mat)
        normalized_separately_data = self.transform(C)
        self.data = data.Data(data=normalized_separately_data, headers=self.data.get_headers(), header2col=self.data.get_mappings())
        return normalized_separately_data

    def rotation_matrix_3d(self, header, degrees):
        theta = np.deg2rad(degrees)
        if self.data.get_header_indices([header]) == [0]: 
            rotation_matrix = np.array([[1, 0, 0, 0],[0, np.cos(theta), -np.sin(theta), 0],[0, np.sin(theta), np.cos(theta), 0], [0, 0, 0, 1]])
        elif self.data.get_header_indices([header]) == [1]: 
            rotation_matrix = np.array([[np.cos(theta), 0, np.sin(theta), 0],[0, 1, 0, 0],[-np.sin(theta), 0, np.cos(theta), 0], [0, 0, 0, 1]])
        else:
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0, 0],[np.sin(theta), np.cos(theta), 0, 0],[0, 0, 1, 0], [0, 0, 0, 1]])
        return rotation_matrix

    def rotate_3d(self, header, degrees):
        rotated_mat = self.rotation_matrix_3d(header, degrees)
        proj_data_homogeneous = self.get_data_homogeneous()
        rotated_data = np.dot(rotated_mat, proj_data_homogeneous.T).T[:, :-1]
        self.data = data.Data(data=rotated_data, headers=self.data.get_headers(), header2col=self.data.get_mappings())
        return rotated_data

    def scatter_color(self, ind_var, dep_var, c_var, title=None):
        x = self.data.select_data([ind_var]).ravel()
        y = self.data.select_data([dep_var]).ravel()
        z = self.data.select_data([c_var]).ravel()
        color_map = palettable.colorbrewer.sequential.Greys_9
        fig, ax = plt.subplots(figsize = (7,7))
        scatter = ax.scatter(x, y, c=z, s=75, cmap=color_map.mpl_colormap, edgecolors= 'grey')
        ax.set_xlabel(ind_var)
        ax.set_ylabel(dep_var)
        if title:
            ax.set_title(title)
        cbar = fig.colorbar(scatter)
        cbar.ax.set_ylabel(c_var)
        pass

    def scatter_4d(self, ind_var, dep_var, c_var, size_var, title=None):
        x = self.data.select_data([ind_var]).ravel()
        y = self.data.select_data([dep_var]).ravel()
        z = self.data.select_data([c_var]).ravel()
        s = self.data.select_data([size_var]).ravel()
        color_map = palettable.colorbrewer.sequential.Greys_9
        fig, ax = plt.subplots(figsize=(7, 7))
        scatter = ax.scatter(x, y, c=z, s=s*10, cmap=color_map.mpl_colormap, edgecolors='grey')    
        ax.set_xlabel(ind_var)
        ax.set_ylabel(dep_var)
        if title:
            ax.set_title(title)          
        cbar = fig.colorbar(scatter)
        cbar.ax.set_ylabel(c_var)
        sizes = [5, 10, 20, 30]
        labels = [f"{s}" for s in sizes]
        markers = [ax.scatter([], [], s=s, c='k', edgecolors='grey') for s in sizes]
        ax.legend(markers, labels, scatterpoints=1, title=size_var, loc='upper right', fontsize = 10)

    def normalize_zscore(self):
        mean = self.mean(self.data.get_headers())
        std = self.std(self.data.get_headers())
        normalized_together_data = (self.data.get_all_data() - mean) / std
        self.data = data.Data(data=normalized_together_data, headers=self.data.get_headers(), header2col=self.data.get_mappings())
        return normalized_together_data
