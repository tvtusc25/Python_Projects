'''naive_bayes_multinomial.py
Naive Bayes classifier with Multinomial likelihood for discrete features
Trey Tuscai
CS 251/2: Data Analysis Visualization
Spring 2023
'''
import numpy as np


class NaiveBayes:
    '''Naive Bayes classifier using Multinomial likeilihoods (discrete data belonging to any
     number of classes)'''
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.class_priors = None
        self.class_likelihoods = None
        pass

    def get_priors(self):
        return self.class_priors

    def get_likelihoods(self):
        return self.class_likelihoods

    def train(self, data, y):
        class_counts = np.bincount(y)
        num_features = data.shape[1]

        self.class_priors = class_counts / np.sum(class_counts)
        self.class_likelihoods = np.zeros((self.num_classes, num_features))

        for c in range(len(set(y))):
            data_c = data[y == c]
            self.class_likelihoods[c, :] = (data_c.sum(axis=0) + 1) / (data_c.sum() + num_features)

        self.class_priors = np.log(self.class_priors)
        self.class_likelihoods = np.log(self.class_likelihoods)
        pass

    def predict(self, data):
        log_class_priors = self.class_priors
        log_class_likelihoods = self.class_likelihoods

        log_likelihoods = np.dot(data, log_class_likelihoods.T)
        log_posteriors = log_likelihoods + log_class_priors
        predictions = np.argmax(log_posteriors, axis=1)
        return predictions

    def accuracy(self, y, y_pred):
        return np.mean(y == y_pred)


    def confusion_matrix(self, y, y_pred):
        conf_mat = np.zeros((self.num_classes, self.num_classes))
        for i in range(len(y)):
            conf_mat[y[i], y_pred[i]] += 1
        return conf_mat
