'''email_preprocessor.py
Preprocess Enron email dataset into features for use in supervised learning algorithms
Trey Tuscai
CS 251/2 Data Analysis Visualization
Spring 2023
'''
import re
import os
import numpy as np


def tokenize_words(text):
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text.lower())


def count_words(email_path='data/enron'):
    word_freq = {}
    num_emails = 0

    for root, _, files in os.walk(email_path):
        for file in files:
            if file.endswith(".txt"):
                with open(os.path.join(root, file), 'r') as f:
                    text = f.read()
                    words = tokenize_words(text)
                    for word in words:
                        if word not in word_freq:
                            word_freq[word] = 0
                        word_freq[word] += 1
                num_emails += 1
    return word_freq, num_emails


def find_top_words(word_freq, num_features=200):
    sorted_words = sorted(word_freq.items(), key=lambda x:x[1], reverse=True)
    top_words = [words[0] for words in sorted_words[:num_features]]
    counts = [words[1] for words in sorted_words[:num_features]]
    return top_words, counts


def make_feature_vectors(top_words, num_emails, email_path='data/enron'):
    feats = np.zeros((num_emails, len(top_words)), dtype=int)
    y = np.zeros(num_emails, dtype=int)
    email_index = 0
    for root, _, files in os.walk(email_path):
        for file in files:
            if file.endswith(".txt"):
                with open(os.path.join(root, file), 'r') as f:
                    text = f.read()
                    words = tokenize_words(text)
                    for word in words:
                        if word in top_words:
                            word_index = top_words.index(word)
                            feats[email_index][word_index] += 1
                y[email_index] = 1 if 'spam' in root else 0
                email_index += 1
    return feats, y


def make_train_test_sets(features, y, test_prop=0.2, shuffle=True):
    inds = np.arange(y.size)
    if shuffle:
        features = features.copy()
        y = y.copy()

        inds = np.arange(y.size)
        np.random.shuffle(inds)
        features = features[inds]
        y = y[inds]

    num_test_samples = int(np.ceil(features.shape[0] * test_prop))
    x_train, x_test = np.split(features, [-num_test_samples])
    y_train, y_test = np.split(y, [-num_test_samples])
    inds_train, inds_test = np.split(inds, [-num_test_samples])
    return x_train, y_train, inds_train, x_test, y_test, inds_test



def retrieve_emails(inds, email_path='data/enron'):
    emails = []
    for root, _, files in os.walk(email_path):
        for file in files:
            if file.endswith(".txt"):
                if int(file.split('.')[0]) in inds:
                    with open(os.path.join(root, file), 'r') as f:
                        text = f.read()
                        emails.append(text)
    return emails
