"""
UIC CS 411 - Project 03: Naive Bayes Text Classifier from Scratch
"""

import os
import csv
import math
import re

from collections import Counter

def tokenize(text):
    text = text.lower()

    # normalize Ukrainian apostrophes
    text = text.replace("’", "'").replace("ʼ", "'")

    # split text into words, removing punctuation
    words = re.findall(r"[а-яіїєґa-z0-9]+(?:'[а-яіїєґa-z0-9]+)*", text)

    return words

# count the number of sentences belonging to positive or negative class in the dataset
class_counts = {
    "positive": 0,
    "negative": 0
}

# vocabulary that contains all words in the dataset
vocabulary = set()

# bag of words that counts word frequencies for each class
bag_of_words = {
    "positive": Counter(),
    "negative": Counter()
}

# total number of word occurences in each class, including repeated words
total_words = {
    "positive": 0,
    "negative": 0
}

def train_model(train_file="data/train.csv"):
    """
    Train the Naive Bayes classifier on the training dataset.
    TODO: Implement your preprocessing, vocabulary extraction,
          class priors, and word likelihoods with Laplace smoothing.
    """

    # clear before retraining
    class_counts["positive"] = 0
    class_counts["negative"] = 0
    
    vocabulary.clear()

    bag_of_words["positive"].clear()
    bag_of_words["negative"].clear()

    total_words["positive"] = 0
    total_words["negative"] = 0

    # open and read the CSV dataset
    with open(train_file, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames != ["text", "label"]:
            raise ValueError("CSV must have the columns: text,label")

        # read each row and separate into text and label
        for row in reader:
            text = row["text"]
            label = row["label"]

            # update count for positive or negative class
            class_counts[label] += 1

            # tokenize the text
            words = tokenize(text)

            # add tokens to the bag of words
            bag_of_words[label].update(words)

            total_words[label] += len(words)

            # add unique tokens to the vocabulary set
            vocabulary.update(words)        


def predict(text):
    """
    Predict sentiment class for an input sentence.
    Returns: 'positive' or 'negative'
    Tie-breaking: If scores are tied, select the class with the higher prior P(c),
                  or default to 'positive' if priors are equal.
    TODO: Implement your log-score calculation and class prediction.
    """
    return "positive"


def evaluate_model(test_file="data/test.csv"):
    """
    Evaluate classifier performance on a test dataset.
    Returns:
        dict: A dictionary containing at least:
            {
                "accuracy": float,
                "confusion_matrix": {"TP": int, "TN": int, "FP": int, "FN": int}
            }
    TODO: Implement your evaluation logic (accuracy, confusion matrix).
    """
    pass


if __name__ == "__main__":
    train_model()
