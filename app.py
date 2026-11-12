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

# prior probabilities for each class
class_priors = {
    "positive": 0.0,
    "negative": 0.0
}

def train_model(train_file="data/train.csv"):
    """
    Train the Naive Bayes classifier on the training dataset.
    """

    # clear before retraining
    class_counts["positive"] = 0
    class_counts["negative"] = 0
    
    vocabulary.clear()

    bag_of_words["positive"].clear()
    bag_of_words["negative"].clear()

    total_words["positive"] = 0
    total_words["negative"] = 0

    # open and read the training CSV dataset
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

        # calculate prior class probabilities
        total_class_count = sum(class_counts.values())
        class_priors["positive"] = class_counts["positive"] / total_class_count
        class_priors["negative"] = class_counts["negative"] / total_class_count

def get_word_likelihood(word, label):
    """
    Calculate word likelihood using Laplacian smoothing: P(w|c) = (count(w, c) + 1) / (N_c + |V|)
        where P(w|c) is the word likelihood in the class,
        count(w, c) is the number of word occurences in the class,
        N_c is the total count of all words in class c,
        |V| is the number of unique words in the training vocabulary.
    Returns: Laplace-smoothed probability
    """

    # count(w, c)
    word_count = bag_of_words[label][word]

    numerator = word_count + 1

    # N_c + |V|
    denominator = total_words[label] + len(vocabulary)

    return numerator / denominator

def predict(text):
    """
    Predict sentiment class for an input sentence.
    Returns: 'positive' or 'negative'
    Tie-breaking: If scores are tied, select the class with the higher prior P(c),
                  or default to 'positive' if priors are equal.
    """

    # tokenize the input sentence
    words = tokenize(text)

    # initialize scores with log class priors
    scores = {
        "positive": math.log(class_priors["positive"]),
        "negative": math.log(class_priors["negative"])
    }

    # calculate the class scores
    for word in words:
        if word in vocabulary:
            for label in scores:
                scores[label] += math.log(get_word_likelihood(word, label))

    # compare scores and select the predicted class with higher score
    if scores["positive"] > scores["negative"]:
        return "positive"
    elif scores["negative"] > scores["positive"]:
        return "negative"

    # if scores are tied, compare class priors
    else:
        if (class_priors["negative"] > class_priors["positive"]):
            return "negative"

        # if class priors are also tied or positive class prior is higher, return "positive"
        else:
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
    """
    confusion_matrix = {
        "TP": 0,
        "TN": 0,
        "FP": 0,
        "FN": 0
    }

    # open and read the testing CSV dataset
    with open(test_file, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            text = row["text"]
            actual_label = row["label"]

            predicted_label = predict(text)

            # compare actual and predicted labels
            if actual_label == predicted_label and actual_label == "positive":
                confusion_matrix["TP"] += 1
            elif actual_label == predicted_label and actual_label == "negative":
                confusion_matrix["TN"] += 1
            elif actual_label == "positive" and predicted_label == "negative":
                confusion_matrix["FN"] += 1
            else:
                confusion_matrix["FP"] += 1

    # calculate total number of predictions and number of correct predictions
    total_predictions = sum(confusion_matrix.values())
    correct_predictions = confusion_matrix["TP"] + confusion_matrix["TN"]

    # calculate accuracy
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
    else:
        accuracy = 0.0

    # return accuracy and confusion matrix as a dict
    result = {
        "accuracy": accuracy,
        "confusion_matrix": confusion_matrix
    }

    return result


if __name__ == "__main__":
    train_model()

    evaluation_result = evaluate_model()
    print(f"Accuracy: {evaluation_result['accuracy'] * 100:.2f}%")
    print("Confusion matrix:", evaluation_result["confusion_matrix"])
