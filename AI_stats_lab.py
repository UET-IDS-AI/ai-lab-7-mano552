"""
AIstats_lab.py

Student starter file for:
1. Naive Bayes spam classification
2. K-Nearest Neighbors on Iris
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def accuracy_score(y_true, y_pred):
    """
    Compute classification accuracy.
    """
    return float(np.mean(y_true == y_pred))

# =========================
# Q1 Naive Bayes
# =========================

def naive_bayes_mle_spam():
    
    texts = [
        "win money now",
        "limited offer win cash",
        "cheap meds available",
        "win big prize now",
        "exclusive offer buy now",
        "cheap pills buy cheap meds",
        "win lottery claim prize",
        "urgent offer win money",
        "free cash bonus now",
        "buy meds online cheap",
        "meeting schedule tomorrow",
        "project discussion meeting",
        "please review the report",
        "team meeting agenda today",
        "project deadline discussion",
        "review the project document",
        "schedule a meeting tomorrow",
        "please send the report",
        "discussion on project update",
        "team sync meeting notes"
    ]

    labels = np.array([
        1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0
    ])

    test_email = "win cash prize now"

    # 1. Tokenization
    tokenized_texts = [text.split() for text in texts]

    # 2. Vocabulary
    vocab = set()
    for tokens in tokenized_texts:
        vocab.update(tokens)
    vocab = list(vocab)

    # 3. Class priors
    priors = {}
    total_docs = len(labels)
    for c in [0, 1]:
        priors[c] = np.sum(labels == c) / total_docs

    # 4. Word probabilities (MLE, no smoothing)
    word_probs = {0: {}, 1: {}}

    for c in [0, 1]:
        # collect all words in class c
        words_in_class = []
        for tokens, label in zip(tokenized_texts, labels):
            if label == c:
                words_in_class.extend(tokens)

        total_words = len(words_in_class)

        for word in vocab:
            count = words_in_class.count(word)
            if total_words > 0:
                word_probs[c][word] = count / total_words
            else:
                word_probs[c][word] = 0

    # 5. Prediction
    test_tokens = test_email.split()

    scores = {}

    for c in [0, 1]:
        score = priors[c]
        for word in test_tokens:
            if word in word_probs[c]:
                score *= word_probs[c][word]
            else:
                score *= 0  # unseen word → 0 probability
        scores[c] = score

    prediction = max(scores, key=scores.get)

    return priors, word_probs, prediction


# =========================
# Q2 KNN
# =========================

def knn_iris(k=3, test_size=0.2, seed=0):

    # 1. Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # 2. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # 3. Euclidean distance function
    def euclidean_distance(a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    # 4. KNN prediction function
    def predict(X_train, y_train, x_test, k):
        distances = []

        for i in range(len(X_train)):
            dist = euclidean_distance(X_train[i], x_test)
            distances.append((dist, y_train[i]))

        # sort by distance
        distances.sort(key=lambda x: x[0])

        # get k nearest
        k_neighbors = distances[:k]

        labels = [label for _, label in k_neighbors]

        # majority vote
        values, counts = np.unique(labels, return_counts=True)
        return values[np.argmax(counts)]

    # 5. Predictions
    y_train_pred = np.array([predict(X_train, y_train, x, k) for x in X_train])
    y_test_pred = np.array([predict(X_train, y_train, x, k) for x in X_test])

    # 6. Accuracy
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    return train_accuracy, test_accuracy, y_test_pred
