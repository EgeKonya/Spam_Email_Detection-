# **Spam Email Classification with Random Forest and Naive Bayes**

## Overview
This repository contains two Python implementations of a Spam Email Classification system using machine learning. The models used are:
1. Random Forest Classifier
2. Naive Bayes Classifier

Both approaches leverage bag-of-words representation (using CountVectorizer) for text data and are applied to classify emails as either spam or ham (non-spam). The dataset used is the spam_ham_dataset.csv.

### Importance
Spam emails are a persistent issue in modern communication, ranging from advertisements to phishing attempts. Machine learning provides robust tools for automating spam detection, significantly reducing manual filtering. This repository demonstrates two popular classification techniques to highlight their differences and practical applications in text-based classification problems.

## Dataset
The dataset consists of labeled emails with the following columns:
* text: The email content.
* label_num: A numerical representation of the label (1 for spam, 0 for ham).

Preprocessing steps include:
* Removing punctuation and newline characters.
* Converting text to lowercase.
* Stemming words and removing stopwords.

## Implementation
### Random Forest Classifier
The Random Forest Classifier is an ensemble learning method based on decision trees. It builds multiple decision trees during training and outputs the mode of the classes for classification tasks.

Advantages:
* Highly robust to overfitting.
* Handles non-linear relationships and complex patterns.
* Scales well with high-dimensional data.

### Naive Bayes Classifier
The Naive Bayes Classifier is a probabilistic model based on Bayes' Theorem. It assumes the features are conditionally independent given the class label (a "naive" assumption). The MultinomialNB implementation is particularly suitable for text classification problems.

Advantages:
* Simple and fast to train.
* Effective with sparse data like text features.
* Works well with smaller datasets.

## Key Differences Between the Models
|       **Feature**      |                **Random Forest**               |                      **Naive Bayes**                     |
|:------------------|:-------------------------------------------|:-----------------------------------------------------|
| Approach           | Ensemble learning using decision trees      | Probabilistic approach using Bayes' Theorem           |
| Interpretability   | Less interpretable; relies on ensemble      | Highly interpretable; uses probabilities              |
| Computational Cost | Relatively high (many trees)                | Low (simple mathematical operations)                  |
| Performance        | Excels with complex relationships           | Excels with smaller datasets and independent features |
| Text Data Handling | Works but may overfit without proper tuning | Tailored for bag-of-words/text data                   |

## Results
Both models were evaluated using a train-test split (60-40). Results may vary depending on the dataset and parameter tuning. 

In general:
* Random Forest may achieve higher accuracy with complex datasets.
* Naive Bayes is faster and performs well when feature independence holds.

## Conclusion
This repository demonstrates the application of two powerful machine learning techniques for spam detection. While Random Forest is ideal for complex relationships, Naive Bayes is a simple yet effective solution for text classification tasks. Together, they showcase the trade-offs between complexity and efficiency in machine learning.
