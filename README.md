# Breast Cancer Classification Projects

This repository contains two machine learning classification exercises using the Breast Cancer Wisconsin Diagnostic dataset from the UCI Machine Learning Repository.

The goal of both exercises is to classify breast cancer samples as either:

- `B` = Benign
- `M` = Malignant

The same dataset is used for both projects so the performance and behavior of different machine learning models can be compared.

---

## Dataset

**Dataset:** Breast Cancer Wisconsin Diagnostic Dataset  
**Source:** UCI Machine Learning Repository  
**UCI ID:** 17

The dataset contains 569 samples and 30 numerical input features describing characteristics of cell nuclei.

Some of the features include:

- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Concavity
- Concave points
- Symmetry
- Fractal dimension

The target variable is the diagnosis:

- Benign
- Malignant

---

# Project 1: Decision Tree and Random Forest

The first project uses two tree-based classification models:

1. Decision Tree Classifier
2. Random Forest Classifier

The dataset is divided into:

- 80% training data
- 20% testing data

A fixed `random_state=67` is used so the same training and testing samples are produced each time the program is run.

`stratify=y` is also used to keep approximately the same ratio of benign and malignant samples in both datasets.

## Decision Tree

The Decision Tree learns a sequence of rules that split the dataset based on different features.

For example, the first split in the generated tree was:

```text
perimeter3 <= 113.75