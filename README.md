# Latent Semantic Indexing (LSI) using Singular Value Decomposition

## Overview

This project implements **Latent Semantic Indexing (LSI)** using **Singular Value Decomposition (SVD)** for document analysis and dimensionality reduction. The objective is to transform a high-dimensional **Bag-of-Words (BoW)** representation into a lower-dimensional latent semantic space, enabling more meaningful document and word representations.

The project combines theoretical concepts with practical implementation, covering text preprocessing, feature extraction, dimensionality reduction, semantic analysis, similarity measurement, and document classification.

---

## Features

- Text preprocessing
  - Lowercase conversion
  - Punctuation removal
- Exploratory text analysis
  - Top 30 most frequent words
  - Word Cloud visualization
- Bag-of-Words (BoW) construction
- TF-IDF concept analysis
- Full Singular Value Decomposition (SVD)
- Truncated SVD
- Randomized SVD implementation from scratch
- Reconstruction error analysis
- Elbow method for selecting latent dimensions
- Semantic interpretation of latent components
- Word similarity analysis
  - Cosine Similarity
  - Euclidean Distance
- Latent-space document representation
- Heatmap visualization of document categories
- Document classification
- Model evaluation on a held-out test set

---

## Project Workflow

1. Text preprocessing
2. Exploratory data analysis
3. Bag-of-Words generation
4. Train/Test split
5. Feature standardization
6. Full SVD computation
7. Truncated SVD
8. Randomized SVD
9. Latent semantic interpretation
10. Word similarity analysis
11. Document classification
12. Model evaluation

---

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- WordCloud

---

## Dataset

The project uses a news dataset containing **2,225 documents**.

- **Training Set:** 2,000 documents
- **Test Set:** 225 documents

A predefined vocabulary (`words.csv`) is used to construct the Bag-of-Words matrix.

---

## Repository Structure

```text
.
├── dataset.csv
├── words.csv
├── notebooks/
│   └── LSI_Project.ipynb
├── src/
│   ├── preprocessing.py
│   ├── svd.py
│   ├── randomized_svd.py
│   ├── similarity.py
│   └── classifier.py
├── figures/
│   ├── wordcloud.png
│   ├── elbow_plot.png
│   ├── heatmap.png
│   └── similarity_results.png
├── report/
│   └── Project_Report.pdf
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Concepts Covered

- Latent Semantic Indexing (LSI)
- Singular Value Decomposition (SVD)
- Truncated SVD
- Randomized SVD
- TF-IDF
- Bag of Words (BoW)
- Cosine Similarity
- Euclidean Distance
- Reconstruction Error
- Elbow Method
- Dimensionality Reduction
- Information Retrieval
- Natural Language Processing (NLP)

---

## Results

This project demonstrates how dimensionality reduction enables a more meaningful semantic representation of documents compared to the original Bag-of-Words space while reducing computational complexity.

It also compares the performance of **Truncated SVD** and **Randomized SVD** in terms of reconstruction error and scalability, highlighting the advantages of randomized algorithms for large-scale text datasets.

---

## Learning Outcomes

By completing this project, the following concepts and techniques were explored:

- Mathematical foundations of Singular Value Decomposition
- Latent semantic representation of documents
- Large-scale matrix factorization
- Text preprocessing and feature engineering
- Information retrieval techniques
- Similarity measurement in vector spaces
- Dimensionality reduction
- Document classification in latent semantic space

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/LSI-Document-Analysis.git
cd LSI-Document-Analysis
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the notebook:

```bash
jupyter notebook notebooks/LSI_Project.ipynb
```

Or execute the Python scripts inside the `src/` directory.

---

## Future Improvements

- Support TF-IDF feature extraction in addition to Bag-of-Words.
- Integrate modern embedding methods (Word2Vec, FastText, BERT) for comparison.
- Develop an interactive visualization dashboard.
- Evaluate additional classification algorithms.

---

## Author

Developed as a **Linear Algebra** course project focusing on **Latent Semantic Indexing (LSI)**, **Singular Value Decomposition (SVD)**, and **Natural Language Processing (NLP)**.
