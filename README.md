# Predicting the 2020 US Election from Twitter Sentiment

This repository contains the code and selected outputs for the data mining project:

**Predicting the 2020 US Election from Tweet Sentiment Data**  
Cem Ergin (with Alexander Lerche and Juri Rüegger),  
IT University of Copenhagen

The project investigates whether sentiment expressed on Twitter can be used to predict U.S. presidential election outcomes at the state level, and explores which political topics dominated online discourse during the 2020 election.

---

## Project motivation

Social media platforms play a central role in modern political communication, shaping public discourse and offering real-time insight into voter sentiment. Twitter, in particular, became a major arena for political discussion during the 2020 U.S. presidential election.

This project explores whether large-scale Twitter sentiment data can be leveraged to:
- Approximate voter preferences at the state level
- Predict election outcomes using machine learning techniques
- Identify key topics discussed by supporters of each candidate

The work also critically examines the limitations and biases inherent in social-media-based political analysis.

---

## Research questions

1. How well can Twitter sentiment predict state-level election results in the 2020 U.S. presidential election?
2. Which topics dominated Twitter discussions for each candidate, and how do these topics vary across states?
3. To what extent do identified topics correlate with voting outcomes?

---

## Method overview

The analysis combines sentiment analysis, machine learning classification, and unsupervised topic modeling.

### Sentiment-based party identification

- A neural network is trained using a labeled Twitter sentiment dataset
- Tweets are classified into negative, neutral, or positive sentiment
- Tweets are pre-filtered by candidate-specific hashtags (Biden / Trump)
- Sentiment polarity is used as a heuristic to infer likely party affiliation

State-level election outcomes are predicted by aggregating inferred party support across tweets originating from each state.

### Supervised classification

To improve robustness, a K-Nearest Neighbors (KNN) classifier is trained to predict the winning party in swing states using features including:
- Predicted tweet sentiment
- Engagement metrics (likes, retweets)
- User follower counts
- Candidate hashtag indicators

### Topic identification

- Tweets are embedded using Sentence-BERT
- Pro-Biden and pro-Trump tweets are clustered separately using k-means
- Topic structure is explored using dimensionality reduction (t-SNE)
- Word clouds are generated to interpret dominant themes within each cluster

Topic prevalence is analyzed at the state level and compared with election outcomes.

---

## Data pipeline

The project follows a multi-stage data processing pipeline:

### Step 1: Data preprocessing

- Cleaning and filtering of tweet metadata
- Removal of non-essential columns
- Normalization of numerical features
- Text preprocessing using NLTK (lowercasing, stopword removal, tokenization)
- Filtering tweets to those with valid U.S. state information

### Step 2: Sentiment modeling

- Training a neural network using Keras (TensorFlow)
- Evaluation using held-out test data and sanity-check examples
- Application of the model to election-related tweets

### Step 3: Prediction and analysis

- Aggregation of sentiment signals at the state level
- Prediction of state-level election outcomes
- Evaluation against actual 2020 results and polling data
- Topic clustering and correlation analysis

---

## Key findings (summary)

- Twitter sentiment correctly predicted 6 out of 7 major swing states.
- Overall performance across all states was moderate (30/51 states correctly predicted).
- A KNN classifier achieved 72% accuracy when predicting swing-state outcomes.
- Topic modeling revealed clear thematic differences between candidate supporters:
  - Pro-Trump discussions focused more on issues such as COVID-19, corruption, and elections.
  - Pro-Biden discussions were more general and sentiment-driven.
- Topic-level correlations with election outcomes were modest, highlighting the limits of social-media-based prediction.

---

## Limitations and bias

Twitter users do not represent the broader electorate. Younger, urban, and politically active populations are overrepresented, introducing systematic bias.

Additionally, sentiment-based heuristics oversimplify complex voting behavior, and event-driven sentiment fluctuations can distort predictions.

These limitations are discussed in detail in the accompanying report.

---

## Data and reproducibility

Due to size and licensing constraints, raw Twitter datasets and large intermediate files are not included in this repository.

The repository focuses on:
- Data processing logic
- Modeling approach
- Analysis methodology
- Visual outputs

The full methodology and results are documented in the included project report.
