This repository contains work done during my Fundamentals to Machine Learning course. Each Jupyter Notebook represents a different project.

# 1. Data Preprocessing
This notebook showcases data preprocessing techniques to clean, transform, and analyze a raw dataset.  
Dataset used: Top 1000 Movies from IMDb.

## Data Preprocessing Pipeline
Four-step data preprocessing pipeline:
1. Data Cleaning
  - Handling missing values using different strategies.
  - Detecting and addressing outliers.
  - Identifying and fixing inconsistencies in categorical and numerical data.
2. Data Transformation
  - Normalizing or standardizing numerical features where necessary.
  - Encoding categorical variables using appropriate encoding techniques (one-hot, label encoding, etc.).
  - Feature scaling for different scenarios.
3. Data Reduction
  - Applying dimensionality reduction techniques.  
  - Removing redundant or irrelevant features based on logical reasoning.  
  - Justifying how data reduction improves performance and interpretability.
4. Feature Engineering
  - Creating new meaningful features using transformations, aggregations, or domain knowledge.  
  -  Using statistical methods (correlation analysis, feature importance) to validate the impact of the new features.  
  - Explaining the rationale behind feature engineering choices.

## Extracting & Visualizing Insights
10 inshights based on the observed data, justified by different vizualisation methods. This includes:
- Distributions of data.
- Correlation between features.
- Rankings based on numerical data.

## Movie Recommendation System (Without AI)
Reccomendation algorithm using **cosine similarity**.  
- Selection of data for the feature matrix.
- Algorithm implementation movie selection based on similarity scores.
- Testing for different movies in the dataset.

# 2. Regression Models
This notebook showcases regression techniques to predict a target variable using various classical AI models.  
Dataset used: Top 1000 Movies from IMDb.

## Feature Selection & Engineering
- Selecting the Target Variable or Dependent variable based on available features.
- Selecting relevant features using statistical methods (correlation matrix, feature importance)
- Applying additional transformations, if necessary, to improve predictive performance.

## Training Regression Models
Three regression models are considered:
1. Linear Regression
2. Decision Tree Regression
3. Random Forest Regression
  
For each regression model:
- Splitting dataset into training and test sets.
- Training it using the selected features.
- Optimizing hyperparameters where applicable.
- Performance evaluation using relevant metrics (e.g., RMSE, MAE, R-squared).
- Extracting insights in terms of feature importance.
- Discussing potential improvements.
  
The regression models are compared in terms of performance using visialization methods.

## Bonus challenge
- Cross Validation
- Ensemble techniques

# 3. Clustering & Classification Techniques
This notebook showcases both unsupervised and supervised clustering techniques used to cluster and classify data entries based on relevant features.  
Dataset used: Top 1000 Movies from IMDb.

## Clustering with K-Means (Unsupervised Learning)
- Analyzing clustering by group of logical features.  
- Preprocessing data: handling missing values, normalizing features.  
- Determining the optimal number of clusters (Elbow Method, Silhouette Score).  
- Fiting the K-Means algorithm and labeling each movie with its cluster.
- Visualizing the clusters in 2D using dimensionality reduction.  
- Interpreting cluster profiles.

## Classification with Logistic Regression (Supervised Learning)
- Creating a binary target variable.
- Choosing appropriate input features (including feature engineering).
- Splitting dataset into training and test sets.
- Training a Logistic Regression model.
- Evaluating using accuracy, confusion matrix, precision, recall, and f1-score.
- Interpreting the coefficients of the model (feature importance).

## Bonus challenge
One extra clustering and one extra classification algorithm are analysed using the same techniques as in the steps above.
- Clustering with DBSCAN (Unsupervised Learning)
- Classification with Support Vector Machines (Supervised Learning)
