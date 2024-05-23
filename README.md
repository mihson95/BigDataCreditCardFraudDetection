# Big Data Approach for Credit Card Fraud Detection

Course Project for CS6502 - Applied Big Data and Visualization

Title - Big Data Approach for Credit Card Fraud Detection

Contributors: <br>Group 11
- Mihir Sontake
- Nithin Skanda
- Jeffrey Forde
- Amar Yandrapu
- Altaf Ahmed


Technologies Utilised:
- PySpark
- Google Colab Notebooks
- Microsoft Power BI

## Overview

This Big Data project focused on credit card fraud detection. It was developed as part of the course "Applied Big Data and Visualizations" (CS6502) at the University of Limerick, taught by Dr. Andrew Ju.

## Description

Credit card fraud is a pervasive issue, escalating in frequency over recent years. Our project endeavours to combat this menace by harnessing the power of big data analytics and machine learning. By scrutinizing transactional data and employing advanced algorithms, we aim to unearth patterns and anomalies indicative of fraudulent activity in real time. Through this initiative, we aspire to fortify the security of financial transactions and curtail the proliferation of fraudulent practices. The output from our Project would be predicting whether a transaction is fraudulent or not fraudulent, based on the historical data which is used to train the machine learning models - Perceptron, Logistic Regression, Random Forest. We utilized PySpark within Notebooks on Google Colab. By harnessing the power of in-memory processing, distributed computing, fault tolerance, and advanced analytics, we aimed to enhance the accuracy and efficiency of fraud detection. Our project’s goal is to safeguard financial institutions and consumers against malicious activities.

## Dataset

[Dataset Link](/Dataset/fraud_prediction_data.csv)

## Notebook Contents

The notebook covers various stages of the project, including data loading, exploratory data analysis (EDA), data preprocessing, model training, and evaluation. It showcases techniques for handling imbalanced datasets, data cleaning, feature engineering, and model evaluation

  - [Overview](/Python_Notebook/Notebook_Readme.md#overview)
  - [Dataset Description](/Python_Notebook/Notebook_Readme.md#dataset-description)
  - [Data Loading (ETL Pipeline)](/Python_Notebook/Notebook_Readme.md#loading-the-dataset)
  - [Data Summarizing (Plots, Data Summary, Categorical and Numerical Features, Imbalance in Dataset)](/Python_Notebook/Notebook_Readme.md#exploratory-data-analysis-eda)
  - [Data Transformation (Adding 3 new Features)](/Python_Notebook/Notebook_Readme.md#creating-new-features)
  - [Data Preparation (Undersampling, Indexing, Hot Encoding, Scaling)](/Python_Notebook/Notebook_Readme.md#string-indexing-hot-encoding-and-vector-assembling-of-features-into-1-vector)
  - Model Training ([Perceptron](/Python_Notebook/Notebook_Readme.md#model-training-perceptron), [Logistic Regression](/Python_Notebook/Notebook_Readme.md#model-training-logistic-regression), [Random Forest](/Python_Notebook/Notebook_Readme.md#model-training-random-forest))
  - [Model Evaluation Metrics including why Perceptron is Preferred](/Python_Notebook/Notebook_Readme.md#conclusion) 

[Notebook Link](/Python_Notebook/Big_Data_Project.ipynb)

[Notebook Readme Link](/Python_Notebook/Notebook_Readme.md)

## Power BI Dashboard

Our Power BI dashboard offers a comprehensive analysis of credit card transactions aimed at detecting fraudulent activities. Through a series of visualizations, we delve into various aspects of transaction data to uncover patterns and anomalies indicative of potential fraud. The dashboard provides insights across multiple dimensions, including:

- [Categorical breakdowns of fraudulent transactions](/Power_BI_Visualizations_Dashboard/PowerBIDashboardReadme.md#category-fraud) 
- [Trends over time through time series analysis](/Power_BI_Visualizations_Dashboard/PowerBIDashboardReadme.md#time-series-analysis)
- [Spending patterns across different transaction categories](/Power_BI_Visualizations_Dashboard/PowerBIDashboardReadme.md#categorical-analysis)
- [Geographical analysis of spending behavior](/Power_BI_Visualizations_Dashboard/PowerBIDashboardReadme.md#geographical-analysis)
- [Financial insights into spending amounts](/Power_BI_Visualizations_Dashboard/PowerBIDashboardReadme.md#financial-analysis)
- [Correlations between spending and city populations](/Power_BI_Visualizations_Dashboard/PowerBIDashboardReadme.md#correlation-analysis)
- [Relational analysis between spending amounts and fraud occurrences](/Power_BI_Visualizations_Dashboard/PowerBIDashboardReadme.md#relational-analysis)
- [Gender analysis providing compelling insights into spending habits and frequency of fraudulent transactions according to the gender](/Power_BI_Visualizations_Dashboard/PowerBIDashboardReadme.md#gender-analysis)

[Power BI Dashboard Link](/Power_BI_Visualizations_Dashboard/PowerBIDashboardReadme.md)

## Key Takeaways

- The incorporation of large-scale data analytics and machine learning tools increased the efficiency of detection and monitoring through our project. 
- Undersampling performs well for model training. 
- The Perceptron classifier gives a marginally higher recall value. Hence utilizing it is beneficial for reducing False Negatives. 
- Visualizations aid in making data-driven decisions and visualizing the dataset. 
- Continuous vigilance and adaptation are essential in combating evolving fraud tactics. 
