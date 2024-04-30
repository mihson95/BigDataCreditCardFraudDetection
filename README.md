# BigDataCreditCardFraudDetection

Course Project for CS6502 - Applied Big Data and Visualization

Title - Big Data Approach for Credit Card Fraud Detection

Group 11 Submission

Technologies Utilised:
- PySpark
- Google Colab Notebooks
- Microsoft Power BI

## Dataset

[Dataset Link](/Dataset/fraud_prediction_data.csv)

## Notebook Contents

The .ipynb Notebook Contains:

- Python Code for performing:
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
