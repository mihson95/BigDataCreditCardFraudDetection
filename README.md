# BigDataCreditCardFraudDetection

Course Project for CS6502 - Applied Big Data and Visualization

Title - Big Data Approach for Credit Card Fraud Detection

Group 11 Submission

Technologies Utilised:
- PySpark
- Google Colab Notebooks
- Microsoft Power BI


## Notebook Contents

The .ipynb Notebook Contains:

- Python Code for performing:
  - Data Loading (ETL Pipeline)
  - Data Summarizing (Plots, Data Summary, Categorical and Numerical Features, Imbalance in Dataset)
  - Data Cleaning (Removing Null Values)
  - Data Transformation (Adding 3 new Features)
  - Data Preparation (Undersampling, Indexing, Hot Encoding, Scaling)
  - Model Training (Logistic Regression, Random Forests, Perceptron)
  - Model Evaluation Metrics including why Perceptron is Preferred. 

[Notebook Link](/Python_Notebook/Big_Data_Project.ipynb)

## Power BI Dashboard

Our Power BI dashboard offers a comprehensive analysis of credit card transactions aimed at detecting fraudulent activities. Through a series of visualizations, we delve into various aspects of transaction data to uncover patterns and anomalies indicative of potential fraud. The dashboard provides insights across multiple dimensions, including:

- [Categorical breakdowns of fraudulent transactions](/Power_BI_ Visualizations _Dashboard/PowerBIDashboardReadme.md#category-fraud) 
- [Trends over time through time series analysis](/Power_BI_ Visualizations _Dashboard/PowerBIDashboardReadme.md#time-series-analysis)
- [Spending patterns across different transaction categories](/Power_BI_ Visualizations _Dashboard/PowerBIDashboardReadme.md#categorical-analysis)
- [Geographical analysis of spending behavior](/Power_BI_ Visualizations _Dashboard/PowerBIDashboardReadme.md#geographical-analysis)
- [Financial insights into spending amounts](/Power_BI_ Visualizations _Dashboard/PowerBIDashboardReadme.md#financial-analysis)
- [Correlations between spending and city populations](/Power_BI_ Visualizations _Dashboard/PowerBIDashboardReadme.md#correlation-analysis)
- [Relational analysis between spending amounts and fraud occurrences](/Power_BI_ Visualizations _Dashboard/PowerBIDashboardReadme.md#relational-analysis)
- [Gender analysis providing compelling insights into spending habits and frequency of fraudulent transactions according to the gender](/Power_BI_ Visualizations _Dashboard/PowerBIDashboardReadme.md#gender-analysis)

[Power BI Dashboard Link](/Power_BI_ Visualizations _Dashboard/PowerBIDashboardReadme.md)

## Key Takeaways

- The incorporation of large-scale data analytics and machine learning tools increased the efficiency of detection and monitoring through our project. 
- Undersampling performs well for model training. 
- The Perceptron classifier gives a marginally higher recall value. Hence utilizing it is beneficial for reducing False Negatives. 
- Visualizations aid in making data-driven decisions and visualizing the dataset. 
- Continuous vigilance and adaptation are essential in combating evolving fraud tactics. 
