# Credit Card Fraud Detection Project

## Overview

This Jupyter Notebook is part of a Big Data project focused on credit card fraud detection. It was developed as part of the course "Applied Big Data and Visualizations" (CS6502) at the University of Limerick, taught by Dr. Andrew Ju. The project was completed by Group 11.

The notebook covers various stages of the project, including data loading, exploratory data analysis (EDA), data preprocessing, model training, and evaluation. It showcases techniques for handling imbalanced datasets, data cleaning, feature engineering, and model evaluation.

## Dataset Description

The dataset used in this project contains 555,719 instances and 22 attributes, consisting of a mix of categorical and numerical data types. It includes information such as transaction details, customer demographics, and merchant information. The target variable, `is_fraud`, indicates whether a transaction is fraudulent (1) or legitimate (0).

## Mounting Folder from Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Install and Import Packages
The notebook installs and imports necessary packages and libraries for the project, including PySpark for distributed computing.

## Creating a Spark Session

```python
import pyspark
from pyspark.sql import SparkSession

spark= SparkSession \
       .builder \
       .appName("Big Data Project") \
       .getOrCreate()

spark
```

## Loading the Dataset

The notebook loads the dataset from Google Drive into a Spark DataFrame for further analysis.

## Exploratory Data Analysis (EDA)

- Displaying the first few rows of the DataFrame
- Viewing the DataFrame schema
- Counting missing values in each column
- Analyzing class imbalance in the dataset
- Visualizing numerical features through histograms, box plots, and pair plots
- Identifying outliers and skewed columns

## Data Preprocessing

- Identifying and transforming categorical columns using String Indexing
- Performing hot encoding for categorical features
- Scaling numerical features

## Creating New Features

```python
from pyspark.sql.functions import to_timestamp, col, dayofweek, when, hour, radians, cos, sin, sqrt, asin, round

# Assuming df is your DataFrame and 'trans_date_trans_time' is the column name
df = df.withColumn("Datetime", to_timestamp("trans_date_trans_time", "dd/MM/yyyy HH:mm"))

df = df.withColumn("DayOfWeek", dayofweek("Datetime"))

df = df.withColumn("is_weekend", when(col("DayOfWeek") >= 6, 1).otherwise(0))

df = df.withColumn("Hour", hour("Datetime"))

df = df.withColumn("time_of_day",
                   when(col("Hour") < 6, "Night")
                   .when(col("Hour") < 12, "Morning")
                   .when(col("Hour") < 18, "Afternoon")
                   .otherwise("Evening"))

def haversine(lon1, lat1, lon2, lat2):
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6371 * c
    return km

df = df.withColumn("distance_km", haversine(df["long"], df["lat"], df["merch_long"], df["merch_lat"]))
df = df.withColumn("distance_km", round(df["distance_km"], 2))

# Droping rows we don't need for the models
df = df.drop('cc_num', 'first', 'last', 'zip', 'trans_num', 'dob','lat','long','city','trans_date_trans_time','gender','street','city_pop','merch_lat','merch_long','Datetime','DayOfWeek','Hour','unix_time','_c0')

# Perform undersampling
fraud_df = df.filter(col("is_fraud") == 1)
fraud_count = fraud_df.count()
non_fraud_df = df.filter(col("is_fraud") == 0)

sampled_non_fraud_df = non_fraud_df.sample(withReplacement=False, fraction=fraud_count / non_fraud_df.count(), seed=42).limit(fraud_count)

balanced_df = fraud_df.unionAll(sampled_non_fraud_df)

# Group by the 'Class' column and count occurrences
class_counts_b = balanced_df.groupBy('is_fraud').count()

# Convert the result to Pandas DataFrame for easy display (optional)
class_counts_b = class_counts_b.toPandas()

# Print the result
print(class_counts_b)
```

## String Indexing, Hot Encoding, and Vector Assembling of Features into 1 Vector

```python
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

numeric_features = ['amt', 'distance_km']
categorical_features = ['merchant', 'category', 'state', 'time_of_day']

indexers = [StringIndexer(inputCol=col, outputCol=col + "_index", handleInvalid="keep") for col in categorical_features]

encoders = [OneHotEncoder(inputCols=[col + "_index"], outputCols=[col + "_encoded"]) for col in categorical_features]

assembler = VectorAssembler(inputCols=[col + "_encoded" for col in categorical_features] + numeric_features,
                            outputCol="raw_features")

scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=False)

transformation_pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler])

transformed_model_b = transformation_pipeline.fit(balanced_df)
b_transformed = transformed_model_b.transform(balanced_df)

train_data, test_data = b_transformed.randomSplit([0.8, 0.2], seed=42)
```

## Model Training: Perceptron

```python
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType

input_layer_size = len(train_data.select("features").first()[0])
output_layer_size = 2  # Binary classification
layers = [input_layer_size, 5, output_layer_size]

mlp = MultilayerPerceptronClassifier(layers=layers, seed=1234, maxIter=100, featuresCol="features", labelCol="is_fraud")

mlp_model = mlp.fit(train_data)

train_predictions_mlp = mlp_model.transform(train_data)
test_predictions_mlp = mlp_model.transform(test_data)

train_predictions_mlp = train_predictions_mlp.withColumn("prediction", col("prediction").cast(DoubleType()))
train_predictions_mlp = train_predictions_mlp.withColumn("is_fraud", col("is_fraud").cast(DoubleType()))
test_predictions_mlp = test_predictions_mlp.withColumn("prediction", col("prediction").cast(DoubleType()))
test_predictions_mlp = test_predictions_mlp.withColumn("is_fraud", col("is_fraud").cast(DoubleType()))

evaluator_auc_mlp = BinaryClassificationEvaluator(labelCol="is_fraud", metricName="areaUnderROC")

train_auc_mlp = evaluator_auc_mlp.evaluate(train_predictions_mlp)
test_auc_mlp = evaluator_auc_mlp.evaluate(test_predictions_mlp)

print(f"MLP Training Set AUC: {train_auc_mlp}")
print(f"MLP Test Set AUC: {test_auc_mlp}")

predictionAndLabelsTrain_mlp = train_predictions_mlp.select("prediction", "is_fraud").rdd.map(lambda row: (float(row[0]), float(row[1])))
predictionAndLabelsTest_mlp = test_predictions_mlp.select("prediction", "is_fraud").rdd.map(lambda row: (float(row[0]), float(row[1])))

metrics_train_mlp = MulticlassMetrics(predictionAndLabelsTrain_mlp)
metrics_test_mlp = MulticlassMetrics(predictionAndLabelsTest_mlp)

precision_train_mlp = metrics_train_mlp.precision(1)
recall_train_mlp = metrics_train_mlp.recall(1)
f1_score_train_mlp = metrics_train_mlp.fMeasure(1.0)
precision_test_mlp = metrics_test_mlp.precision(1)
recall_test_mlp = metrics_test_mlp.recall(1)
f1_score_test_mlp = metrics_test_mlp.fMeasure(1.0)

print("MLP Training Precision:", precision_train_mlp)
print("MLP Training Recall:", recall_train_mlp)
print("MLP Training F1-Score:", f1_score_train_mlp)
print("MLP Testing Precision:", precision_test_mlp)
print("MLP Testing Recall:", recall_test_mlp)
print("MLP Testing F1-Score:", f1_score_test_mlp)
```

## Model Training: Logistic Regression

```python
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType

lr = LogisticRegression(featuresCol="features", labelCol="is_fraud")

lr_model = lr.fit(train_data)

train_predictions = lr_model.transform(train_data)
test_predictions = lr_model.transform(test_data)

train_predictions = train_predictions.withColumn("prediction", col("prediction").cast(DoubleType()))
train_predictions = train_predictions.withColumn("is_fraud", col("is_fraud").cast(DoubleType()))
test_predictions = test_predictions.withColumn("prediction", col("prediction").cast(DoubleType()))
test_predictions = test_predictions.withColumn("is_fraud", col("is_fraud").cast(DoubleType()))

evaluator = BinaryClassificationEvaluator(labelCol="is_fraud", metricName="areaUnderROC")
evaluator2 = MulticlassClassificationEvaluator(
    labelCol="is_fraud",
    predictionCol="prediction",
    metricName="accuracy"
)

precision = evaluator2.evaluate(test_predictions)
print("Model accuracy:", precision)

auc_train = evaluator.evaluate(train_predictions)
auc_test = evaluator.evaluate(test_predictions)

print("Training AUC:", auc_train)
print("Testing AUC:", auc_test)

metrics_train = MulticlassMetrics(train_predictions.select("prediction", "is_fraud").rdd)
metrics_test = MulticlassMetrics(test_predictions.select("prediction", "is_fraud").rdd)

precision_train = metrics_train.precision(1)
recall_train = metrics_train.recall(1)
f1_score_train = metrics_train.fMeasure(1.0)
precision_test = metrics_test.precision(1)
recall_test = metrics_test.recall(1)
f1_score_test = metrics_test.fMeasure(1.0)

print("Training Precision:", precision_train)
print("Training Recall:", recall_train)
print("Training F1-Score:", f1_score_train)
print("Testing Precision:", precision_test)
print("Testing Recall:", recall_test)
print("Testing F1-Score:", f1_score_test)
```

## Model Training: Random Forest

```python
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.functions import col, when
from pyspark.sql.types import DoubleType

rf = RandomForestClassifier(featuresCol="features", labelCol="is_fraud",
                            numTrees=100, maxDepth=10, seed=42)

rf_model = rf.fit(train_data)

train_predictions = rf_model.transform(train_data)
test_predictions = rf_model.transform(test_data)

train_predictions = train_predictions.withColumn("prediction", col("prediction").cast(DoubleType()))
test_predictions = test_predictions.withColumn("prediction", col("prediction").cast(DoubleType()))
train_predictions = train_predictions.withColumn("is_fraud", col("is_fraud").cast(DoubleType()))
test_predictions = test_predictions.withColumn("is_fraud", col("is_fraud").cast(DoubleType()))

evaluator = BinaryClassificationEvaluator(labelCol="is_fraud", metricName="areaUnderROC")
auc_train = evaluator.evaluate(train_predictions)
auc_test = evaluator.evaluate(test_predictions)

print("Training AUC:", auc_train)
print("Testing AUC:", auc_test)

metrics_train = MulticlassMetrics(train_predictions.select("prediction", "is_fraud").rdd)
metrics_test = MulticlassMetrics(test_predictions.select("prediction", "is_fraud").rdd)

precision_train = metrics_train.precision(1)
recall_train = metrics_train.recall(1)
f1_score_train = metrics_train.fMeasure(1.0)
precision_test = metrics_test.precision(1)
recall_test = metrics_test.recall(1)
f1_score_test = metrics_test.fMeasure(1.0)

print("Training Precision:", precision_train)
print("Training Recall:", recall_train)
print("Training F1-Score:", f1_score_train)
print("Testing Precision:", precision_test)
print("Testing Recall:", recall_test)
print("Testing F1-Score:", f1_score_test)
```

## Conclusion

The **Perceptron** was chosen as the final model for our project due to **marginally higher testing recall**, which is *critical in minimizing the risk of missing fraudulent transactions.*

## Notebook Files

- [Big_Data_Project.ipynb](/Python_Notebook/Big_Data_Project.ipynb) : Jupyter Notebook containing the project code.

## Requirements
The notebook requires the following Python libraries:

- pyspark
- matplotlib
- seaborn
- pandas

You can install these libraries using pip:

```
pip install pyspark matplotlib seaborn pandas
```

