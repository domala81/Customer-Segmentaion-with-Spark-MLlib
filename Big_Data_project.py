#!/usr/bin/env python
# coding: utf-8

# In[1]:


# pip install findspark


# In[2]:


import findspark

# Providing the correct Spark home directory path
spark_home = "/opt/homebrew/Cellar/apache-spark/3.5.1/libexec/"

# Initializing findspark with the Spark home directory path
findspark. init (spark_home)


# In[3]:


import pyspark
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("GouriApp") \
    .getOrCreate()

spark


# In[4]:


#Data Exploration and Understanding


# In[5]:


# Load the dataset
data_path = "/Users/gouridumale/Downloads/HotelCustomersDataset.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

import pandas as pd

# Display the schema and first few rows of the dataframe
df.printSchema()

# Convert PySpark DataFrame to Pandas DataFrame
pandas_df = df.toPandas()

# Display Pandas DataFrame
display(pandas_df)


# Display the first few rows of the dataframe
#df.show(5)


# In[6]:


#Exploring Data Characteristics

# Display summary statistics of numerical columns
df_summary=df.describe()

import pandas as pd

# Convert PySpark DataFrame to Pandas DataFrame
pandas_df_summary = df_summary.toPandas()

# Display Pandas DataFrame
print(pandas_df_summary)


# Count the number of rows in the dataframe
print("Total number of rows:", df.count())

# Count the number of columns in the dataframe
print("Total number of columns:", len(df.columns))


# In[7]:


#Exploring Missing Values

from pyspark.sql.functions import col, count, when

# Check for missing values in each column
missing_counts = df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns])
missing_counts.show()


# In[8]:


# Convert PySpark DataFrame to Pandas DataFrame
pandas_df = df.toPandas()

# Check for missing values in each column
missing_counts = pandas_df.isnull().sum()
print(missing_counts)


# In[9]:


#Visualizing Data Distributions

import matplotlib.pyplot as plt
import seaborn as sns

# Plot histograms for numerical features
numeric_cols = [col[0] for col in df.dtypes if col[1] != 'string']
numeric_df = df.select(numeric_cols).toPandas()

for col in numeric_cols:
    plt.figure(figsize=(8, 6))
    sns.histplot(numeric_df[col], bins=20, kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()


# In[10]:


# Calculate correlation matrix for numerical features
correlation_matrix = df.select(numeric_cols).toPandas().corr()

# Visualize correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()


# In[11]:


#Exploring Categorical Variables

# List categorical columns
categorical_cols = [col[0] for col in df.dtypes if col[1] == 'string']

# Display unique values and their counts for each categorical column
for col in categorical_cols:
    print(f"Column: {col}")
    df.groupBy(col).count().show()


# In[12]:


#DATA PROCESSING


# In[15]:


from pyspark.sql.functions import mean, when

# Replace missing values with appropriate strategy

# Replace missing values in numerical columns with mean
numerical_cols = ['Age', 'DaysSinceCreation', 'AverageLeadTime', 'LodgingRevenue', 'OtherRevenue', 'BookingsCanceled', 'BookingsNoShowed', 'BookingsCheckedIn', 'PersonsNights', 'RoomNights', 'DaysSinceLastStay', 'DaysSinceFirstStay']
for col in numerical_cols:
    mean_value = df.select(mean(col)).collect()[0][0]
    df = df.withColumn(col, when(df[col].isNull(), mean_value).otherwise(df[col]))

# Replace missing values in categorical columns with most frequent value
categorical_cols = ['Nationality', 'NameHash', 'DocIDHash', 'DistributionChannel', 'MarketSegment', 'SRHighFloor', 'SRLowFloor', 'SRAccessibleRoom', 'SRMediumFloor', 'SRBathtub', 'SRShower', 'SRCrib', 'SRKingSizeBed', 'SRTwinBed', 'SRNearElevator', 'SRAwayFromElevator', 'SRNoAlcoholInMiniBar', 'SRQuietRoom']
for col in categorical_cols:
    mode_value = df.groupBy(col).count().orderBy('count', ascending=False).first()[0]
    df = df.withColumn(col, when(df[col].isNull(), mode_value).otherwise(df[col]))



# In[16]:


from pyspark.ml.feature import StringIndexer

# Define the columns to be indexed
columns_to_index = ['Nationality']

# Drop the existing 'Nationality_index' column if it exists
if 'Nationality_index' in df.columns:
    df = df.drop('Nationality_index')

# Create StringIndexer objects for each column
indexers = [StringIndexer(inputCol=col, outputCol=col+"_index").fit(df) for col in columns_to_index]

# Apply StringIndexer transformation to each column
for indexer in indexers:
    df = indexer.transform(df)



# In[ ]:





# In[ ]:




