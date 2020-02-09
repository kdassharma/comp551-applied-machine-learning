#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import csv 
import seaborn as sns


# ## Subtasks for Data Set 2 (Adult Data Set):
# 

# In[6]:


# Reading dataset and printing missing value information

df = pd.read_csv('./Adult Data Set/adult.data', index_col = False, header=None, names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country','salary'])

# Dropping education column as education-num is representative of the same thing
df.drop(['education'], axis = 1,inplace=True)

# The continuous variable fnlwgt represents final weight, which is the number of 
# units in the target population that the responding unit represents. Irrelevant as 
# a feature, and hence being dropped.
df.drop(['fnlwgt'], axis = 1, inplace=True)

# df.drop(['workclass'], axis = 1, inplace=True)
# df.drop(['marital-status'], axis = 1, inplace=True)
# df.drop(['occupation'], axis = 1, inplace=True)
# df.drop(['relationship'], axis = 1, inplace=True)
# df.drop(['race'], axis = 1, inplace=True)
# df.drop(['sex'], axis = 1, inplace=True)
# df.drop(['native-country'], axis = 1, inplace=True)

df['salary'] = df['salary'].apply(lambda x: 0 if (x == ' <=50K' ) else 1)
# Replacing the missing values with NaNs and finding missing values
df.replace(' ?', np.nan, inplace=True)
#missing_values_table(df)
print("-----")

# Replacing missing values with most frequent value in that column
#df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))


# Finding the distribution of the positive and negative classes and numerial features 
# print(df['salary'].value_counts(normalize=True,dropna=True,ascending=True))
# print("-----")
# print(df['age'].value_counts(normalize=True,dropna=True,ascending=True))
# print("-----")
# print(df['education-num'].value_counts(normalize=True,dropna=True,ascending=True))
# print("-----")
# print(df['capital-gain'].value_counts(normalize=True,dropna=True,ascending=True))
# print("-----")
# print(df['capital-loss'].value_counts(normalize=True,dropna=True,ascending=True))
# print("-----")
# print(df['hours-per-week'].value_counts(normalize=True,dropna=True,ascending=True))
# print("-----")

# print(df.describe())
# Dropping these features because data is malformed
df.drop(['capital-gain'], axis = 1, inplace=True)
df.drop(['capital-loss'], axis = 1, inplace=True)

# One hot encoding using pandas get_dummies and then dropping 
# one category for each categorial feature to preserve linear dependency (https://datascience.stackexchange.com/questions/27957/why-do-we-need-to-discard-one-dummy-variable/27993#27993)
df['workclass'] = df['workclass'].astype('category')
df['marital-status'] = df['marital-status'].astype('category')
df['occupation'] = df['occupation'].astype('category')
df['relationship'] = df['relationship'].astype('category')
df['race'] = df['race'].astype('category')
df['sex'] = df['sex'].astype('category')
df['native-country'] = df['native-country'].astype('category')
df = pd.get_dummies(df , drop_first= True)

df = df.apply(lambda x: x.fillna(x.mean()), axis=0)
print("-----")

# Plotting the correlation between features
corr = df.corr()
sns.heatmap(corr)
print("-----")


# In[ ]:


def missing_values_table(df):
   
    # Total missing values
    mis_val = df.isnull().sum()
    
    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    
    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    
    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    
    # Sort the table by percentage of missing descending
    # .iloc[:, 1]!= 0: filter on missing missing values not equal to zero
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(2)  # round(2), keep 2 digits
    
    # Print some summary information
    print("Your selected dataframe has {} columns.".format(df.shape[1]) + '\n' + 
    "There are {} columns that have missing values.".format(mis_val_table_ren_columns.shape[0]))
    
    # Return the dataframe with missing information
    return mis_val_table_ren_columns


# In[ ]:





# 
