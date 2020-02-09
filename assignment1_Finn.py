#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import csv 


# <h1>Subtasks for Data Set 3 (Auto MPG Data Set):</h1>

# <body>
#     <h2>Step One</h2>
#     
#     - First we must acquire, preprocess, and analyze the data
# </body>

# In[ ]:


# Reading dataset into a dataframe
DataFrame = pd.read_csv('./Auto MPG Data Set/auto-mpg.data', index_col = False, header=None, names=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model-year', 'origin', 'car-name'], delim_whitespace=True)

# As we can see from the .names file, only horsepower has missing values - there are 6 such rows without it.
# I've decided to remove these since horsepower is not categorical, and the instructions say: "You should
# remove any examples with missing or malformed features and note this in your report".

# Get names of indexes for which column horsepower has value '?'
indexNames = DataFrame[ (DataFrame['horsepower'] == '?') ].index
# Delete these row indexes from dataFrame
DataFrame.drop(indexNames , inplace=True)
# Change type of column to float now that missing values are gone
DataFrame =DataFrame.astype({'horsepower': float})


# However, now we see that car name is kind of a useless value as they all are different models.
# We will just drop it
DataFrame.drop(['car-name'] , inplace=True, axis = 1)

# Now we display the first few rows, looks good!
DataFrame.head(10)


# In[ ]:


# However, we have 4 continuous features, which must be coverted to discrete to use naive bayes
# I will do this by assigning quartiles with a value of 1, 2, 3, or 4
bin_labels = [1.0, 2.0, 3.0, 4.0]
DataFrame['displacement'] = pd.qcut(DataFrame['displacement'],q=4,labels=False)
DataFrame['horsepower'] = pd.qcut(DataFrame['horsepower'],q=4,labels=False)
DataFrame['weight'] = pd.qcut(DataFrame['weight'],q=4,labels=False)
DataFrame['acceleration'] = pd.qcut(DataFrame['acceleration'],q=4,labels=False)
DataFrame.head(10)


# In[ ]:


# Finally, since the goal is linear classification, we must transform the dependent varible(MPG) into
# a binary choice. I will transform it into high(1) gas mileage or low(0) gas mileage, depending on whether it
# is above or below the mean value of 23.44
DataFrame['mpg'] = DataFrame['mpg'].apply(lambda x: 1 if (x >= 23.445918) else 0)
DataFrame.head(10)


# 

# In[ ]:


# lets see if the features are correlated


# In[ ]:


corr = DataFrame.corr()
sns.heatmap(corr)


# In[ ]:


# yes, we can see cylinders, horsepower, displacement, and weight are all very correlated,
# so I will drop 3 of the 4 since they are redundant (from my testing this greatly increased accuracy)
DataFrame.drop(['cylinders', 'horsepower', 'displacement'] , inplace=True, axis = 1)
DataFrame.head(10)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




