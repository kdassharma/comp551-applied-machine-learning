#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

# In[2]:


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
# We will parse it so it is only the car brand, making it a multi-valued discrete feature
DataFrame['car-name'] = DataFrame['car-name'].apply(lambda x: x.split()[0])


# Now we display the first few rows, looks good!
DataFrame.head(10)


# In[3]:


#Our data is clean and ready to use! Let's analyze it a bit
DataFrame.describe()


# In[4]:


# An interesting observation, displacement is not very nicely distributed (also what is it??)
# Also, this leaves out the car name, I'm curious what the most popular brand is and what not (ford)
DataFrame["car-name"].describe()


# In[5]:


# Now lets look at some graphs, I'm curious how some of the feautures plot against MPG
# First lets plot the continuos variables, they all have pretty clear trends, although acceleration
# is slightly less clear
# From a quick glance, these might be logistic rather than linear realtions
DataFrame.plot(kind='scatter',x='displacement',y='mpg',color='red')
DataFrame.plot(kind='scatter',x='horsepower',y='mpg',color='blue')
DataFrame.plot(kind='scatter',x='displacement',y='mpg',color='green')
DataFrame.plot(kind='scatter',x='weight',y='mpg',color='orange')
DataFrame.plot(kind='scatter',x='acceleration',y='mpg',color='cyan')
plt.show()


# In[6]:


# Now I would like to look at the discrete variables
DataFrame.plot(kind='scatter',x='cylinders',y='mpg',color='red')
DataFrame.plot(kind='scatter',x='model-year',y='mpg',color='blue')
DataFrame.plot(kind='scatter',x='origin',y='mpg',color='orange')
DataFrame.plot(kind='scatter',x='car-name',y='mpg',color='green',figsize=(12,4))
plt.xticks(rotation=90)
plt.show()


# In[7]:


# Finally, since the goal is linear classification, we must transform the dependent varible(MPG) into
# a binary choice. I will transform it into high(1) gas mileage or low(0) gas mileage, depending on whether it
# is above or below the mean value of 23.44
DataFrame['mpg'] = DataFrame['mpg'].apply(lambda x: 1 if (x >= 23.445918) else 0)
# As is shows below, the mean is very close to 0.5, so there is a realtively even split between low an high gas
# mileage
DataFrame['mpg'].describe()


# 

# In[8]:


# lets see if the features are correlated


# In[9]:


corr = DataFrame.corr()
sns.heatmap(corr)


# In[ ]:





# In[ ]:




