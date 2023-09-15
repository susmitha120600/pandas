#!/usr/bin/env python
# coding: utf-8

# # IMPORTING LIBRARIES
# 

# In[ ]:


import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename


# # ANALYSING AND READ THE DATA FRAME

# In[2]:


social=pd.read_csv("social media.csv")
social


# In[3]:


import missingno as msno
msno.matrix(social)


# # OVERVIEW OF THE DATASET

# In[4]:


import matplotlib.pyplot as plt
import seaborn as sns

# Overview of the dataset
print(social.head())


# # SUMMARY STATISTICS

# In[5]:


# Summary statistics
print(social.describe())


# # CORRELATION HEAT MAP IS WARMER COLOR REPRESENT POSITIVE CORRELATION &COOLER COLOR REPRESENT NEGATIVE CORRELATION

# In[8]:


# Correlation heatmap
correlation_matrix = correlation_matrix = social.corr(numeric_only=True)
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


# # PLATFORM WISE USER ENGAGEMENT

# In[9]:


# Platform-wise user engagement
plt.figure(figsize=(10, 6))
sns.boxplot(x='Platform', y='User Engagement', data=social, palette='pastel')
plt.title('Platform-wise User Engagement')
plt.xlabel('Platform')
plt.ylabel('User Engagement')
plt.show()


# # DISTRIBUTION OF POST MEDIA TYPES

# In[10]:


# Distribution of post media types
plt.figure(figsize=(8, 6))
sns.countplot(x='Media Type', data=social, palette='bright')
plt.title('Distribution of Post Media Types')
plt.xlabel('Media Type')
plt.ylabel('Count')
plt.show()


# In[ ]:




