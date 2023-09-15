#!/usr/bin/env python
# coding: utf-8

# # Movie Visualization &Recommendation System
# 
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objects as go
init_notebook_mode(connected=True)

import warnings
warnings.filterwarnings("ignore")


# # Exploratory Data Analysis and Data Cleaning

# In[2]:


movie = pd.read_csv('movie recommendation.csv')
movie.head()


# In[3]:


import pandasql as psql # using sql for some exploration

Q1 = """  
    with temp as (select cast(replace(VOTES,",","") as integer) as VOTES from movie)
    
    select MOVIES, YEAR, GENRE,RATING, cast(replace(VOTES,",","") as integer) as VOTES
    from movie 
    where RATING >= (select avg(RATING) as avr_rating from movie)
    and VOTES >= (select avg(VOTES) from temp)
    order by RATING desc, VOTES desc
    limit 20
    """
psql.sqldf(Q1)


# In[4]:


Q2 = """
     select GENRE, max(RATING) as highest_rate
     from movie  
     group by GENRE
     order by highest_rate desc
     limit 10
     """
psql.sqldf(Q2)


# # Checking for missing values

# In[5]:


# Checking for Missing Values
movie.isna().sum()


# In[6]:


print("Missing Values:\n")
for col in movie.columns:
    missing = movie[col].isna().sum()
    percent = missing / movie.shape[0] * 100
    print("%s: %.2f%% (%d)" % (col,percent,missing))


# # Cleaning some of the features

# In[7]:


for col in ['GENRE','ONE-LINE',]:
    movie[col] = movie[col].str.replace("\n","").str.strip()

movie.head()


# In[ ]:





# # Data Visualization

# In[8]:


movie['Year'] = movie['YEAR'].str.extract(r'([0-9]{4}–.*|[0-9]{4})')
movie['Year'] = movie['Year'].str.strip().replace(")","")

def extract_year(year):
    if year[-3:] == '– )':
        return year.replace('– )',"–")
    else:
        return year.replace(')',"")

movie['Year'] = movie['Year'].fillna('Unknown')
movie['Year'] = movie['Year'].apply(lambda y: extract_year(y))
    
year_count = movie[movie['Year'] != 'Unknown']['Year'].value_counts().reset_index().rename(columns = {'Year':'Count','index':'Year'})
year_count.head()


# # Years of distribution

# In[9]:


colors = ['paleturquoise'] * 10
colors[0],colors[2],colors[4],colors[-1] = 'darkcyan','darkcyan','darkcyan','darkcyan'

fig = px.bar(data_frame = year_count.head(10),
             x = 'Year', y = 'Count')

fig.update_traces(marker_color = colors)

fig.update_layout(title = 'Year(s) Distribution')

fig.show()


# # Rating

# In[10]:


print("Statistical value of [{}]".format('Rating'))

# Average Rating 
print("Mean:", round(movie['RATING'].mean(),2))

# Median Rating
print("Median:", movie['RATING'].median())

# Max Rating
print("Max:", movie['RATING'].max())


# # Rating Distribution

# In[11]:


fig = px.bar(data_frame = movie['RATING'].value_counts().reset_index().head(10),
             x = 'index', y = 'RATING',
             title = 'Rating Distribution')

fig.update_yaxes(title = 'Count')

fig.update_xaxes(type ='category',
                 title = 'Rating (out of 10)')

fig.show()


# # Runtime Distribution

# In[12]:


fig = px.bar(data_frame = movie['RunTime'].value_counts().reset_index().head(10),
             x = 'index', y = 'RunTime',
             title = 'Runtime Distribution')

fig.update_yaxes(title = 'Count')

fig.update_xaxes(type ='category',
                 title = 'Runtime (mins)')

fig.show()


# # Voting

# In[13]:


movie.info()


# In[14]:


movie['VOTES'] = movie['VOTES'].str.replace(",","")
movie['VOTES'] 


# In[15]:


movie['VOTES'] = movie['VOTES'].fillna(0)
movie['VOTES'] = movie['VOTES'].astype(int)
movie['VOTES'].sort_values(ascending = False)


# # Genre

# In[16]:


movie_genre = movie['GENRE'].value_counts().reset_index().rename(columns={'GENRE':'Count','index':'Genre'})

fig = px.bar(data_frame = movie_genre.sort_values(by='Count',ascending = False).head(10),
             x = 'Genre', y = 'Count')

fig.update_layout(title = 'Top 10 Genre Combination')

fig.show()


# # Looking at Individual Genre

# In[17]:


from collections import Counter

genre_raw = movie['GENRE'].dropna().to_list()
genre_list = list()

for genres in genre_raw:
    genres = genres.split(", ")
    for g in genres:
        genre_list.append(g)
        
genre_df = pd.DataFrame.from_dict(Counter(genre_list), orient = 'index').rename(columns = {0:'Count'})
genre_df.head()


# # Genre Count Distribution

# In[18]:


# Genre Count Ditribution
fig = px.pie(data_frame = genre_df,
             values = 'Count',
             names = genre_df.index,
             color_discrete_sequence = px.colors.qualitative.Safe)

fig.update_traces(textposition = 'inside',
                  textinfo = 'label+percent',
                  pull = [0.05] * len(genre_df.index.to_list()))

fig.update_layout(title = {'text':'Genre Distribution'},
                  legend_title = 'Gender',
                  uniformtext_minsize=13,
                  uniformtext_mode='hide',
                  font = dict(
                      family = "Courier New, monospace",
                      size = 18,
                      color = 'black'
                  ))


# # Gross

# In[19]:


gross_df = movie[~movie['Gross'].isna()] # New Dataframe with no NaN in Gross column

# Extract the numerical value
def extract_gross(gross):
    return float(gross.replace("$","").replace("M",""))

# Unit is Million US Dollar
gross_df['Gross'] = gross_df['Gross'].apply(lambda g: extract_gross(g))

# Highest Gross Movie
print("Highest Gross movie:",gross_df.iloc[gross_df['Gross'].argmax()]['MOVIES'])


# In[20]:


fig = px.bar(data_frame = gross_df.sort_values(by='Gross', ascending = False).head(10),
             x = 'MOVIES', y = 'Gross',
             title = 'Top 10 Gross Movie')
fig.update_layout(yaxis_title = 'Million US Dollar')
fig.show()


# # CONCLUSION

# In[ ]:


# for the Movie Recommendation System the Cosine similarity algorithm has been used to recommend the best movie entered by
#the user based on different factors such as the genre of the movie,overview,the ratings given to the movie

