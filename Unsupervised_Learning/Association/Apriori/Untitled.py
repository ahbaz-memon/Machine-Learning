#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
from mlxtend.frequent_patterns import apriori # machine learning xtend tool for apriori algorithm
from mlxtend.frequent_patterns import association_rules # machine learning xtend tool for association rules
from mlxtend.frequent_patterns import fpgrowth # machine learning xtend tool for fp growth algorithm


# In[2]:


data = pd.read_csv("bigmart data.csv")
data


# In[3]:


sns.displot(data['Item_Visibility'] )


# In[4]:


bin = np.arange(0,0.3,0.05/8)
# data['Item_Visibility'] >= 0.05
bin_str = [str(e) for e in bin]


# In[5]:


len(bin_str)


# In[ ]:





# In[6]:


data['Item_Type'].value_counts()


# In[7]:


data['Outlet_Location_Type'].value_counts()


# In[8]:


data['Item_Fat_Content'].value_counts()


# In[9]:


data['Item_Identifier'].value_counts()


# In[11]:


#data['Item_Identifier'] == 'FDG33' or data['item_visibility'] >= 0.06 


# In[12]:


#data[data['Item_Identifier'] == 'FDG33' or data['item_visibility'] >=0.06]['Item_Type']


# In[13]:


#data['Outlet_Establishment_Year'].value_counts().values


# In[ ]:


data['Item_Outlet_Sales'].value_counts()


# In[ ]:


l = []
for e in data['Item_Outlet_Sales'].value_counts().index:
    l.append(list(set(data[data['Item_Outlet_Sales'] == e]['Item_Type'].values)))


# In[ ]:


data = l


# In[ ]:


columns = set([]) # empty set since set skip the same entries
for l in data: # a list in data
    for e in l: # a element in list
        columns.add(e)
columns = list(columns) # convert set to list to easy iterate
columns = sorted(columns) # sort elements by alphabetical order
columns


# In[ ]:


bool_data = []
for l in data: # a list in data
    temp = []
    for c in columns: # every column in columns
        if c in l: # if column is in the list
            temp.append(True) # append the value True 
        else:
            temp.append(False) # append the value False
    bool_data.append(temp) 


# In[ ]:


bool_data


# In[ ]:


data = pd.DataFrame(bool_data, columns = columns) # creating data frame


# In[ ]:


data


# In[ ]:


apr = apriori(df = data, min_support = 0.1, use_colnames = True)
apr


# In[ ]:




