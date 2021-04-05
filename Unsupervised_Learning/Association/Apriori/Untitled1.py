#!/usr/bin/env python
# coding: utf-8

# ### Header Files

# In[1]:


import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# ### Dumy Data

# In[2]:


data = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
           ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
           ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]


# ### Extracting unique columns

# In[3]:


columns = set(['temp'])
for l in data:
    s = set(l)
    columns = columns.union(s)
columns = list(columns)
columns.remove('temp')
print(columns)


# ### Extracting boolean data respective of colums

# In[4]:


data_bool = []
for l in data:
    temp = []
    for c in columns:
        if c in l:
            temp.append(True)
        else:
            temp.append(False)
    data_bool.append(temp)        
data_bool         


# ### Building data frame

# In[5]:


data_df = pd.DataFrame(data = data_bool, columns = columns)
data_df


# ### using apriiori algo

# In[6]:


apr = apriori(data_df, min_support = 0.6, use_colnames = True)
apr


# ### accosiation rules

# In[7]:


# Note : need to see doccumentation of association rule.... didn't get


# In[8]:


ar = association_rules(apr, min_threshold=0.8)
ar

