#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori


# In[2]:


dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
           ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
           ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]


# In[4]:


te_ary


# In[5]:


te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
df


# In[6]:


apr = apriori(df, min_support=0.6)
apr


# In[8]:


data = {1 : ['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'], 
        2 : ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'], 
        3 : ['Milk', 'Apple', 'Kidney Beans', 'Eggs'], 
        4 : ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'], 
        5 : ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']}


# In[34]:


columns = set(['temp'])
for l in data:
    s = set(data[l])
    columns = columns.union(s)
columns = list(columns)
columns.remove('temp')
print(columns)


# In[48]:


df = pd.DataFrame(data=data[1],columns=columns)


# In[47]:


np.array(dataset)


# In[ ]:


pd.same

