#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from matplotlib import pyplot as plt
from sklearn import cluster
from sklearn.datasets import make_blobs

import seaborn as sns


# In[3]:


data = make_blobs(n_samples = 100,
                  n_features = 2,
                  centers = 5,
                  cluster_std = 0.75,
                  random_state = 6
                 )[0]


# In[3]:


plt.figure(dpi = 500)
plt.title('Data without  clustering')
plt.xlabel('X1')
plt.ylabel('X2')
plt.scatter(data[:, 0], data[:, 1], alpha = 0.75)
plt.show()


# In[4]:


kmeans = cluster.KMeans(n_clusters = 5, 
                        n_jobs = -1
                       )


# In[5]:


kmeans.fit(data)


# In[6]:


y = kmeans.labels_
y


# In[7]:


mean_centers = kmeans.cluster_centers_
mean_centers


# In[8]:


plt.figure(dpi = 500)

plt.title('Data with clustering')
plt.xlabel('X1')
plt.ylabel('X2')

plt.scatter(mean_centers[:, 0], 
            mean_centers[:, 1], 
            c = 'k',
            alpha = 1,
            marker = '*'
           )
plt.scatter(data[:, 0], 
            data[:, 1], 
            c = y,
            alpha = 0.75,
           )

plt.show()


# In[ ]:




