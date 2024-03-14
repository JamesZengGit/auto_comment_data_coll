#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd

# Create an empty DataFrame
df = pd.DataFrame({'DataColumn': [None, None, None, None]})

print(len(df))

# Define the list of lists
list_of_lists = [None, ['item1', 'item2', 'item3'], ['item4', 'item5'], ['item6']]

for j in range(len(list_of_lists)):
    print(j)
    print(df['DataColumn'][j])
    df['DataColumn'][j] = list_of_lists[j]

# Assign the list of lists to the DataFrame column
df['DataColumn'] = list_of_lists

# Print the DataFrame
print(df)

