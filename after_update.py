#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import os
import json


# In[28]:


# Specify the folder path containing TSV files
folder_path = 'E:\Git\JamesZengGit\_en'
combined_df = pd.DataFrame()

for file_name in os.listdir(folder_path):
    if file_name.endswith('.tsc'):
        file_path = os.path.join(folder_path, file_name)

        # Check if the file is empty or contains only whitespace
        with open(file_path, 'r', encoding='utf-8') as file:
            if os.path.getsize(file_path) == 0 or not any(c.strip() for c in file):
                print(f"Warning: Empty or whitespace-only file found: {file_path}")
                continue

        # Read each TSV file and append to the combined DataFrame
        df = pd.read_csv(file_path, sep="\t")
        combined_df = combined_df.append(df, ignore_index=True)

# Now 'combined_df' contains the data from all TSV files in the folder


# In[59]:


pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)

print(combined_df.iloc[[11]].copy())

pd.reset_option('display.max_colwidth')
pd.reset_option('display.max_columns')


# In[13]:


for file_name in file_list:
    if file_name.endswith('.json'):
        file_path = os.path.join(folder_path, file_name)
        try:
            df = pd.read_json(file_path)
            combined_df = combined_df.append(df, ignore_index=True)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")


# In[14]:


print(file_list)


# In[9]:


print(combined_df_p)

