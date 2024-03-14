#!/usr/bin/env python
# coding: utf-8

# In[2]:


import subprocess
import pandas as pd
import json


# In[3]:


def generate_param(owner, repo):
    ACCESS_TOKEN = "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"
    REPO_OWNER = owner
    REPO_NAME = repo
    return ACCESS_TOKEN, REPO_OWNER, REPO_NAME


# In[4]:


ACCESS_TOKEN, REPO_OWNER, REPO_NAME = generate_param("palantir", "atlasdb")


# In[5]:


def generate_pullsID(ACCESS_TOKEN, REPO_OWNER, REPO_NAME):
    curl_plist = f'curl -H "Authorization: token {ACCESS_TOKEN}" https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/pulls?state=all'
    try:
        response_plist = subprocess.check_output(curl_plist, shell=True)
        reviews_data = json.loads(response_plist)
    except subprocess.CalledProcessError:
        print("Error occurred while fetching data.")
        exit(1)
    
    reviews_df1 = pd.DataFrame(reviews_data)
    review_id = list(reviews_df1["id"])
    pull_id = list(reviews_df1["number"])
    return pull_id, review_id


# In[6]:


id_p, id_r = generate_pullsID(ACCESS_TOKEN, REPO_OWNER, REPO_NAME)


# In[9]:


PULL_NUMBER = id_p[0]
REVIEW_ID = id_r[0]

# Construct the cURL command
curl_command = f'curl -L \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer {ACCESS_TOKEN}" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/pulls/{PULL_NUMBER}/comments'

# Execute the cURL command and capture the output
try:
    response = subprocess.check_output(curl_command, shell=True)
    reviews_data = json.loads(response)
except subprocess.CalledProcessError:
    print("Error occurred while fetching data.")
    exit(1)

# Convert the JSON data into a Pandas DataFrame
changes_va = pd.DataFrame(reviews_data)
changes_va.rename(columns = {'diff_hunk':'diff_chunk'}, inplace = True)

# Optionally, you can filter and display the necessary information from the DataFrame

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
display(changes_va)
pd.reset_option('display.max_colwidth')
pd.reset_option('display.max_rows')


# In[9]:


from git import Repo
import os


# In[66]:


def add_before(commit_hash, file_path, start_line, end_line):
    clone_path = "E:/Git/JamesZengGit/testPrPlayground2/testPrPlayground"
    # Initialize the repository object
    repo = Repo(clone_path)

    # Replace 'COMMIT_HASH' with the hash of the commit you want to acces
    # commit_hash = 'ce5be0c182c63c35bb6e5251d432097aae1b84ac	'

    # Get the commit object
    commit = repo.commit(commit_hash)
    
    # file_path = 'README.md'
    
    # start_line = 3
    # end_line = 3
    
    file_content = commit.tree[file_path].data_stream.read().decode('utf-8')
    
    lines = file_content.splitlines()
    
    changed_lines = lines[start_line - 1:end_line]
    
    return '\n'.join(changed_lines)


# In[67]:


def add_surrounding(commit_hash, file_path, start_line, end_line):
    clone_path = "E:/Git/JamesZengGit/testPrPlayground2/testPrPlayground"
    # Initialize the repository object
    repo = Repo(clone_path)
    
    # Replace 'COMMIT_HASH' with the hash of the commit you want to access
    # commit_hash = 'ce5be0c182c63c35bb6e5251d432097aae1b84ac	'

    # Get the commit object
    commit = repo.commit(commit_hash)

    # file_path = 'README.md'

    # start_line = 3
    # end_line = 3

    file_content = commit.tree[file_path].data_stream.read().decode('utf-8')

    lines = file_content.splitlines()

    x = 10
    y = 10
    
    changed_lines = lines[max(0, start_line - 1 - x):min(end_line + y, len(lines))]

    return '\n'.join(changed_lines)


# In[68]:


def add_after(commit_hash, file_path, start_line, end_line):
    clone_path = "E:/Git/JamesZengGit/testPrPlayground2/testPrPlayground"
    # Initialize the repository object
    repo = Repo(clone_path)
    
    # Replace 'COMMIT_HASH' with the hash of the commit you want to access
    # commit_hash = 'ce5be0c182c63c35bb6e5251d432097aae1b84ac	'

    # Get the commit object
    commit = repo.commit(commit_hash)

    # file_path = 'README.md'

    # start_line = 3
    # end_line = 3

    file_content = commit.tree[file_path].data_stream.read().decode('utf-8')

    lines = file_content.splitlines()
    
    changed_lines = lines[start_line - 1:end_line]

    return '\n'.join(changed_lines)


# In[69]:


import numpy as np


# In[18]:


if np.isnan(changes_va['original_start_line'][2]):
    changes_va['start_line'][2] = changes_va['line'][2]
    print(2)
new = generate_after(changes_va['commit_id'][2], changes_va['path'][2], changes_va['line'][2].astype(int), changes_va['line'][2].astype(int))
new


# In[71]:


changes_va['original'], changes_va['context'], changes_va['new'] = "", "", ""
for x in range(0, 3):
    if np.isnan(changes_va['original_start_line'][x]):
        start_line = 'line'
        changes_va['original'][x] = generate_original(changes_va['original_commit_id' ][x], changes_va['path'][x], changes_va['original_line'][x].astype(int), changes_va['original_line'][x].astype(int))
        changes_va['context'][x] = generate_context(changes_va['original_commit_id'][x], changes_va['path'][x], changes_va['original_line'][x].astype(int), changes_va['original_line'][x].astype(int))
        original_start_line = 'original_line'
        print(1)
    else:
        changes_va['original'][x] = generate_original(changes_va['original_commit_id' ][x], changes_va['path'][x], changes_va['original_start_line'][x].astype(int), changes_va['original_line'][x].astype(int))
        changes_va['context'][x] = generate_context(changes_va['original_commit_id'][x], changes_va['path'][x], changes_va['original_start_line'][x].astype(int), changes_va['original_line'][x].astype(int))
        start_line = 'start_line'
        print(2)
    changes_va['new'][x] = generate_after(changes_va['commit_id'][x], changes_va['path'][x], changes_va[start_line][x].astype(int), changes_va['line'][x].astype(int))
    print(3)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
display(changes_va)
pd.reset_option('display.max_colwidth')
pd.reset_option('display.max_rows')


# In[ ]:


for chunk in d


# In[116]:


commit = repo.commit(changes_va['original_commit_id'][0])
file_content = commit.tree[file_path].data_stream.read().decode('utf-8')
lines = file_content.splitlines()
display(lines)


# In[95]:


context = generate_context(changes_va['original_commit_id'][0], changes_va['path'][0], changes_va['original_start_line'][0].astype(int), changes_va['original_line'][0].astype(int))
context


# In[97]:


new = generate_after(changes_va['commit_id'][0], changes_va['path'][0], changes_va['start_line'][0].astype(int), changes_va['line'][0].astype(int))
new


# In[78]:


import re
voraciousness_comment = pd.DataFrame()


# In[61]:


changed_code


# In[57]:


changes_va['new'][0]


# In[89]:


patterns = re.compile(r'\n\+(\t)*#.+')

for j in range(len(changes_va)):
    matches = patterns.finditer(changes_va['diff_chunk'][j])
    if matches:
        voraciousness_comment = voraciousness_comment.append(changes_va.loc[j])


# In[90]:


voraciousness_comment


# In[1]:


import git


# In[ ]:




