#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import re


# In[30]:


git_comment = pd.DataFrame()
patterns = r'^[(\s)|(\t)]*(\#|\'\'\')'
a['p'] = None
for j in range(len(a)):
    matches = re.findall(patterns, a['after'][j], re.MULTILINE)
    print(matches)
    if a['after'][j] != None:
        if matches:
            frames = [git_comment, pd.DataFrame(a.loc[j]).transpose()]
            git_comment = pd.concat(frames)
            git_comment.iloc[j, -1] = matches
            print(matches)
print(git_comment)


# In[1]:


import os
import subprocess

# Directory where your Git repositories are stored
git_directory = 'E:/Git/JamesZengGit'

# List all directories in the git_directory
repository_dirs = [d for d in os.listdir(git_directory) if os.path.isdir(os.path.join(git_directory, d))]

# Iterate through the repositories and update them
for repo_dir in repository_dirs:
    repo_path = os.path.join(git_directory, repo_dir)
    try:
        # Change directory to the repository
        os.chdir(repo_path)
        
        # Run 'git pull' to update the repository
        subprocess.run(['git', 'pull'], check=True)
        print(f'Updated repository in {repo_path}')
    except Exception as e:
        print(f'Error updating repository in {repo_path}: {str(e)}')

# Change back to the original working directory (if necessary)
os.chdir('E:/anaconda_windows/jupyterlab')


# In[11]:


data = pd.read_table("E:\Git\JamesZengGit\_en\data_(interactions-py)_interactions.py.tsc",delimiter = "\t")


# In[25]:


a = data


# In[30]:


len(a)


# In[28]:


a = {"after": ['''They are gre,
     ##niofnsf
     \'''fwionrwrewrwwr
     ''',
     ''' #iofenwfi
     #fnew
     ''',
     '''nfsio#nionfoiwnfw
     ''']}
a = pd.DataFrame(a)
a['after'][0]


# In[14]:


pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_rows", None)
# display(data.loc[1])
pd.reset_option("display.max_colwidth")
pd.reset_option("display.max_rows")


# In[36]:


git_comment

