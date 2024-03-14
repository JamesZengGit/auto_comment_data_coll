#!/usr/bin/env python
# coding: utf-8

# In[2]:


import subprocess
import json
import pandas as pd


# In[6]:


npc = pd.read_csv('E:\Git\JamesZengGit\AnnotatedCommentsOnComments.tsv', delimiter = '\t')


# In[8]:


npc.columns


# In[3]:


full = "https://github.com/openmrs/openmrs-module-fhir2/pull/2"
GITHUB_ACCESS_TOKEN = "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"
OWNER = "openmrs"
REPO = "openmrs-module-fhir2"
PULL_NUMBER = 2


# In[3]:


full = "https://github.com/palantir/atlasdb/pull/3310#discussion_r197728214"
GITHUB_ACCESS_TOKEN = "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"
OWNER = "palantir"
REPO = "atlasdb"
PULL_NUMBER = 3310
REVIEW_ID = 131236602
# Construct the cURL command
curl_command = f'curl -L \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer {GITHUB_ACCESS_TOKEN}" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/repos/{OWNER}/{REPO}/pulls/{PULL_NUMBER}/reviews/{REVIEW_ID}/comments'

# Execute the cURL command and capture the output
try:
    response = subprocess.check_output(curl_command, shell=True)
    reviews_data = json.loads(response)
except subprocess.CalledProcessError:
    print("Error occurred while fetching data."/
    exit(1)

# Convert the JSON data into a Pandas DataFrame
reviews_df1 = pd.DataFrame(reviews_data)

# Optionally, you can filter and display the necessary information from the DataFrame

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
display(reviews_data)
pd.reset_option('display.max_colwidth')
pd.reset_option('display.max_rows')


# In[5]:


# Construct the cURL command
curl_command = f'curl -H "Authorization: token {GITHUB_ACCESS_TOKEN}" \
                https://api.github.com/repos/{OWNER}/{REPO}/pulls/{PULL_NUMBER}/reviews'

# Execute the cURL command and capture the output
try:
    response = subprocess.check_output(curl_command, shell=True)
    reviews_data = json.loads(response)
except subprocess.CalledProcessError:
    print("Error occurred while fetching data.")
    exit(1)

# Convert the JSON data into a Pandas DataFrame
reviews_df = pd.DataFrame(reviews_data)

# Optionally, you can filter and display the necessary information from the DataFrame

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
display(reviews_df.head(1))
pd.reset_option('display.max_colwidth')
pd.reset_option('display.max_rows')


# In[44]:


full = "https://github.com/openmrs/openmrs-module-fhir2/pull/119"
GITHUB_ACCESS_TOKEN = "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"
OWNER = "openmrs"
REPO = "openmrs-module-fhir2"
PULL_NUMBER = 119
# Construct the cURL command
curl_command = f'curl -L \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer {GITHUB_ACCESS_TOKEN}" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/repos/{OWNER}/{REPO}/pulls/{PULL_NUMBER}/comments'

# Execute the cURL command and capture the output
try:
    response = subprocess.check_output(curl_command, shell=True)
    reviews_data = json.loads(response)
except subprocess.CalledProcessError:
    print("Error occurred while fetching data.")
    exit(1)

# Convert the JSON data into a Pandas DataFrame
reviews_df = pd.DataFrame(reviews_data)

# Optionally, you can filter and display the necessary information from the DataFrame
# filtered_df = reviews_df[['user', 'state', 'submitted_at', 'body']]

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
display(reviews_df)
pd.reset_option('display.max_colwidth')
pd.reset_option('display.max_rows')


# In[ ]:




