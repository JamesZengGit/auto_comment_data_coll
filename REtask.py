#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re


# In[48]:


text_to_go = [
    '@@ -1,3 +1,3 @@\n-# testPrPlayground\n+### testPrPlayground\n https://zenodo.org/record/5886145', 
    '@@ -1,3 +1,3 @@\n-# testPrPlayground\n+### testPrPlayground\n https://zenodo.org/record/5886145\n-This is a README, welcome.\n+### This is a README, welcome.', 
    '@@ -1,3 +1,4 @@\n ### testPrPlayground\n https://zenodo.org/record/5886145\n+### This is a README, welcome.\n+git checkout some-existing-branch'
]


# In[55]:


# There are three cases:
# n- means the comment was deleted
# n- n+ means the comment might be edited
# n+ means a comment
patterns = re.compile(r'\n\+(\t)*#.+')

for item in text_to_go:
    matches = patterns.finditer(item)
    for match in matches:
        print(match)

#for match in matches:
#    print(match)


# In[37]:


text_to_go[0][35:38]


# In[ ]:




