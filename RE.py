#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re


# In[2]:


# Rawstring
string = "\tTab"
rawstring = r"\tTab"
print(string)
print(rawstring)


# In[8]:


text_to_search = '''
abceaofinianfeafewafenafeabcfewanf
efwanfoia
fewafabfaifeoaabcewfeww
3922r3
:[lp[f Ha ah wfwew9hf
'''
target = re.compile(r'abc')
matches = target.finditer(text_to_search)

print('This is matches')
print(matches)
print('values')
for match in matches:
    print(match)
# print(text_to_search[26:29])


# In[ ]:


# Metacharacters (escape them)
. ^ $ * + ? { } [ ] \ | ( )


# In[9]:


text_to_search = '''
google.com
'''
\A
target = re.compile(r'google\.com')
matches = target.finditer(text_to_search)
for match in matches:
    print(match)


# In[ ]:




