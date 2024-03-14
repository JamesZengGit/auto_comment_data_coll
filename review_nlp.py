#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag


# In[14]:


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


# In[12]:


# Replace 'your_file.tsv' with the actual path to your TSV file
with_comments_path = 'E:/Git/JamesZengGit/CommentsOnComments_Python.tsv'

# Use read_csv with the 'sep' parameter to specify the delimiter
# In this case, '\t' is used to indicate a tab-separated file
df = pd.read_csv(with_comments_path, sep='\t')

# Display the DataFrame
# print(df)


# In[11]:


pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)

print(df[['index', 'filename', 'pr_number', 'comment_body']])

pd.reset_option('display.max_colwidth')
pd.reset_option('display.max_columns')


# In[15]:


def identify_review_keywords(comment):
    tokens = word_tokenize(comment)
    tagged_tokens = pos_tag(tokens)

    # Extract nouns and adjectives as potential keywords
    keywords = [word for word, pos in tagged_tokens if pos.startswith('NN') or pos.startswith('JJ')]

    return keywords


# In[18]:


'''
    Identify the keywords by joining.
'''
# Concatenate all comments into a single string
all_comments = ' '.join(df['comment_body'])

# Apply the identify_keywords function to the concatenated comments
overall_keywords = identify_review_keywords(all_comments)

# Display the overall keywords
# print("Overall Keywords:", overall_keywords)
# IOPub data rate exceeded.


# In[20]:


output_file_path = 'keywords.txt'
with open(output_file_path, 'w', encoding='utf-8') as file:
    file.write('\n'.join(overall_keywords))

print(f"Overall Keywords have been saved to '{output_file_path}'.")


# In[23]:


'''
    Identify by keywords of keywords.
'''
# Apply the identify_keywords function to the CommentBody column
df['Keywords'] = df['comment_body'].apply(identify_review_keywords)

# Flatten the list of lists into a single list of all keywords
keywords_of_keywords = [keyword for keywords_list in df['Keywords'] for keyword in keywords_list]

# Display the overall keyword
# print("Overall Keywords:", overall_keywords)
# IOPub data rate exceeded.


# In[24]:


output_file_path = 'keywords_of_keywords.txt'
with open(output_file_path, 'w', encoding='utf-8') as file:
    file.write('\n'.join(keywords_of_keywords))

print(f"Overall Keywords have been saved to '{output_file_path}'.")

