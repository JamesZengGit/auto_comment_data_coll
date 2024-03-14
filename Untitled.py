#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import subprocess
import json
import pandas as pd

from git import Repo
import os

import numpy as np

import re


# In[3]:


folder_path = r'E:\Git\JamesZengGit'
repository_urls = []

for folder_name in os.listdir(folder_path):
    # Assuming folder_name is the name of the repository
    repo_name = folder_name.strip()  # Remove any leading/trailing whitespace
    
    # Use Git to get the remote URL of the repository
    try:
        remote_url = subprocess.check_output(
            ['git', '-C', os.path.join(folder_path, folder_name), 'config', '--get', 'remote.origin.url'],
            universal_newlines=True
        ).strip()
        
        # Extract the owner from the remote URL (assuming it's in the format "https://github.com/owner/repo.git")
        if 'github.com' in remote_url:
            owner = remote_url.split('/')[-2]
            github_url = f'https://github.com/{owner}/{repo_name}'
            repository_urls.append(github_url)
    except subprocess.CalledProcessError:
        # Handle Git command errors or repositories without a remote URL
        print(f"Failed to get remote URL for {repo_name}")

# Create a pandas DataFrame with the GitHub URLs
red = pd.DataFrame({'GitHub URL': repository_urls})


# In[2]:


def get_info(lic):
    parts = lic.replace("https://github.com/", "").split("/")
    try:
        return parts[0], parts[1]
    except Exception as e:
        print("an error occured: ", str(e))
        return None, None


# In[3]:


# Using
import subprocess
import json
import requests  # Import the requests library

def generate_pullsID(ACCESS_TOKEN, REPO_OWNER, REPO_NAME):
    # Use the requests library to handle redirection
    api_url = f'https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/pulls?state=all'
    headers = {'Authorization': f'token {ACCESS_TOKEN}'}
    
    try:
        response = requests.get(api_url, headers=headers, allow_redirects=False)  # Disable automatic redirection
        if response.status_code == 301 or response.status_code == 302:
            # Follow the redirection URL
            redirected_url = response.headers['Location']
            response = requests.get(redirected_url, headers=headers)
        
        response.raise_for_status()  # Raise an exception for non-success status codes
        
        reviews_data = response.json()  # Parse the JSON response
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while fetching data: {e}")
        exit(1)
    
    if not isinstance(reviews_data, list):
        print("Unexpected data format received from API.")
        exit(1)
    
    reviews_df1 = pd.DataFrame(reviews_data)
    
    # Check if 'id' column exists in the DataFrame
    if 'id' in reviews_df1.columns:
        review_id = list(reviews_df1["id"])
        pull_id = list(reviews_df1["number"])
        return pull_id, review_id, reviews_df1
    else:
        print("The 'id' column is missing in the DataFrame.")
        print(len(reviews_df1))
        # You might want to handle this situation accordingly, e.g., by returning None or raising an exception
        
        return None, None, None

# PULLS, id_r, pr_info = generate_pullsID(ACCESS_TOKEN, REPO_OWNER, REPO_NAME)


# In[4]:


def generate_pullsID(ACCESS_TOKEN, REPO_OWNER, REPO_NAME):
    curl_plist = f'curl -H "Authorization: token {ACCESS_TOKEN}" https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/pulls?state=all'
    try:
        response_plist = subprocess.check_output(curl_plist, shell=True)
        reviews_data = json.loads(response_plist)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while fetching data: {e}")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}")
        exit(1)
    
    if not isinstance(reviews_data, list):
        print("Unexpected data format received from API.")
        exit(1)
    
    reviews_df1 = pd.DataFrame(reviews_data)
    review_id = list(reviews_df1["id"])
    pull_id = list(reviews_df1["number"])
    return pull_id, review_id, reviews_df1

# PULLS, id_r, pr_info = generate_pullsID(ACCESS_TOKEN, REPO_OWNER, REPO_NAME)


# In[4]:


ind = set()
'''
    Single Repository
'''
def get_issues_for_repository(owner, repo, token):
    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    params = {
        "state": "closed",
        "sort": "comments",
        "direction": "desc"
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    response = requests.get(url, params=params, headers=headers)
    data = response.json()
    return data

def main():
    for repo in popular_repositories:
        owner = repo["owner"]["login"]
        repo_name = repo["name"]
        issues_data = get_issues_for_repository(owner, repo_name, ACCESS_TOKEN)
        
        for issue in issues_data:
            if issue['comments'] > 1 and 'pull_request' in issue:
                # print(f"Repository: {owner}/{repo_name}")
                # print(f"Issue Title: {issue['title']}")
                # print(f"Stars: {repo['stargazers_count']}")
                # print(f"Issue URL: {issue['html_url']}")
                # print("=" * 50)
                ind.add(f"{issue['html_url']}")

if __name__ == "__main__":
    main()
# print(ind)


# In[ ]:


ind[0]


# In[64]:


data


# In[7]:


params = {
        "q": "stars:>0",
        "sort": "stars",
        "order": "desc",
        "page": 2,
        "per_page": 100,
    }
response = requests.get(BASE_URL, headers=headers, params=params)
response.jason


# In[11]:


import requests

def check_access_token_validity(access_token):
    # GitHub user profile endpoint
    user_profile_url = "https://api.github.com/user"
    
    headers = {"Authorization": f"Bearer {access_token}"}
    
    try:
        response = requests.get(user_profile_url, headers=headers)
        response.raise_for_status()  # Raise an exception for bad responses (4xx and 5xx)
        
        # If the request is successful, the access token is valid
        return True
    
    except requests.exceptions.HTTPError as err:
        # If the request failed, the access token is likely invalid
        print(f"Error checking access token validity: {err}")
        return False

# Example usage
access_token = "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"
if check_access_token_validity(access_token):
    print("Access token is valid.")
else:
    print("Access token is not valid.")


# In[6]:


import requests

'''
    Fetch Repositories.
'''

ACCESS_TOKEN = "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"

BASE_URL = "https://api.github.com/search/repositories"
headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}

def fetch_popular_repositories(page, per_page):
    params = {
        "q": "language:python stars:>0 size:<30000000",
        "sort": "stars",
        "order": "desc",
        "page": page,
        "per_page": per_page,
    }
    response = requests.get(BASE_URL, headers=headers, params=params)
    return response.json()

def get_all_popular_repositories(total_count, per_page):
    all_repositories, all_repo_urls = [], []
    pages = (total_count + per_page - 1) // per_page

    for page in range(1, pages + 1):
        repositories = fetch_popular_repositories(page, per_page)
        all_repositories.extend(repositories["items"])
        
        # Generate repository URLs and add to the list
        repo_urls = [item.get("html_url") for item in all_repositories]
        all_repo_urls.extend(repo_urls)

    return all_repositories, all_repo_urls

total_count = 1000  # Number of popular repositories you want to fetch
per_page = 200     # Number of repositories per page

popular_repositories, repo_urls = get_all_popular_repositories(total_count, per_page)
repo_urls_set = set(repo_urls)

print(len(popular_repositories))
# Now you can work with the list of popular repositories


# In[12]:


'''
    Generate data frame for reviews in all repos, iterate repos pull requests reviews 
    info pr_id comments_id diff_chunk
'''
def iterate_review_diff(reps):
    ACCESS_TOKEN = "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"
    with_comment_reviews = pd.DataFrame()
    for repo in reps:
        # red = url.replace("https://github.com/", "").split("/")
        # lic = 'https://github.com/JamesZengGit/testPrPlayground'
        REPO_OWNER, REPO_NAME = get_info(repo)

        PULLS, id_r, pr_info = generate_pullsID(ACCESS_TOKEN, REPO_OWNER, REPO_NAME)
        # when pull do not have id
        if PULLS is None:
            continue
        # data frame for all comments in a repo
        commit_info = iterate_pr_for_commit(ACCESS_TOKEN, REPO_OWNER, PULLS, REPO_NAME)
        
        # search diff
        for index, review in commit_info.iterrows():
            if select_diff_comments(str(review[['diff_chunk']])):
                with_comment_reviews_frame = [with_comment_reviews, review]
                with_comment_reviews = pd.concat(with_comment_reviews_frame)
    
    return with_comment_reviews

iterate_review_diff(repo_urls_set)


# In[7]:


'''
    Select the data with comment-pattern on the column diff-chunk
'''
def select_diff_comments(comments_data):
    # regular expression pattern
    pattern = re.compile(r'.*(TODO|TO DO)|.*(# )|.*(""")|.*(\'\'\').*')
    matches = re.findall(pattern, comments_data)
    
    return matches


# In[ ]:


DESTINATION_FILE = f'E:/Git/JamesZengGit/{REPO_NAME}'
POSITION_FILE = f'{lic}.git'
print(lic)
try:
    clone_repository(POSITION_FILE, DESTINATION_FILE)
except:
    print("error")


# In[8]:


def comments_info(ACCESS, REPO_OWNER, REPO_NAME, PULL_NUMBER):
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
    except subprocess.CalledProcessError as e:
        print(f'{REPO_OWNER}/{REPO_NAME}')
        if "HTTP 403" in e.output:
            print("Access to the GitHub API is forbidden. Check your API token and permissions.")
        elif "API rate limit exceeded" in e.output:
            print("API rate limit exceeded. Please wait for it to reset or optimize your code.")
        else:
            print(f"Error occurred while fetching data: {e}")
        reviews_data = None  # Assign a default value
        exit(1)

    # Convert the JSON data into a Pandas DataFrame
    if reviews_data:
        # print(reviews_data)
        try:
            changes_va = pd.DataFrame(reviews_data)  # Add index argument
        except:
            print(f'{REPO_OWNER}/{REPO_NAME}')
            print(reviews_data)
        changes_va.rename(columns={'diff_hunk': 'diff_chunk'}, inplace=True)
        
        return changes_va
    else:
        # print(f"No reviews data for pull request {PULL_NUMBER}")
        # print(reviews_data)
        return pd.DataFrame()


# In[9]:


def iterate_pr_for_commit(ACCESS, OWNER, PULLS, NAME):
    comments = pd.DataFrame()
    for PULL_NUMBER in PULLS:
        frames = [comments, comments_info(ACCESS, OWNER, NAME, PULL_NUMBER)]
        comments = pd.concat(frames, ignore_index = True)
   # comments = comments_info(PULLS[0])
    return comments
        # for j in range(len(comments_info)):  


# In[46]:


print(PULLS)
print(len(commit_info))


# In[21]:


def clone_repository(REPOSITORY_URL, DESTINATION_FILE):
    try:
        result = subprocess.run(["git", "clone", REPOSITORY_URL, DESTINATION_FILE], capture_output=True, text=True, check=True)
        print("Repository cloned successfully!")
        print("Standard Output:", result.stdout)
        print("Standard Error:", result.stderr)
    except subprocess.CalledProcessError as e:
        print("Error:", e.returncode)
        print("Standard Output:", e.stdout)
        print("Standard Error:", e.stderr)


# In[ ]:





# In[52]:


test = {'1': [2, 3, 1], '2': [3, 2, 3]}
testframe = pd.DataFrame(test)
testframe.at[2, 0] = matches


# In[46]:


for j in range(len(storage)):
        if storage['after'][j] != None and storage['after'][j] != storage['before'][j]:
            matches = re.findall(patterns, storage['after'][j], re.MULTILINE)
            if matches:
                print(storage.iloc[j, 30])
                print(matches)
                storage.iloc[j, 30] = matches


# In[15]:


rep = set()
for url in ind:
    red = url.replace("https://github.com/", "").split("/")
    rep.add("https://github.com/" + red[0] + "/" + red[1])


# In[26]:


clone_repository('https://github.com/celery/django-celery.git', 'E:/Git/JamesZengGit/django-celery')


# In[18]:


rep = {'https://github.com/LonamiWebs/Telethon', 'https://github.com/encode/django-rest-framework', 'https://github.com/huggingface/transformers', 'https://github.com/ccxt/ccxt', 'https://github.com/serge-chat/serge', 'https://github.com/robotframework/robotframework', 'https://github.com/openai/gpt-2', 'https://github.com/encode/apistar', 'https://github.com/lucidrains/deep-daze', 'https://github.com/qtile/qtile', 'https://github.com/dbcli/mycli', 'https://github.com/thu-ml/tianshou', 'https://github.com/tensorflow/tensor2tensor', 'https://github.com/vi3k6i5/flashtext', 'https://github.com/nltk/nltk', 'https://github.com/pydantic/pydantic', 'https://github.com/hhyo/Archery', 'https://github.com/stephenmcd/mezzanine', 'https://github.com/jina-ai/clip-as-service', 'https://github.com/Kozea/WeasyPrint', 'https://github.com/TheAlgorithms/Python', 'https://github.com/tony9402/baekjoon', 'https://github.com/fail2ban/fail2ban', 'https://github.com/spotify/luigi', 'https://github.com/raspberrypi/documentation', 'https://github.com/saltstack/salt', 'https://github.com/aws/aws-sam-cli', 'https://github.com/facebookresearch/xformers', 'https://github.com/facebookresearch/mmf', 'https://github.com/webpy/webpy', 'https://github.com/thumbor/thumbor', 'https://github.com/explosion/spaCy', 'https://github.com/Miserlou/Zappa', 'https://github.com/mps-youtube/yewtube', 'https://github.com/dmlc/dgl', 'https://github.com/RasaHQ/rasa', 'https://github.com/StackStorm/st2', 'https://github.com/celery/celery', 'https://github.com/posativ/isso', 'https://github.com/LAION-AI/Open-Assistant', 'https://github.com/amueller/word_cloud', 'https://github.com/online-ml/river', 'https://github.com/spec-first/connexion', 'https://github.com/pyodide/pyodide', 'https://github.com/gevent/gevent', 'https://github.com/okfn-brasil/serenata-de-amor'}


# In[130]:


for red in range(6,7):
    ACCESS_TOKEN = "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"
    # red = url.replace("https://github.com/", "").split("/")
    lic = 'https://github.com/JamesZengGit/testPrPlayground'
    REPO_NAME = get_info(lic)

    PULLS, id_r, pr_info = generate_pullsID(ACCESS_TOKEN, REPO_OWNER, REPO_NAME)
    commit_info = iterate_pr_for_commit(PULLS)

    DESTINATION_FILE = f'E:/Git/JamesZengGit/{REPO_NAME}'
    POSITION_FILE = f'{lic}.git'
    print(lic)
    try:
        clone_repository(POSITION_FILE, DESTINATION_FILE)
    except:
        print("error")


# In[30]:


len(red)


# In[23]:


stored_data = pd.DataFrame()


# In[31]:


red_df = red
r = list(red.iloc[100:300, 0])


# In[32]:


for lic in r:
    ACCESS_TOKEN = "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"
    
    # red = url.replace("https://github.com/", "").split("/")
    print(lic)
    REPO_NAME, REPO_OWNER = get_info(lic)

    PULLS, id_r, pr_info = generate_pullsID(ACCESS_TOKEN, REPO_OWNER, REPO_NAME)
    commit_info = iterate_pr_for_commit(PULLS)

    DESTINATION_FILE = f'E:/Git/JamesZengGit/{REPO_NAME}'
    POSITION_FILE = f'{lic}.git'

    commit = commit_info

    repo = Repo(DESTINATION_FILE)

    commit['before_context'], commit['before'], commit['after'], commit['p'] = None, None, None, None
    storage = core(commit, DESTINATION_FILE, repo)
    changes_variable = pd.DataFrame()
    changes_variable['before_context'] = ''
    # print(add_surrounding(commit['original_commit_id'][0], commit['path'][0], 269, 272, DESTINATION_FILE))

    git_comment = pd.DataFrame()
    patterns = r'^[(\s)|(\t)]*(public|\#|\'\'\')+.*'
    for j in range(len(storage)):
        if storage['after'][j] != None and storage['after'][j] != storage['before'][j]:
            matches = re.findall(patterns, storage['after'][j], re.MULTILINE)
            if matches:
                print(matches)
                # storage.iloc[j, 30] = matches
                frames = [git_comment, pd.DataFrame(storage.loc[j]).transpose()]
                git_comment = pd.concat(frames)

    stored_data = pd.concat([stored_data, git_comment])
    my_name = r"E:\Git\JamesZengGit\_en\data_({REPO_OWNER})_{REPO_NAME}.tsv"
    read_df = git_comment.to_csv(my_name, sep="\t", index=False)
    print(read_df)


# In[ ]:


popular_reposotories_p = popular_repositories


# In[29]:


stored_data


# In[3]:


stored_data


# In[42]:


matches


# In[132]:


REPO_NAME


# In[25]:


display(red)


# In[75]:


git_comment


# In[124]:





# In[9]:


ACCESS_TOKEN = "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"
REPO_NAME, REPO_OWNER = get_info('https://github.com/pyocd/pyOCD')

PULLS, id_r, pr_info = generate_pullsID(ACCESS_TOKEN, REPO_OWNER, REPO_NAME)
commit_info = iterate_pr_for_commit(PULLS)

    DESTINATION_FILE = f'E:/Git/JamesZengGit/{REPO_NAME}'
    POSITION_FILE = f'{lic}.git'
    clone_repository(POSITION_FILE, DESTINATION_FILE)

    commit = commit_info

    repo = Repo(DESTINATION_FILE)

    commit['before_context'], commit['before'], commit['after'] = None, None, None
    storage = core(commit, DESTINATION_FILE, repo)
    changes_variable = pd.DataFrame()
    changes_variable['before_context'] = ''
    # print(add_surrounding(commit['original_commit_id'][0], commit['path'][0], 269, 272, DESTINATION_FILE))

    git_comment = pd.DataFrame()
    patterns = re.compile(r'\n\+(\t)*#.+')
    for j in range(len(storage)):
        matches = patterns.search(storage['after'][j])
        if storage['before_context'][j] != None:
            if matches:
                frames = [git_comment, pd.DataFrame(storage.loc[j]).transpose()]
                git_comment = pd.concat(frames)

    my_name = f"E:\Git\JamesZengGit\_en\data_({REPO_OWNER})_{REPO_NAME}.tsc"
    read_df = git_comment.to_csv(my_name, sep="\t", index=False)
    print(read_df)


# In[8]:


ACCESS_TOKEN = "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"
REPO_NAME, REPO_OWNER = get_info("https://github.com/pypxe/PyPXE")

PULLS, id_r, pr_info = generate_pullsID(ACCESS_TOKEN, REPO_OWNER, REPO_NAME)
commit_info = iterate_pr_for_commit(PULLS)

DESTINATION_FILE = f'E:/Git/JamesZengGit/{REPO_NAME}'
POSITION_FILE = f'{lic}.git'
# clone_repository(POSITION_FILE, DESTINATION_FILE)

commit = commit_info
repo = Repo(DESTINATION_FILE)

commit['before_context'], commit['before'], commit['after'] = None, None, None
storage = core(commit, DESTINATION_FILE, repo)


# In[137]:


repo.commit('0f145ed16b5d2ce8f66450cf24e2249a6448c1c2')


# In[138]:


repo


# In[48]:


DESTINATION_FILE


# In[9]:


def add_surrounding(commit_hash, file_path, start_line, end_line, clone_path):
    # Initialize the repository object
    repo = Repo(clone_path)
    
    # Replace 'COMMIT_HASH' with the hash of the commit you want to access
    # commit_hash = 'ce5be0c182c63c35bb6e5251d432097aae1b84ac	'

    # Get the commit object
    commit = repo.commit(commit_hash)

    file_content = commit.tree[file_path].data_stream.read().decode('utf-8')

    lines = file_content.splitlines()

    x = 10
    y = 10
    
    changed_lines = lines[max(0, start_line - 1 - x):min(end_line + y, len(lines))]

    return '\n'.join(changed_lines)


# In[10]:


def add_before(commit_hash, file_path, start_line, end_line, clone_path):
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


# In[11]:


def add_after(commit_hash, file_path, start_line, end_line, clone_path):
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


# In[12]:


def process_none(value):
    if None != value:
        processed_value = int(value)
    else:
        processed_value = value.astype(int)
    return processed_value


# In[123]:


None.astype(


# In[13]:


# commit_info = commit_info.reset_index()
def core(changes_va, file, repo):
    for x in range(len(changes_va)):
        start_line, line, original_start_line, original_line = 'start_line', 'line', 'original_start_line', 'original_line'
        if pd.isna(changes_va['original_start_line'][x]) and pd.isna(changes_va['start_line'][x]):
            if pd.isna(changes_va['line'][x]):
                if pd.isna(changes_va['original_line'][x]):
                    continue
                start_line, line = 'original_line', 'original_line'
                original_start_line = 'original_line'
            elif pd.isna(changes_va['original_line'][x]):
                start_line = 'line'
                original_line = 'line'
                original_start_line = 'line'
            else:
                start_line = 'line'
                original_start_line = 'original_line'
        elif pd.isna(changes_va['start_line'][x]) and pd.isna(changes_va['line'][x]):
            if pd.isna(changes_va['original_line'][x]):
                continue
            start_line = 'original_start_line'
            line = 'original_line'
        original_start_value = process_none(changes_va[original_start_line][x])
        original_value = process_none(changes_va[original_line][x])
        start_line_value, a_line_value = process_none(changes_va[start_line][x]), process_none(changes_va[line][x])
        def is_valid_hash(repo, commit_hash):
            try:
                commit = repo.commit(commit_hash)
                return True
            except:
                return False
        if is_valid_hash(repo, changes_va['original_commit_id'][x]) and is_valid_hash(repo, changes_va['commit_id'][x]):
            try:
                changes_va.loc[x, 'before_context'] = add_surrounding(changes_va.loc[x, 'original_commit_id'], changes_va.loc[x, 'path'], original_start_value, original_value, file)
                changes_va.loc[x, 'before'] = add_before(changes_va.loc[x, 'original_commit_id'], changes_va.loc[x, 'path'], original_start_value, original_value, file)
                changes_va.loc[x, 'after'] = add_after(changes_va.loc[x, 'commit_id'], changes_va.loc[x, 'path'], start_line_value, a_line_value, file)
            except KeyError as e:
                # display(e)
                continue
    return changes_va


# In[14]:


git_comment.iloc[:,30]


# In[44]:


# commit_info = commit_info.reset_index()
def core(changes_va, file, repo):
    for x in range(len(changes_va)):
        start_line, line, original_start_line, original_line = 'start_line', 'line', 'original_start_line', 'original_line'
        if pd.isna(changes_va['original_start_line'][x]) and pd.isna(changes_va['start_line'][x]):
            # one-line chunk
            if pd.isna(changes_va['line'][x]):
                if pd.isna(changes_va['original_line'][x]):
                    continue
                start_line, line = 'original_line', 'original_line'
                original_start_line = 'original_line'
            elif pd.isna(changes_va['original_line'][x]):
                start_line = 'line'
                original_line = 'line'
                original_start_line = 'line'
            else:
                start_line = 'line'
                original_start_line = 'original_line'
        elif pd.isna(changes_va['start_line'][x]) and pd.isna(changes_va['line'][x]):
            # no-change
            if pd.isna(changes_va['original_start_line'][x]):
                original_start_line, start_line = 'original_line', 'original_line'
                line = 'original_line'
            elif pd.isna(changes_va['original_line']):
                original_line = 'original_start_line'
                start_line = line = "original_start_line"
            else:
            start_line = 'original_start_line'
            line = 'original_line'
        original_start_value = process_none(changes_va[original_start_line][x])
        original_value = process_none(changes_va[original_line][x])
        start_line_value, a_line_value = process_none(changes_va[start_line][x]), process_none(changes_va[line][x])
        def is_valid_hash(repo, commit_hash):
            try:
                commit = repo.commit(commit_hash)
                return True
            except:
                return False
        if is_valid_hash(repo, changes_va['original_commit_id'][x]) and is_valid_hash(repo, changes_va['commit_id'][x]):
            changes_va.loc[x, 'before_context'] = add_surrounding(changes_va.loc[x, 'original_commit_id'], changes_va.loc[x, 'path'], original_start_value, original_value, file)
            changes_va.loc[x, 'before'] = add_before(changes_va.loc[x, 'original_commit_id'], changes_va.loc[x, 'path'], original_start_value, original_value, file)
            changes_va.loc[x, 'after'] = add_after(changes_va.loc[x, 'commit_id'], changes_va.loc[x, 'path'], start_line_value, a_line_value, file)
    return changes_va, lines
    # changes_va = changes_va[changes_va['after'] != None]


# In[99]:


commit


# In[50]:


# commit = commit_info.drop(9)
print(len(commit_info))
# commit = commit.drop(43)
print(len(commit))
# commit = commit.reset_index(drop=True)


# In[51]:


commit.loc[0]


# In[20]:


# print(commit['original_start_line'][1]==None)


# In[49]:


if np.isnan(commit_info['original_start_line'][2]):
    commit_info['start_line'][2] = commit_info['line'][2]
    print(2)
new = add_after(commit_info['commit_id'][2], commit_info['path'][2], commit_info['line'][2].astype(int), commit_info['line'][2].astype(int), DESTINATION_FILE)


# In[2]:


import nltk
from nltk.tokenize import word_tokenize

#nltk.download('punkt')

def tokenize_comment(comment):
    # Tokenize the comment into words
    tokens = word_tokenize(comment)
    return tokens

# Example usage
comment = "This (is) a sample comment for tokenization."
tokens = tokenize_comment(comment)

# Print the tokens
print(tokens)


# In[54]:


repo = Repo("E:/Git/JamesZengGit/npc_gzip")
# Replace 'COMMIT_HASH' with the hash of the commit you want to access
# commit_hash = 'ce5be0c182c63c35bb6e5251d432097aae1b84ac	'

# Get the commit object
commit = repo.commit(commit_info['commit_id'][2])

# file_path = 'README.md'

# start_line = 3
# end_line = 3

file_content = commit.tree[commit_info['path'][2]].data_stream.read().decode('utf-8')

lines = file_content.splitlines()
    
changed_lines = lines[commit_info['start_line'][2].astype(int) - 1:commit_info['line'][2].astype(int)]


# In[84]:


display(commit_info.loc[3])


# In[67]:


pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
display(storage)
pd.reset_option('display.max_colwidth')
pd.reset_option('display.max_rows')


# In[28]:


PULL_NUMBER = PULLS[5]
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
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
display(changes_va)
pd.reset_option('display.max_colwidth')
pd.reset_option('display.max_rows')

