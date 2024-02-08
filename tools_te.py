import json
import pandas as pd
from git import Repo
import os
import request
import subprocess

import numpy as np
import re

'''
    Get author, repository name from a GitHub address.
'''
def get_info(repo_link):
    parts = repo_link.replace("https://github.com/", "").split("/")
    try:
        return parts[0], parts[1]
    except Exception as e:
        print("an error occured: ", str(e))
        return None, None
      
