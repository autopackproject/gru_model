import shutil
import os
import pandas as pd
from pandas import DataFrame
import re
import sys
import validators
import glob

# WIP
def get_input_from_output(specfile):
    """Download the input data (c++ repository) from the specfile url"""
    for line in open(specfile, 'r'):
        if re.search("URL: ", line):
            print(line)
            break
        if line == None:
            print('no matches found')

    if re.search("%{", line):
        url_var = re.search('%{(.*)}', line).group(1)
        print("Found: " + url_var)

        for line in open(specfile, 'r'):
            if re.search(url_var, line):
                print(line)
                url = re.search('(?P<url>https?://[^\s]+)', line).group("url")
                print("URL: " + url)
                break
            if line == None:
                print('no matches found')
    else:
        url = re.search('(?P<url>https?://[^\s]+)', line).group("url")
        print("URL: " + url)
    
    return url

# WIP
def download_input(url):
    """Download the input data (c++ repository) from the url"""
    if validators.url(url):
        print("Downloading from " + url)
        os.system("wget " + url)
    else:
        print("Invalid URL")


def convert_repo_to_strings(root_dir):
    for filename in glob.iglob(root_dir + '**/**', recursive=True):
        with open(root_dir + 'repo_content.txt', 'w') as outfile:
            with open(root_dir + filename, 'r') as infile:
                outfile.write(infile.read())
    
    return 0
