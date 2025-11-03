import os

import pandas as pd
from tqdm import tqdm
from chemdataextractor import Document
import xml.etree.ElementTree as ET
from xml.dom.minidom import parse
import re
import xml
import json
from bs4 import BeautifulSoup

CWD = os.getcwd()

df_mof = pd.read_csv('all_MOFs_KG.csv')
mof_dois_master_list = list(dict.fromkeys([i for i in df_mof['DOI'] if type(i) == str]))


# helpful function to check if output directory exists. If not then create one
def check_dir(cwd, output_folder_name):
    # define directories
    dir_output = os.path.join(os.path.dirname(cwd), output_folder_name)

    # Create an output directory if it does not exist yet
    if os.path.exists(dir_output):
        print('Output directory exists.')
    else:
        os.makedirs(dir_output)
        print('Output directory created.')
    return dir_output


# Elsevier
elsevier_name2doi = {}
elsevier_doi2name = {}
elsevier_output_folder = 'elsevier_mof_complete_txt'
elsevier_output_dir = check_dir(CWD, elsevier_output_folder)

# dir_data_elsevier = os.path.join(os.path.dirname(CWD), 'elsevier_corpus')  # part 1
dir_data_elsevier = os.path.join(os.path.dirname(CWD), 'elsevier_corpus_121421')  # part 2
elsevier_list_articles = [i for i in os.listdir(dir_data_elsevier) if '.xml' in i]
for elsevier_article in tqdm(elsevier_list_articles):
    file_name = elsevier_article.replace('.xml', '')
    file_dir = os.path.join(dir_data_elsevier, elsevier_article)
    article_doi = None
    # parse xml tree in current directory
    try:
        dom = parse(file_dir)
        # locate element
        data = dom.documentElement
        # try if current xml file contains a doi value
        article_doi = data.getElementsByTagName("prism:doi")[0].childNodes[0].nodeValue
    except:
        continue
    if type(article_doi) == str and len(article_doi) > 5:  # if doi exists, then
        elsevier_name2doi[file_name] = article_doi
        elsevier_doi2name[article_doi] = file_name

# save dictionaries between doi and file name
with open("elsevier_name2doi_complete.json", "w") as f:
    json.dump(elsevier_name2doi, f)
with open("elsevier_doi2name_complete.json", "w") as f:
    json.dump(elsevier_doi2name, f)


def all_print(para_list):
    L = []
    for para in para_list:
        if para.nodeValue is None:
            L.extend(all_print(para.childNodes))
        else:
            abs = re.sub("[\>]\s+[\<]", "", para.nodeValue)
            abs = re.sub("[\<].*?[\>]", "", abs)
            abs = re.sub("(\r\n|\r|\n)[\s]*", "", abs)
            L.append(abs)
    return L


def elsevier_reader(data):
    if_written = False
    if any(['originalText' in i.localName for i in data.childNodes]):  # if raw text is present in current xml
        for element in data.childNodes:
            # if current xml file contains text body content (abstract is separated)
            if element.localName == 'originalText':
                print('1')
                for subelement_level1 in element.childNodes:
                    if 'doc' in subelement_level1.localName:
                        print('2')
                        for subelement_level2 in subelement_level1.childNodes:
                            if 'rawtext' in subelement_level2.localName:
                                print('3')
                                raw_text = subelement_level2.childNodes[0].nodeValue
                                if type(raw_text) == str and len(raw_text) > 10:
                                    print('4')
                                    # write identified text into txt file
                                    with open(os.path.join(elsevier_output_dir, file_name + '.txt'), 'w',
                                              encoding='utf-8') as file:
                                        file.write(raw_text.strip())
                                        if_written = True

    elif len(data.getElementsByTagName("article")) > 0:
        try:
            article = data.getElementsByTagName("article")[0]
            elements = article.getElementsByTagName("ce:sections")[0]
            para = elements.getElementsByTagName("ce:para")
        except:
            if_written = False
        article = []
        for i in para:
            L = all_print(i.childNodes)
            par = ""
            for i in L:
                par += i
            article.append(par + "\n")
        raw_text = '\n'.join(article)
        with open(os.path.join(elsevier_output_dir, file_name + '.txt'), 'w', encoding='utf-8') as file:
            file.write(raw_text.strip())
            if_written = True
    return if_written


# matched 189 mof synthesis articles
matched_doi_elsevier = list(set(list(elsevier_doi2name)).intersection(set(mof_dois_master_list)))
matched_name_elsevier = [i for i in list(elsevier_name2doi) if elsevier_name2doi[i] in matched_doi_elsevier]

# len(set(matched_name_elsevier).intersection(set([i.replace('.xml', '') for i in elsevier_list_articles])))
# the above line shows that all matched articles are in the second half of the elsevier corpus

bad_dois_elsevier = []
for elsevier_article in tqdm(matched_name_elsevier):
    file_name = elsevier_article.replace('.xml', '')
    file_dir = os.path.join(dir_data_elsevier, elsevier_article + '.xml')
    article_doi = None
    # parse xml tree in current directory
    try:
        dom = parse(file_dir)
        # locate element
        data = dom.documentElement
        # try if current xml file contains a doi value
        article_doi = data.getElementsByTagName("prism:doi")[0].childNodes[0].nodeValue
    except:
        continue
    if type(article_doi) == str and len(article_doi) > 5:  # if doi exists, then
        result = elsevier_reader(data)
        if not result:
            bad_dois_elsevier += [article_doi]


# RSC

def extract_doi_and_text(file_path):
    # Read the HTML file
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Extract the DOI number from meta tag or DOI link
    doi_meta = soup.find('meta', attrs={'name': 'DC.Identifier', 'scheme': 'doi'})
    doi = doi_meta['content'] if doi_meta else None

    # Extract plain text content from the HTML
    text_content = soup.get_text(separator=' ', strip=False)

    return doi, text_content


# get all file names first
rsc_file_names = []
rsc_base_dir = os.path.join(os.path.dirname(CWD), 'RSC')

rsc_folders = [i for i in os.listdir(rsc_base_dir) if
               os.path.isdir(os.path.join(rsc_base_dir, i))]

for rsc_folder in rsc_folders:
    rsc_file_dir = os.path.join(rsc_base_dir, rsc_folder, '10.1039')
    rsc_file_names += os.listdir(rsc_file_dir)

rsc_file_names = list(dict.fromkeys(rsc_file_names))

rsc_matched_mof_articles = []
for rsc_file in rsc_file_names:
    temp_rsc_part_doi = rsc_file.split('.')[0]
    for mof_doi in mof_dois_master_list:
        if temp_rsc_part_doi.lower() in mof_doi.lower():
            rsc_matched_mof_articles += [rsc_file]

rsc_output_folder = 'rsc_mof_complete_txt'
rsc_output_dir = check_dir(CWD, rsc_output_folder)


rsc_doi2name = {}
count = 0
rsc_bad_files = []
for rsc_folder in rsc_folders:
    rsc_file_dir = os.path.join(rsc_base_dir, rsc_folder, '10.1039')
    rsc_file_names = [i for i in os.listdir(rsc_file_dir)]
    temp_matched_files = [i for i in rsc_file_names if i in rsc_matched_mof_articles]
    for tmp_matched_file in tqdm(temp_matched_files):
        tmp_file_name = tmp_matched_file.split('.')[0]
        tmp_file_dir = os.path.join(rsc_file_dir, tmp_matched_file)
        try:
            doi, text_content = extract_doi_and_text(tmp_file_dir)
            if doi is not None:
                rsc_doi2name[doi] = tmp_matched_file
                with open(os.path.join(rsc_output_dir, tmp_file_name + '.txt'), 'w', encoding='utf-8') as file:
                    file.write(text_content.strip())
                count += 1
            else:
                rsc_bad_files += [tmp_file_dir]

        except:
            rsc_bad_files += [tmp_file_dir]
            continue


