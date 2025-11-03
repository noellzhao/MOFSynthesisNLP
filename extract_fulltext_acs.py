import os
from tqdm import tqdm
import xml.etree.ElementTree as ET
import json
import pandas as pd
import re

# get current working directory
cwd = os.getcwd()
# get directory of where data are stored
dir_raw_data = os.path.join(os.path.dirname(cwd), 'ACS_xmls')
# define where output data will be stored
dir_output = os.path.join(os.path.dirname(cwd), 'acs_mof_complete_json_revised')
# check if output directory exists; create one if it doesn't
if os.path.exists(dir_output):
    print('Output directory exists.')
else:
    os.makedirs(dir_output)
    print('Output directory created.')

# get a list of all target dois across multiple sources (e.g., acs, rsc, springer nature and etc)
df_mof_master = pd.read_csv('all_MOFs_KG.csv')
target_dois_all_source = [doi.lower() for doi in list(df_mof_master['DOI']) if type(doi) == str]

# read json file and get dois related to ACS publisher
with open('acs_doi2name_complete.json', 'r') as f:
    # load pre-processed dictionary
    acs_doi2name = json.load(f)
acs_all_dois = [i.lower() for i in list(dict.fromkeys(acs_doi2name))]
# get the list of target articles in ACS collection
mof_doi_acs = list(set(target_dois_all_source).intersection(set(acs_all_dois)))

test_xml = os.path.join(dir_raw_data, 'xml1', 'ic5b00039.xml')


def extract_doi(xml_file_path):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    # look for <article-id> with attribute pub-id-type="doi"
    doi_element = root.find('.//article-id[@pub-id-type="doi"]')
    if doi_element is not None:
        return doi_element.text.strip()
    return None


def contains_full_text(xml_file_path):
    """
    Determine if an XML file from ACS publisher contains full-text content.
    :param xml_file_path: input path to the input XML file
    :return: True if contains full text; otherwise return False
    """
    # parse XML file and get the root from XML structure
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    body = root.find('.//body')  # check for <body> element
    if body is not None:
        sections = body.findall('.//sec')  # check for <sec> section within <body>
        paragraphs = body.findall('.//p')  # check for <p> section within <body>
        # if <body> element contains either sections or paragraphs, assume full text
        if sections or paragraphs:
            return True
    # if no sections or paragraphs found, or <body> component is missing then assume no full text
    return False


def text_cleaning(text):
    """
    Clean text extracted from ACS by replacing newlines with spaces and reducing repetitive space to one.
    :param text: str
    :return: cleaned string
    """
    if text:
        text = text.replace('\n', ' ')  # replace newline with space
        text = re.sub(r'\s+', ' ', text)  # reduce repetitive spaces to one
        return text.strip()  # trim leading and trailing spaces
    return text


def extract_section_text(section):
    """
    recursively extract text from sections and their nested sections
    :param section: a section component in given XML file
    :return: texts in dictionary/json
    """
    content = {}
    title = section.find('title')
    if title is not None:
        content["Title"] = text_cleaning(''.join(title.itertext()))
    else:
        content["Title"] = None
    paragraphs = [''.join(p.itertext()).strip() for p in section.findall('p')]
    content["Paragraphs"] = [text_cleaning(paragraph) for paragraph in paragraphs]

    # process nested sections
    nested_sections = section.findall('sec')
    if nested_sections:
        content['Subsection'] = [extract_section_text(sub_sec) for sub_sec in nested_sections]
    return content


def extract_full_text_w_titles(xml_file_path):
    """
    extract full text stored in XMLs from ACS, including titles and section names
    :param xml_file_path: path to input XML file
    :return: a dictionary/json file containing full text content of input XML file
    """
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    # create an empty dictionary file to store output data
    full_text = {}
    # get article title
    title = root.find('.//article-meta/title-group/article-title')
    if title is not None:
        full_text["article_title"] = ''.join(title.itertext()).strip()
    # extract abstract
    abstract = root.find('.//abstract')
    if abstract is not None:
        full_text["abstract"] = text_cleaning(''.join(abstract.itertext()))

    # extract body sections
    body = root.find('.//body')
    if body is not None:
        sections = body.findall('sec')
        full_text["sections"] = [extract_section_text(section) for section in sections]

    return full_text


# acs_test = extract_full_text_w_titles(test_xml)
count_mof = 0
count_mof_fulltext = 0
xml_folders = [i for i in os.listdir(dir_raw_data) if 'xml' in i]
for xml_folder in xml_folders:
    temp_dir_folder = os.path.join(dir_raw_data, xml_folder)
    temp_list_xmls = [i for i in os.listdir(temp_dir_folder) if '.xml' in i]
    for temp_xml in tqdm(temp_list_xmls):
        temp_dir_xml = os.path.join(temp_dir_folder, temp_xml)
        try:
            temp_doi = extract_doi(temp_dir_xml)
        except:
            continue
        if temp_doi.lower() in mof_doi_acs:
            count_mof += 1
            if contains_full_text(temp_dir_xml):
                acs_fulltext = extract_full_text_w_titles(temp_dir_xml)
                acs_fulltext['doi'] = temp_doi
                temp_filename = temp_xml.replace('.xml', '')
                with open(os.path.join(dir_output, "{}.json".format(temp_filename)), 'w') as f:
                    json.dump(acs_fulltext, f)
                count_mof_fulltext += 1
