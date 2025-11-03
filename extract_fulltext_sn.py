import os
from tqdm import tqdm
import xml.etree.ElementTree as ET
import json
import pandas as pd

# get current working directory
cwd = os.getcwd()
# get directory of where data are stored
dir_raw_data = os.path.join(os.path.dirname(cwd), 'xintong_sn_all')
# define where output data will be stored
dir_output = os.path.join(os.path.dirname(cwd), 'sn_mof_complete_txt')
# check if output directory exists; create one if it doesn't
if os.path.exists(dir_output):
    print('Output directory exists.')
else:
    os.makedirs(dir_output)
    print('Output directory created.')

df_mof_master = pd.read_csv('all_MOFs_KG.csv')
target_dois_all_source = [doi.lower() for doi in list(df_mof_master['DOI']) if type(doi) == str]


def contains_full_texts(xml_file_path):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # check for key elements associated with full text
    body = root.find('.//body')  # check for <body> element
    if body is not None:
        sections = body.findall('.//sec')  # check for <sec> within <body>
        paragraphs = body.findall('.//p')  # check for <p> within <body>

        # if <body> contains sections or paragraphs, assume full text
        if sections or paragraphs:
            return True
    # if no key elements detected, assume no full text included
    return False


def extract_full_texts(xml_file_path):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    full_text = {}

    # search the title of current article if exists
    title = root.find('.//article-meta/title-group/article-title')
    if title is not None:
        full_text['Title'] = title.text

    # extract abstract
    abstract = root.find('.//abstract')
    if abstract is not None:
        full_text['Abstract'] = '\n'.join(abstract.itertext()).strip()

    body = root.find(".//body")
    if body is not None:
        sections = []
        for section in body.findall(".//sec"):
            section_title = section.find("title")
            paragraphs = [''.join(p.itertext()).strip() for p in section.findall('p')]
            sections.append({
                'section_title': section_title.text if section_title is not None else None,
                'paragraphs': paragraphs
            })
        full_text['Sections'] = sections
    return full_text


def extract_doi(xml_file_path):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # search for the DOI element
    doi_element = root.find('.//article-id[@pub-id-type="doi"]')
    if doi_element is not None:
        return doi_element.text
    return None


# test_xml = '/Users/xintongzhao/Documents/PycharmProjects/xintong_sn_all/resw1_2_4/10.1007\\s10822-018-0111-4.xml'


xml_folders = [i for i in os.listdir(dir_raw_data) if 'res' in i]
num_full_text = 0
num_total_mof = 0
for xml_folder in xml_folders:
    temp_dir = os.path.join(dir_raw_data, xml_folder)
    temp_xmls = [i for i in os.listdir(temp_dir) if '.xml' in i]
    for temp_xml in tqdm(temp_xmls):
        # get the directory of current input xml file
        temp_dir_xml = os.path.join(temp_dir, temp_xml)
        try:
            temp_doi = extract_doi(temp_dir_xml)
        except:
            continue
        if temp_doi is not None and temp_doi.lower() in target_dois_all_source:
            num_total_mof += 1
            if contains_full_texts(temp_dir_xml):  # if the current input xml has full text content
                # then we add the count of full-text data by 1
                num_full_text += 1
                full_article = extract_full_texts(temp_dir_xml)
                full_article['doi'] = temp_doi
                with open(os.path.join(dir_output, '{}.json'.format(temp_xml.replace('.xml', ''))), 'w', encoding='utf-8') as f:
                    json.dump(full_article, f)
