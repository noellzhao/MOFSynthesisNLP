import json
import os
import openai
from openai import OpenAI
from sklearn.metrics import classification_report
import pandas as pd
from tqdm import tqdm
import time
from sklearn.metrics import confusion_matrix
import tiktoken

# add API key
os.environ['OPENAI_API_KEY'] = '[ENTER YOUR API KEY HERE]'

client = OpenAI()

# used to compare performance
def syn_detect_comparison_only(input_paragraph):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an assistant to help me find any given paragraph is describing specific step-by-step "
                                "materials synthesis procedure. Answer 1 if you think the given paragraph is describing detailed synthesis steps; otherwise "
                                "answer 0."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": input_paragraph
                    }
                ]
            },
        ],
        response_format={
            "type": "text"
        },
        temperature=0.1,
        max_completion_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content.strip()


def syn_detect(input_paragraph):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an assistant to help me find any given paragraph is describing specific "
                                "step-by-step"
                                "synthesis procedure. Below are examples of paragraphs that describing detailed "
                                "synthesis steps,"
                                "and examples of paragraphs that are not describing specific synthesis "
                                "steps.\n\nExamples of"
                                "paragraphs that are describing a synthesis procedure step-by-step:\n\n[ENTER YOUR "
                                "EXAMPLES HERE]\n\nExample paragraphs that are not describing specific synthesis "
                                "procedures:\n\n[NEGATIVE EXAMPLES HERE (false positives, true negatives "
                                "etc.)]\n\nNow it's"
                                "your turn. Answer 1 if you think this paragraph is describing detailed synthesis "
                                "steps; otherwise"
                                "answer 0."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": input_paragraph
                    }
                ]
            },
        ],
        response_format={
            "type": "text"
        },
        temperature=0.1,
        max_completion_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content.strip()


# ========== Start of Performance Report ==========

test_csv = pd.read_csv("Training_SynParas.csv")
test_csv.dropna(subset=['Paragraph'], inplace=True)
test_paragraphs = list(test_csv['Paragraph'])
test_labels = list(test_csv['if_synthesis'])
assert len(test_paragraphs) == len(test_labels)
y_preds = []
y_runtimes = []

# =============comparative analysis only==================
for val_paragraph in tqdm(test_paragraphs):
    start_time = time.time()
    pred = syn_detect_comparison_only(val_paragraph)
    end_time = time.time()
    y_preds += [pred]
    y_runtimes += [end_time - start_time]

y_preds_int = []
for label in y_preds:
    label = label.strip()[0]
    if label.isdigit():
        y_preds_int += [int(label)]
    elif label.isalpha():
        if label.lower() == 'n':
            y_preds_int += [0]
        elif label.lower() == 'y':
            y_preds_int += [1]
    else:
        y_preds_int += [0]
# ======= End of Comparison =======

y_preds = []
y_runtimes = []
for val_paragraph in tqdm(test_paragraphs):
    start_time = time.time()
    pred = syn_detect(val_paragraph)
    end_time = time.time()
    y_preds += [pred]
    y_runtimes += [end_time - start_time]

y_preds_int = []
for label in y_preds:
    label = label.strip()[0]
    if label.isdigit():
        y_preds_int += [int(label)]
    elif label.isalpha():
        if label.lower() == 'n':
            y_preds_int += [0]
        elif label.lower() == 'y':
            y_preds_int += [1]
    else:
        y_preds_int += [0]

# print out performance report
# get confusion matrix
cm = confusion_matrix(test_labels, y_preds_int)
# print out detailed classification result for both y-true and y-preds
classification_details = ''
classification_details += "y_true\ty_pred\n"
for i in range(len(y_preds)):
    classification_details += str(test_labels[i]) + '\t' + y_preds[i] + '\n'

# get list of token length for testing set
encoding = tiktoken.get_encoding("cl100k_base")
num_tokens = [str(len(encoding.encode(para))) for para in test_paragraphs]
# write everything into a txt file
log_txt = ""
log_txt += "===Model Info===\nmodel: 'gpt-4o-2024-08-06'\n\n"
log_txt += "===Classification Performance===\n" + classification_report(test_labels, y_preds_int) + '\n\n'
log_txt += "===Runtime By Case===\n{}\n\n".format('\n'.join([str(i) for i in y_runtimes]))
log_txt += "===Classification Details===\n{}\n\n".format(classification_details)
log_txt += "===Confusion Matrix===\n{}\n\n".format(cm) + '\n\n'
log_txt += "===Paragraph Length (in Token)===\n{}\n\n".format('\n'.join(num_tokens)) + '\n\n'
with open('log-syn_detect-1222.txt', 'w', encoding='utf-8') as f:
    f.write(log_txt)

# ========== End of Performance Report ==========
# ========== Start of Synthesis Paragraphs Extraction ==========
'''
To Dos:
1. For each json from ACS, read all paragraphs and use the function above to identify all synthesis paragraphs - done for acs
    - don't forget to record both non-syn and syn paragraphs for future BERT model training - done for acs
2. Once a synthesis paragraph is identified, find its corresponding outcome MOF and link to the paragraph - done
3. For each extracted synthesis paragraph, parse them into action graph
4. For each component in the action graph, use dictionary match to figure its entity type
5. Create visualizations to show possible entity types that are not usually fully-awared by researchers (modulator, acid, vessels ...)
'''


def extract_paragraphs_w_titles(json_data, current_doi, parent_title=""):
    """
    Recursively traverse the JSON structure to extract paragraphs and their corresponding titles'
    :param current_doi: str - doi number of given article
    :param json_data: Json data (dict)
    :param parent_title: The title of the current section/subsection
    :return: a list of tuples where each tuple contains the title and a paragraph
    """
    extracted_content = []

    if isinstance(json_data, dict):
        # check for a title and paragraphs in the current section
        current_title = json_data.get("Title", parent_title)  # use parent title if no current title
        paragraphs = json_data.get("Paragraphs", [])
        for paragraph in paragraphs:
            extracted_content.append([current_title, paragraph, current_doi])
        # recursively process subsections if they exist
        if "Subsection" in json_data:
            for subsection in json_data['Subsection']:
                extracted_content.extend(extract_paragraphs_w_titles(subsection, current_doi, current_title))
    elif isinstance(json_data, list):
        for item in json_data:
            extracted_content.extend(extract_paragraphs_w_titles(item, current_doi, parent_title))
    return extracted_content


def find_output_material(input_text):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an assistant to help me note down outcome material of given synthesis "
                                "procedure.\n\nFor example, when it says \"synthesis of Zn[ibmd[(DMF)\", "
                                "then you should answer \"Zn[ibmd[(DMF)\" only; when it mentions in the synthesis "
                                "procedure saying \"the synthesized material XYZ... a mixture of solvent...\", "
                                "then you should answer \"XYZ\" only. \n\nIf there is no outcome material mentioned, "
                                "then you should answer \"None\"."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": input_text
                    }
                ]
            },
        ],
        response_format={
            "type": "text"
        },
        temperature=0.1,
        max_completion_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content.strip()


def paragraph_categorization_for_article(list_paragraph_w_title):
    """
    Categorize each paragraph in the input file whether they are related with synthesis steps. If yes,
    further identify the chemical composition of synthesis outcome
    :param list_paragraph_w_title: a list of tuples
    where the first component in each tuple is title, second component is the actual text content of paragraphs
    :return: two lists of tuples, where the first one is synthesis related and second is not synthesis related
    """
    paragraphs_not_synthesis = []
    paragraphs_are_synthesis = []
    for item in list_paragraph_w_title:
        if len(item) == 3:  # check if all information are included
            temp_paragraph = item[1]
            temp_category = syn_detect(temp_paragraph)
            if temp_category[0] == '1':
                temp_syn_record = list(item)
                temp_output = find_output_material(temp_syn_record[0])
                temp_syn_record.append(temp_output)
                paragraphs_are_synthesis += [temp_syn_record]

            else:
                paragraphs_not_synthesis += [list(item)]
    return paragraphs_are_synthesis, paragraphs_not_synthesis


def write_result(list_paragraphs, opt_filename):
    # save categorized results to txt file
    opt_str = ''
    for para in list_paragraphs:
        temp_record = '\t'.join([str(element) for element in para]) + '\n'
        opt_str += temp_record
    opt_str = opt_str.strip()
    with open("{}.txt".format(opt_filename), 'w', encoding='utf-8') as file:
        file.write(opt_str)


# get all json files from ACS and SN output folder
acs_output_name = 'acs_mof_complete_json'
sn_output_name = 'sn_mof_complete_txt'

# get the directory of above folders - ACS Processing
cwd = os.getcwd()
dir_parsed_data = os.path.join(os.path.dirname(cwd), acs_output_name)

parsed_json_acs = [i for i in os.listdir(dir_parsed_data) if '.json' in i]

# create two lists to store extracted results
overall_not_syn = []
overall_is_syn = []
# for each json file in the target directory
for json_file in tqdm(parsed_json_acs):
    temp_dir_file = os.path.join(dir_parsed_data, json_file)
    with open(temp_dir_file, 'r') as f:
        data = json.load(f)
    temp_doi = data.get('doi', '')
    if "sections" in data:
        paragraphs_w_titles = extract_paragraphs_w_titles(data["sections"], temp_doi)
    else:
        paragraphs_w_titles = extract_paragraphs_w_titles(data, temp_doi)
    if len(paragraphs_w_titles) > 0:  # if there are paragraphs existed
        temp_list_paras_syn, temp_list_paras_not_syn = paragraph_categorization_for_article(paragraphs_w_titles)
        overall_not_syn += temp_list_paras_not_syn
        overall_is_syn += temp_list_paras_syn

clear_opt_acs = [i for i in overall_is_syn if 'none' not in i[3].lower()]
write_result(clear_opt_acs, "acs_overall_clear_output1225")

# SN Processing
dir_parsed_data = os.path.join(os.path.dirname(cwd), sn_output_name)

parsed_json_sn = [i for i in os.listdir(dir_parsed_data) if '.json' in i]

# create two lists to store extracted results
overall_not_syn = []
overall_is_syn = []
# for each json file in the target directory
for json_file in tqdm(parsed_json_sn):
    temp_dir_file = os.path.join(dir_parsed_data, json_file)
    with open(temp_dir_file, 'r') as f:
        data = json.load(f)
    temp_doi = data.get('doi', '')
    if "sections" in data:
        paragraphs_w_titles = extract_paragraphs_w_titles(data["sections"], temp_doi)
    else:
        paragraphs_w_titles = extract_paragraphs_w_titles(data, temp_doi)
    if len(paragraphs_w_titles) > 0:  # if there are paragraphs existed
        temp_list_paras_syn, temp_list_paras_not_syn = paragraph_categorization_for_article(paragraphs_w_titles)
        overall_not_syn += temp_list_paras_not_syn
        overall_is_syn += temp_list_paras_syn

clear_opt_acs = [i for i in overall_is_syn if 'none' not in i[3].lower()]
write_result(clear_opt_acs, "acs_overall_clear_output1225")


# ==== Create Action Graph ====


def create_action_graph(syn_paragraph):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful assistant. Help me to find words describing synthesis actions as "
                                "they spelled in given text, then place them in order they happened. I need you to "
                                "also find all entity-entity pairs that are related with each synthesis action step. "
                                "\n\nHere is an example:\n[SYNTHESIS PARAGRAPH]\n\n[PROCEDURE WITH SEQUENCE, "
                                "SEE DISSERTATION APPENDIX]\n\nNow"
                                "it's your turn. Find the entity-entity pair from given text, and write as above.\n"
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": syn_paragraph
                    }
                ]
            },
        ],
        response_format={
            "type": "text"
        },
        temperature=0.1,
        max_completion_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content


with open('acs_overall_clear_output1225.txt', 'r', encoding='utf-8') as f:
    filtered_paragraphs = f.read()

filtered_paragraphs = filtered_paragraphs.split('\n')
paragraph_contents = []
resulted_mofs = []
related_dois = []
for record in filtered_paragraphs:
    paragraph_contents += [record.split('\t')[1]]
    resulted_mofs += [record.split('\t')[3]]
    related_dois += [record.split('\t')[2]]

codified_graphs = []
codified_dois = []
codified_mofs = []
for idx in tqdm(range(len(paragraph_contents))):
    codified_graphs += [create_action_graph(paragraph_contents[idx])]
    codified_dois += [related_dois[idx]]
    codified_mofs += [resulted_mofs[idx]]

with open('codified_synthesis.txt', 'w', encoding='utf-8') as f:
    output = ''
    for idx in range(len(codified_mofs)):
        output += codified_dois[idx] + '\t' + codified_mofs[idx] + '\t' + codified_graphs[idx] + "\nEND\n"
    f.write(output)

# ----- just for testing

with open('acs_overall_synthesis_paragraph.txt', 'r', encoding='utf-8') as f:
    acs_syn_paras = f.read()
acs_syn_paras = acs_syn_paras.split('\n')
