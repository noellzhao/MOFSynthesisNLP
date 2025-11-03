import json
import os
import pandas as pd
import nltk
from nltk import sent_tokenize
from collections import Counter

# get current working directory of project
cwd = os.getcwd()
data_path = os.path.join(cwd, 'data')
# get a test sample
sample_anno_path = os.path.join(data_path, 'jacob_prelabeld_notswap.jsonl')
# sample_anno_path = os.path.join(data_path, 'Kyle_0802.jsonl')
# pre-processing annotation files
# remove empty columns and rows
annoObj = pd.read_json(path_or_buf=sample_anno_path, lines=True)
annoObj = annoObj.drop(columns=['Comments'])
# remove rows that are empty
annoObj = annoObj[annoObj['entities'].map(lambda d: len(d)) > 0]
annoObj = annoObj[annoObj['relations'].map(lambda d: len(d)) > 0].reset_index()

texts = annoObj['text'].tolist()
entities = annoObj['entities'].tolist()
relations = annoObj['relations'].tolist()

entity_types = []
for record in entities:
    for component in record:
        entity_types += [component['label']]
entity_types_dist = Counter(entity_types)

# to do: read all entity types and find texts with type "vessel", "acid" and "base".
# then print them out
for idx in range(len(entities)):
    for entity_record in entities[idx]:
        if entity_record['label']=='vessel':
            print("{}".format(str(idx)+' vessel'))
        elif entity_record['label']=='acid':
            print("{}".format(str(idx)+' acid'))
        elif entity_record['label']=='base':
            print("{}".format(str(idx) + ' base'))

# select idx 12 and 13




relation_types = []
for record in relations:
    for component in record:
        relation_types += [component['type']]
relation_types = list(dict.fromkeys(relation_types))

assert len(entities) == len(texts) == len(relations)

prompts_txt = []
completion_txt = []
for idx in range(len(relations)):
    temp_completion = []
    for relation in relations[idx]:
        from_id = relation['from_id']
        to_id = relation['to_id']
        from_entity = [i for i in entities[idx] if i['id'] == from_id][0]
        to_entity = [i for i in entities[idx] if i['id'] == to_id][0]

        from_entity_str = texts[idx][from_entity['start_offset']:from_entity['end_offset']].strip()
        to_entity_str = texts[idx][to_entity['start_offset']:to_entity['end_offset']].strip()
        if ',' in to_entity_str:
            to_entity_part1 = to_entity_str.split(',')[0].strip()
            to_entity_part2 = to_entity_str.split(',')[1].strip()
            temp_completion += ["{} - {}".format(from_entity_str, to_entity_part1)]
            temp_completion += ["{} - {}".format(from_entity_str, to_entity_part2)]
        else:
            temp_completion += ["{} - {}".format(from_entity_str, to_entity_str)]
    completion_txt += [temp_completion]
    # completion_txt += [' '+'\n'.join(temp_completion) + " END"]
    # the following is doing sentence level prompt and completion processing
    prompt_sentence_level = sent_tokenize(texts[idx].strip())
    relation_sentence_lv = []
    for syn_sent in prompt_sentence_level:
        temp_relations = []
        for relation in temp_completion:
            ent1 = relation.split(' , ')[0]
            ent2 = relation.split(' , ')[1]
            if ent1 in syn_sent and ent2 in syn_sent:
                temp_relations += [relation]
        relation_sentence_lv += [temp_relations]
    assert len(prompt_sentence_level) == len(relation_sentence_lv)
    for i in range(len(prompt_sentence_level)):
        prompts_txt += [prompt_sentence_level[i].strip() + "\n\n###\n\n"]
        completion_txt += [' ' + '\n'.join(relation_sentence_lv[i]) + " END"]

assert len(prompts_txt) == len(completion_txt)

final_prompts = []
for ind_anno in range(len(completion_txt)):
    prompt = {}
    prompt["prompt"] = prompts_txt[ind_anno]
    prompt["completion"] = completion_txt[ind_anno]
    final_prompts += [prompt]

with open("mof_relation_pairs_with_formatting_sentence.jsonl", 'w') as f:
    for item in final_prompts:
        f.write(json.dumps(item) + "\n")


