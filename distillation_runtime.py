from transformers import TextClassificationPipeline
import pandas as pd
from datasets import Dataset
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import random
import time


# 1. Load model and tokenizer
model_path = "./bert_binary_syn_classifier"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# 2. Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 3. Create the pipeline manually
classifier = TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    return_all_scores=True  # optional, shows probs for all labels
)

def get_binary_label(prediction_list):
    top_label = max(prediction_list, key=lambda x: x['score'])['label']
    return 1 if top_label == 'LABEL_1' else 0


test_csv = pd.read_csv("Training_SynParas.csv")
test_csv.dropna(subset=['Paragraph'], inplace=True)
val_texts = list(test_csv['Paragraph'])
val_labels = list(test_csv['if_synthesis'])



start = time.time()

pred_labels = classifier(val_texts,truncation=True, max_length=500)
pred_labels = [get_binary_label(pred) for pred in pred_labels]

end = time.time()

print("Runtime:", end - start, "seconds")