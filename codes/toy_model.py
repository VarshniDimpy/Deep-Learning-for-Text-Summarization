import pandas as pd
import transformers
import numpy as np
import torch
from transformers import EncoderDecoderModel, BertTokenizer
from transformers import BertConfig,EncoderDecoderConfig,EncoderDecoderModel
from transformers import AdamW
optimizer = AdamW(model.parameters(), lr=3e-5)

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased',model_max_length = 2000)
train = pd.read_csv('train_clean_withNumbers_withEngPhrases.csv',lineterminator='\n')

articles = list(train['article'])
summaries = list(train['summary'])
tokenized_encoder = []
tokenized_decoder = []
count =0
for (a,s) in zip(articles,summaries) :
    article=tokenizer.encode(a.replace('\n',''),add_sepcial_tokens=True)
    summary=tokenizer.encode(s.replace('\n',''),add_sepcial_tokens=True)  
    tokenized_encoder.append(article)
    tokenized_decoder.append(summary)
    count+=1
    if count == 100:
      break

max_len = 0
for i in tokenized_encoder:
    if len(i) > max_len:
        max_len = len(i)
print(max_len)
max_len = 0
for i in tokenized_decoder:
    if len(i) > max_len:
        max_len = len(i)
print(max_len)
padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized_decoder])
input_ids_ = torch.LongTensor(np.array(padded))
attention_mask_ = np.where(padded != 0, 1, 0)
padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized_encoder])
input_ids = torch.LongTensor(np.array(padded))
attention_mask = np.where(padded != 0, 1, 0)

attention_mask=torch.Tensor(attention_mask)
attention_mask_=torch.Tensor(attention_mask_)

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
#model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-multilingual-cased', 'bert-base-multilingual-cased')

config_encoder = BertConfig()
config_decoder = BertConfig()

config_encoder.max_length = 1566
config_decoder.max_length = 101

config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)

model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-multilingual-cased', 'bert-base-multilingual-cased',config=config) # initialize Bert2Bert

model.to(device)

for i in range(10):
  optimizer.zero_grad()
  loss= model(input_ids=input_ids[:1].to(device), decoder_input_ids=input_ids_[:1].to(device), lm_labels=input_ids_[:1].to(device),attention_mask=attention_mask[:1].to(device),decoder_attention_mask = attention_mask_[:1].to(device))[:1]
  print(loss[0].item())
  loss[0].backward()
  optimizer.step()