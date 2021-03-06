# -*- coding: utf-8 -*-
"""Untitled3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Cec9uzCf0lRck-hvbxquMdpZ9v2EjyLD
"""
'''
from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/My Drive/DL_final')

!pip install transformers
'''
import pandas as pd
import transformers
import numpy as np
import torch
from transformers import EncoderDecoderModel, BertTokenizer
from transformers import BertConfig,EncoderDecoderConfig,EncoderDecoderModel
from transformers import AdamW

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased',model_max_length = 2000)
train = pd.read_csv('train_clean_withNumbers_withEngPhrases.csv',lineterminator='\n')

articles = list(train['article'])
summaries = list(train['summary'])

len(articles),len(summaries)

tokenized_encoder = []
tokenized_decoder = []
count =0
for (a,s) in zip(articles,summaries) :
    article=tokenizer.encode(a.replace('\n',''),add_sepcial_tokens=True)
    summary=tokenizer.encode(s.replace('\n',''),add_sepcial_tokens=True)  
    if len(article)>512:
      continue
    tokenized_encoder.append(article)
    tokenized_decoder.append(summary)
    count+=1
    print(count," ! done")

len(tokenized_encoder),len(tokenized_decoder)

max_len = 0
for i in tokenized_encoder:
    if len(i) > max_len:
        max_len = len(i)
print(max_len)
padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized_encoder])
input_ids = torch.LongTensor(np.array(padded))
attention_mask = np.where(padded != 0, 1, 0)

max_len = 0
for i in tokenized_decoder:
    if len(i) > max_len:
        max_len = len(i)
print(max_len)
padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized_decoder])
input_ids_ = torch.LongTensor(np.array(padded))
attention_mask_ = np.where(padded != 0, 1, 0)

attention_mask=torch.Tensor(attention_mask)
attention_mask_=torch.Tensor(attention_mask_)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-multilingual-cased', 'bert-base-multilingual-cased')

config_encoder = BertConfig(max_position_embeddings=2048)
config_decoder = BertConfig(max_position_embeddings=121)

config_encoder.max_length = 1566
config_decoder.max_length = 101

config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)

model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-multilingual-cased', 'bert-base-multilingual-cased',config=config) # initialize Bert2Bert
optimizer = AdamW(model.parameters(), lr=3e-5)

model.to(device)

attention_mask.shape

from torch.utils import data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class TrainLoader(data.Dataset):

  def __init__(self,input_ids,attention_mask,input_ids_,attention_mask_):

    self.input_ids=input_ids
    self.attention_mask=attention_mask
    self.input_ids_=input_ids_
    self.attention_mask_=attention_mask_
    
    self.length=len(self.input_ids)


  def __getitem__(self,idx):
      
      input_ids_e=self.input_ids[idx]
      attention_mask_e=self.attention_mask[idx]
      input_ids_d=self.input_ids_[idx]
      attention_mask_d=self.attention_mask_[idx]
      return input_ids_e,attention_mask_e,input_ids_d,attention_mask_d

  def __len__(self):
      return self.length

len(input_ids)

train_target=TrainLoader(input_ids,attention_mask,input_ids_,attention_mask_)
train_set,val_set = torch.utils.data.random_split(train_target,[30000,5070])
train_loader=torch.utils.data.DataLoader(train_set,batch_size=1,shuffle=True)
test_loader=torch.utils.data.DataLoader(val_set,batch_size=1,shuffle=False)

input_ids_e,attention_mask_e,input_ids_d,attention_mask_d = next(iter(train_loader))

input_ids_e.shape,attention_mask_e.shape

print(attention_mask.shape,input_ids.shape)

epochs = 40

for epoch in range(epochs):
  total_loss=0.0
  count=0
  for input_ids_e,attention_mask_e,input_ids_d,attention_mask_d in train_loader:
    optimizer.zero_grad()
    loss= model(input_ids=input_ids_e.to(device), decoder_input_ids=input_ids_d.to(device), lm_labels=input_ids_d.to(device),attention_mask=attention_mask_e.to(device),decoder_attention_mask = attention_mask_d.to(device))[:1]
    loss[0].backward()
    total_loss+=loss[0].item()
    optimizer.step()
    if count ==100:
      print("batch : ",count," loss : ",loss[0].iem())
    count+=1
  print(epoch," epoch loss = ",total_loss/count)

model.eval()

pred_summaries,test_summaries = [],[]

for input_ids_e,attention_mask_e,input_ids_d,attention_mask_d in test_loader:
  generated = model.generate(input_ids_e.to(device),do_sample=True,top_k=0,decoder_start_token_id=model.config.decoder.pad_token_id,max_length=121)
  output = tokenizer.decode(generated[0], skip_special_tokens=True)
  pred_summaries.append(output)
  output = tokenizer.decode(input_ids_d[0], skip_special_tokens=True)
  test_summaries.append(output)


import rouge
from rouge import Rouge
rouge=Rouge()
scores = rouge.get_scores(pred_summaries,test_summaries,avg=True)

print("scores = ", scores)

'''
import pandas as pd
import transformers
import numpy as np
import torch
from transformers import EncoderDecoderModel, BertTokenizer
from transformers import BertConfig,EncoderDecoderConfig,EncoderDecoderModel
from transformers import AdamW

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
    if len(article)>512:
      continue
    tokenized_encoder.append(article)
    tokenized_decoder.append(summary)
    count+=1
    if count==100:
      break


max_len = 0
for i in tokenized_encoder:
    if len(i) > max_len:
        max_len = len(i)
print(max_len)
padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized_encoder])
input_ids = torch.LongTensor(np.array(padded))
attention_mask = np.where(padded != 0, 1, 0)

max_len = 0
for i in tokenized_decoder:
    if len(i) > max_len:
        max_len = len(i)
print(max_len)
padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized_decoder])
input_ids_ = torch.LongTensor(np.array(padded))
attention_mask_ = np.where(padded != 0, 1, 0)

attention_mask=torch.Tensor(attention_mask)
attention_mask_=torch.Tensor(attention_mask_)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-multilingual-cased', 'bert-base-multilingual-cased')

config_encoder = BertConfig(max_position_embeddings=2048)
config_decoder = BertConfig(max_position_embeddings=2048)

config_encoder.max_length = 1566
config_decoder.max_length = 101

config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)

model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-multilingual-cased', 'bert-base-multilingual-cased',config=config) # initialize Bert2Bert
optimizer = AdamW(model.parameters(), lr=3e-5)

model.to(device)

generated = model.generate(input_ids[:1].to(device),do_sample=True,top_k=0,decoder_start_token_id=model.config.decoder.pad_token_id,max_length=101)

import os
os.listdir()

input_ids[:2].shape

for i in range(10):
  optimizer.zero_grad()
  loss= model(input_ids=input_ids[:2].to(device), decoder_input_ids=input_ids_[:2].to(device), lm_labels=input_ids_[:2].to(device),attention_mask=attention_mask[:2].to(device),decoder_attention_mask = attention_mask_[:2].to(device))[:1]
  print(loss[0].item())
  loss[0].backward()
  optimizer.step()

generated = model.generate(input_ids[:1].to(device),do_sample=True,top_k=0,decoder_start_token_id=model.config.decoder.pad_token_id,max_length=101)

generated[0].shape

output = tokenizer.decode(generated[0], skip_special_tokens=True)

output

summaries[0]

input_ids_[1]
'''