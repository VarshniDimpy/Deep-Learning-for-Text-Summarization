# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 16:37:09 2020

@author: Lenovo
"""

# change cwd to directory where there is train.csv file
# create new folder new_dataset & create 2 subfolders articles & summaries inside it
# all the numbers are changed to <NUM> tag
# all english words & tags are removed
# hindi punctuations are also removed
# final txt files consist of hindi words separated by space

import pandas as pd
import string
import re

def preprocessed(article,test_list1,test_list2):
    num=[chr(x) for x in range(ord('0'), ord('9') + 1)]
    article=article.replace('\n',' ')
    article=article.replace('\r',' ')
    article=article.replace('\t',' ')
    article=article.replace('\xa0',' ')
    article=article.replace('\u200d',' ')
    article=article.replace('‘',' ')
    article=article.replace('’',' ')
    article=article.strip()
    article=article.translate(str.maketrans("","", string.punctuation))# punctuation removal - https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string
    #remove words starting with http
    sentences=article.split("। ")
    new_article=""
    for sentence in sentences:
        s=re.sub(r'^https?:\/\/.*[\r\n]*', '',sentence, flags=re.MULTILINE)#remove urls
        tokens=s.split(" ")
        new_sentence=""
        for token in tokens:
            if len(token)>=1:
                if (token[0] in test_list1) or (token[0] in test_list2):
                    pass
                elif (token[0]in num):
                    new_sentence+="<NUM> "
                else:
                    new_sentence+=token
                    new_sentence+=" "
        new_article+=new_sentence
        new_article=new_article.replace('।','')
    return new_article
    
#path='C:\\Users\\Lenovo\\Documents\\DL Project\\'
#filename='train.csv'

train_df=pd.read_csv('train.csv',lineterminator='\n')

all_articles=train_df['article']
all_summaries=train_df['summary']
test_list1 = [chr(x) for x in range(ord('a'), ord('z') + 1)]
test_list2 = [chr(x) for x in range(ord('A'), ord('Z') + 1)]

new_articles=[]
new_summaries=[]
for i in range(len(all_articles)):
    summary=all_summaries[i]
    if str(summary)!='nan':
        article=all_articles[i]
        new_summary=preprocessed(summary,test_list1,test_list2)
        new_article=preprocessed(article,test_list1,test_list2)
        filename=str(i)+".txt"
        path_article="./new_dataset/articles/"+filename
        #file1=open(path,"w+")
        with open(path_article, "w+", encoding="utf-8") as f:
            f.write(new_article)
        path_summary="./new_dataset/summaries/"+filename
        #file1=open(path,"w+")
        with open(path_summary, "w+", encoding="utf-8") as f:
            f.write(new_summary)
        new_articles.append(new_article)
        new_summaries.append(new_summary)
    



