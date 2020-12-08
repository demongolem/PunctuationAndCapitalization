from os import path
import sys
sys.path.append(path.abspath('../DataProject'))

from ds_read.NewswireDataset import read_into_pandas
from df_utils.LabelUtils import  replace_column_with_label_representation

import numpy as np
import pandas as pd
import time
import torch

from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

def main():
    
    # 1 get data into dataframe
    df = read_into_pandas()
    (mlb_category, df) = replace_column_with_label_representation(df, 'category', 'category_int')
    df_train, df_test = train_test_split(df, test_size=0.2)    

    # 2 transform into BERT format
    df_bert = pd.DataFrame({
        'id':df_train['id'],
        'label':df_train['category_int'],
        'alpha':['a']*df_train.shape[0],
        'text': df_train['text'].str[:512].replace(r'\n', ' ', regex=True)
    })
    df_bert_train, df_bert_dev = train_test_split(df_bert, test_size=0.01)
    df_bert_test = pd.DataFrame({
        'id':df_test['id'],
        'text': df_test['text'].str[:512].replace(r'\n', ' ', regex=True)
    })
    # Saving dataframes to .tsv format as required by BERT
    df_bert_train.to_csv('../datasets/Newswire_BERT/train.tsv', sep='\t', index=False, header=False)
    df_bert_dev.to_csv('../datasets/Newswire_BERT/dev.tsv', sep='\t', index=False, header=False)
    df_bert_test.to_csv('../datasets/Newswire_BERT/test.tsv', sep='\t', index=False, header=False)

    # 3 load pretrained model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', return_dict=True)

    # 4 transform
    tokenized = df_bert_train['text'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

    print('Padding')
    
    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)
    
    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
    
    print('Shape after padding ' + str(np.array(padded).shape))
    
    attention_mask = np.where(padded != 0, 1, 0)
    attention_mask.shape
    
    input_ids = torch.tensor(padded)
    attention_mask = torch.tensor(attention_mask).to('cuda:0')
    
    print('Embedding model start')

    model.train()
    
    with torch.no_grad():
        input_ids = input_ids.clone().detach().to(torch.int64).to('cuda:0')
        model = model.to('cuda:0')
        labels = torch.tensor(df_bert_train['label'].values).to(torch.int64).to('cuda:0')
        print(labels)
        last_hidden_states = model(input_ids, attention_mask=attention_mask, labels=labels)
        print(model)
        model.save_pretrained('models/BERT1')

if __name__ == '__main__':
    main()