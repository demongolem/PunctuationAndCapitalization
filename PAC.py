# -*- coding: utf-8 -*-

# imports
from ml import RandomForest, KerasLSTM, Bert
import numpy as np
from os import path
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
import string
import sys
sys.path.append(path.abspath('../DataProject'))

from df_utils.TrainTestSplit import my_train_test_split
from ds_read.NewswireDataset import read_into_pandas
from file_utils.ObjectReaders import fetch_dictionary
from nlp_utils.CharacterUtils import create_df_from_text_lists
from spacy_utils.SpacyUtils import tokenize_column

def read_in_data():
    # read in data
    df = read_into_pandas()
    return df

def manipulate_data(df, csv_file):
    # manipulate data / data features
    # .replace('\n', ' ').replace('\t', ' ').strip()
    df['text_bare'] = [t.lower().translate(str.maketrans('', '', string.punctuation)) for t in df['text']]
    df['bare_tokens'] = tokenize_column(df, 'text_bare')
    print(df.columns)
    
    all_texts = df['text'].tolist()
    all_bare_texts = df['text_bare'].tolist()
    all_ids = df['id'].to_list()
    
    df_punct = create_df_from_text_lists(all_texts, all_bare_texts, all_ids)
    df_punct.to_csv(csv_file)
    
    return df_punct

def form_X(df_punct):
    X = df_punct[["prev_char", "curr_char", "next_char"]]
    X.to_csv('dataframes/X.csv')
    return X

def form_y(df_punct, y_name):
    y = df_punct[[y_name]]
    return y

def split_data_keras(df_punct, y_name):
    X_train_keras,X_test_keras,y_train_keras,y_test_keras = my_train_test_split(df_punct[["id", "prev_char", "curr_char", "next_char"]], df_punct[["id", y_name]])
    X_train_keras.to_csv('dataframes/X_train_keras.csv')
    y_train_keras.to_csv('dataframes/y_train_keras.csv')
    return (X_train_keras, y_train_keras, X_test_keras, y_test_keras)

def evaluate_keras(y_name, X_test_keras, y_test_keras):
    word2index = fetch_dictionary('models/word2index.txt')
    word2index_y = fetch_dictionary('models/word2index_y.txt')
    (y_pred_keras, cm) = KerasLSTM.predict('models/LSTM_' + y_name + '.joblib', X_test_keras, 
                                     y_test=y_test_keras, y_name = y_name, 
                                     word2index = word2index, word2index_y = word2index_y)
    y_pred_keras = pd.DataFrame(y_pred_keras, columns=[y_name])
    y_test_keras = pd.DataFrame(y_test_keras, columns=[y_name])
    y_test_keras.reset_index(drop=True, inplace=True)
    y_comparison = pd.DataFrame(np.where(y_pred_keras == y_test_keras, 1, 0), columns=['correct'])
    y_pred_keras.columns = ['pred']
    y_test_keras.columns = ['truth']
    res_df = pd.concat([y_pred_keras, y_test_keras, y_comparison], axis=1)
    res_df.to_csv('results/Y_LSTM_results_' + y_name + '.csv')
    
    if y_name == 'next_punct':
        print(f1_score(y_test_keras, y_pred_keras, average=None, pos_label=True))
    else:
        print(f1_score(y_test_keras, y_pred_keras, average="binary", pos_label=True))

    labels = unique_labels(y_test_keras, y_pred_keras)
    conf_df = pd.DataFrame(confusion_matrix(y_test_keras, y_pred_keras, labels=labels))
    conf_df.index.name = 'True labels'

    #cm = confusion_matrix(y_test_keras, y_pred_keras)
    #with open('results/Y_LSTM_confusion_matrix_' + y_name +'.txt', 'w') as fs:
    #    fs.write(str(cm))    

    conf_df.to_csv('results/Y_LSTM_confusion_matrix_' + y_name +'.txt')

def main(labels, methods, epochs):
    df = read_in_data()
    
    df_punct = manipulate_data(df, 'dataframes/df_punct.csv')
    
    X = form_X(df_punct)
    
    for y_name in labels:
        print('Working with ' + y_name)
        y = form_y(df_punct, y_name)
    
        for method in methods:
    
            if 'RandomForest' == method:
                X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state = 0)
                print('Random Forest begin')
                RandomForest.train(X_train, y_train, 'models/RF_' + y_name + '.joblib')
                print('Random Forest end')
                print('Random Forest begin')
                y_pred = RandomForest.predict('models/RF_' + y_name + '.joblib', X_test)
                y_pred = pd.DataFrame(y_pred, columns=[y_name])
                y_test = pd.DataFrame(y_test, columns=[y_name])
                y_test.reset_index(drop=True, inplace=True)
                y_comparison = pd.DataFrame(np.where(y_pred == y_test, 1, 0), columns=['correct'])
                y_pred.columns = ['pred']
                y_test.columns = ['truth']
                res_df = pd.concat([y_pred, y_test, y_comparison], axis=1)
                res_df.to_csv('results/Y_results_' + y_name + '.csv')
                
                if y_name == 'next_punct':
                    print(f1_score(y_test, y_pred, average=None, pos_label=True))
                else:
                    print(f1_score(y_test, y_pred, average="binary", pos_label=True))
                cm = confusion_matrix(y_test, y_pred)
                with open('results/Y_confusion_matrix_' + y_name +'.txt', 'w') as fs:
                    fs.write(str(cm))
                print('Random Forest end')
            elif 'BERT' == method:
                # this path obviously has not been fully developed
                print('BERT begin')
                Bert.train(X_train_bert, y_train_bert, y_name, 'models/BERT_' + y_name + '.joblib')
                print('BERT LSTM end')
            elif 'Keras' == method:
                (X_train_keras, y_train_keras, X_test_keras, y_test_keras) = split_data_keras(df_punct, y_name)

                print('Keras LSTM train begin')
                KerasLSTM.train(X_train_keras, y_train_keras, y_name, epochs=epochs, model_name='models/LSTM_' + y_name + '.joblib')
                print('Keras LSTM train end')
                
                print('Keras LSTM test begin')
                evaluate_keras(y_name, X_test_keras, y_test_keras)
                print('Keras LSTM test end')
            else:
                print('Unknown method ' + method)
    
if __name__ == "__main__":
    #labels = ['next_punct', 'is_upper']
    labels = ['next_punct']
    methods = ['Keras']
    epochs = 5
    main(labels,  methods, epochs)