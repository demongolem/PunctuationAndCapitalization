from collections import Counter
from itertools import chain
from keras.models import Sequential, load_model
from keras.layers import Embedding, LSTM, Dense
import numpy as np
from sklearn.metrics import confusion_matrix

unknown_token = '<unk>'

def train(X_train, y_train, y_column_name, epochs=5, model_name=None):    
    
    print(X_train.columns)
    
    list1 = X_train['prev_char'].tolist()
    list2 = X_train['curr_char'].tolist()
    list3 = X_train['next_char'].tolist()
    list4 = X_train['id'].tolist()

    combined_list = []
    for l1, l2, l3, l4 in zip(list1, list2, list3, list4):
        combined_list.append(l1)
        if l3 == '<END>':
            combined_list.append(l2)
            combined_list.append(l3)
            
    word2index = {unknown_token: 0}
    index2word = [unknown_token]
    word2index_y = {unknown_token: 0}
    index2word_y = [unknown_token]

    counter = Counter(combined_list)
    for word, count in counter.items():
        index2word.append(word)
        word2index[word] = len(word2index)

    X_train['prev_char'] = X_train['prev_char'].transform(lambda x: word2index[x]).copy(deep=False)
    X_train['curr_char'] = X_train['curr_char'].transform(lambda x: word2index[x]).copy(deep=False)
    X_train['next_char'] = X_train['next_char'].transform(lambda x: word2index[x]).copy(deep=False)
    print(X_train.head())

    print(X_train.columns)
    
    counter = Counter(y_train[y_column_name].tolist())
    for word, count in counter.items():
        index2word_y.append(word)
        word2index_y[word] = len(word2index_y)

    y_train[y_column_name] = y_train[y_column_name].transform(lambda x: word2index_y[x]).copy(deep=False)
    print(y_train.head())

    with open('models/word2index.txt', 'w', encoding='utf-8') as f:
        print(word2index, file=f)
    with open('models/word2index_y.txt', 'w', encoding='utf-8') as f:
        print(word2index_y, file=f)

    del X_train['id']
    del y_train['id']

    num_classes = len(word2index)
    maxlen = len(X_train.columns)

    # num_output_classes = len(word2index_y)
    num_output_classes = 1
    
    embedding_size = 50
    
    model = Sequential()
    model.add(Embedding(num_classes, embedding_size, input_length = maxlen))
    model.add(LSTM(256, return_sequences = True))
    model.add(LSTM(256))
    model.add(Dense(num_output_classes, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

    print(model.summary())    
    
    model.fit(X_train, y_train, epochs=epochs)    
    model.save(model_name)

def predict(model_name, X_test, y_test = None, y_name = None, word2index= None, word2index_y= None):
    model = load_model(model_name)
    X_test['prev_char'] = X_test['prev_char'].transform(lambda x: word2index[unknown_token] if x not in word2index else word2index[x]).copy(deep=False)
    X_test['curr_char'] = X_test['curr_char'].transform(lambda x: word2index[unknown_token] if x not in word2index else word2index[x]).copy(deep=False)
    X_test['next_char'] = X_test['next_char'].transform(lambda x: word2index[unknown_token] if x not in word2index else word2index[x]).copy(deep=False)
    if y_test is not None:
        y_test[y_name] = y_test[y_name].transform(lambda x: word2index_y[unknown_token] if x not in word2index_y else word2index_y[x]).copy(deep=False)
        y_pred = model.predict(np.asarray(X_test[['prev_char', 'curr_char', 'next_char']]).astype(np.int16), verbose=2)
        matrix = confusion_matrix(y_test[y_name], y_pred)
        return (y_pred, matrix)
    else:
        y_pred = model.predict(np.asarray(X_test[['prev_char', 'curr_char', 'next_char']]).astype(np.int16), verbose = 2)
        return (y_pred, None)