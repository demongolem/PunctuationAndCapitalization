'''
Created on Nov 19, 2020

@author: Mendy
'''
from keras.models import load_model

def predict(model_name, X_test):
    model = load_model(model_name)

if __name__ == '__main__':
    X_test = None
    model_name = None
    predict(model_name, X_test)