from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier

def train(X_train, y_train, model_name=None):
    #Create a Gaussian Classifier
    clf=RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)

    #Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train,y_train.values.ravel())
    
    if model_name:
        dump(clf, model_name)

def predict(model_name, X_test):
    clf = load(model_name)
    y_pred = clf.predict(X_test)
    return y_pred