from os import path
import sys
sys.path.append(path.abspath('../DataProject'))

from ds_read.NewswireDataset import read_into_pandas
from df_utils.LabelUtils import  replace_column_with_label_representation
from df_utils.DiskUtils import dataframe_to_disk

import ktrain
from ktrain import text

from joblib import dump, load
from sklearn.model_selection import train_test_split

# 1 get data into dataframe
df = read_into_pandas()
(lb_category, df) = replace_column_with_label_representation(df, 'category', 'category_int')
df_train, df_test = train_test_split(df, test_size=0.2)    
dataframe_to_disk(df_train, '../datasets/Newswire/train.csv')
dataframe_to_disk(df_test, '../datasets/Newswire/test.csv')
dump(lb_category, '../datasets/Newswire/cat_encoder.joblib') 
exit(1)

train_X = df_train['text'].values
train_y = df_train['category_int'].values
test_X = df_test['text'].values
test_y = df_test['category_int'].values

# 2 (distil)bert version
model_name = 'distilbert-base-uncased'
class_names = lb_category.classes_
trans = text.Transformer(model_name, maxlen=512, class_names=class_names)

# 3 train
train_data = trans.preprocess_train(train_X, train_y)
test_data = trans.preprocess_test(test_X, test_y)
model = trans.get_classifier()
learner = ktrain.get_learner(model, train_data, val_data=test_data, batch_size=16, use_multiprocessing=True)
#best_lr = learner.lr_find(show_plot=False, max_epochs=1)
best_lr = 0.0001
learner.fit_onecycle(best_lr, epochs=1)
cm = learner.validate(class_names=class_names)
print(cm)
tl = learner.view_top_losses(n=5, preproc=trans)
print(tl)

# 4 test
predictor = ktrain.get_predictor(learner.model, preproc=trans)
test_string = 'The Chiefs beat the Ravens last night 32-25.'
prediction = predictor.predict(test_string)
print(prediction)
expl = predictor.explain()
print(expl)
predictor.save('models/ktrain1')