from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
import pandas as pd
import numpy as np
import xgboost as xgb
from preprocess import factorize_category_both
from sklearn.cross_validation import train_test_split


high_corr_columns = ['v8', 'v23', 'v25', 'v36', 'v37', 'v46', 'v51', 'v53', 'v54', 'v63', 'v73', 'v81', 'v82', 'v89', 'v92',
      'v95', 'v105', 'v107', 'v108', 'v109', 'v116', 'v117', 'v118', 'v119', 'v123', 'v124', 'v128']
train_columns_to_drop = ['ID', 'target'] + high_corr_columns
print train_columns_to_drop
test_columns_to_drop = ['ID'] + high_corr_columns
num_classes = 2
print 'load data'
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
submission = pd.read_csv('./data/sample_submission.csv')
train_id = train['ID'].values
train_target = train['target'].values
train_feat = train.drop(train_columns_to_drop, axis=1)
test_feat = test.drop(test_columns_to_drop, axis=1)
factorize_category_both(train_feat, test_feat)
train_feat.fillna(-1,inplace=True)
test_feat.fillna(-1,inplace=True)

#X_train, X_valid, y_train, y_valid = train_test_split(
#    train_feat.values, train_target, test_size=0.2, random_state=42)
batch_size = 256 
nb_epoch = 2

model = Sequential()
model.add(Dense(2000, input_dim=train_feat.shape[1], init='uniform', activation='relu'))
model.add(Dense(2000, input_dim=train_feat.shape[1], init='uniform', activation='relu'))
model.add(Dense(2000, input_dim=train_feat.shape[1], init='uniform', activation='relu'))
model.add(Dense(2000, input_dim=train_feat.shape[1], init='uniform', activation='relu'))
#model.add(Dropout(0.2))
#model.add(Dense(512))
#model.add(Activation('relu'))
#model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.summary()

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

#history = model.fit(X_train, y_train,
#                   batch_size=batch_size, nb_epoch=nb_epoch,
#                   verbose=1, validation_data=(X_valid, y_valid))

history = model.fit(train_feat.values, train_target,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_split=0.2)

print test_feat.shape
score = model.predict(test_feat, verbose=1)
print score
