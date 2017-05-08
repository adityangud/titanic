import pandas as pd
import logging
import time
import numpy as np
from features import Sex, Embarked, Parch, SibSp, Pclass
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import csv
import sys

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)
start_time = time.time()
logging.info(" ----------- Start  ------------")

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
test_y = pd.read_csv('gender_submission.csv')
test_passenger_ids = list(test_y.PassengerId)
test_y = np.array(test_y.Survived)


train_pclass = np.array(train_data.Pclass)
train_sex = np.array(train_data.Sex)
train_sibsp = np.array(train_data.SibSp)
train_parch = np.array(train_data.Parch)
#train_ticket = np.array(train_data.Ticket)
train_fare = np.array(train_data.Fare)
train_embarked = np.array(train_data.Embarked)


test_pclass = np.array(test_data.Pclass)
test_sex = np.array(test_data.Sex)
test_sibsp = np.array(test_data.SibSp)
test_parch = np.array(test_data.Parch)
#train_ticket = np.array(train_data.Ticket)
test_fare = np.array(test_data.Fare)
test_embarked = np.array(test_data.Embarked)

'''
train_cabin = list(train_data.Cabin)
for i in xrange(train_cabin):
    if pd.isnull(train_cabin[i]):
        train_cabin[i] = ""
train_cabin = np.array(train_cabin)
'''

train_y = np.array(train_data.Survived)

training_feature_vectors = []
testing_feature_vectors = []

sex = Sex()
train_x = sex.fit_transform(train_sex)
training_feature_vectors.append(train_x)
test_x = sex.fit_transform(test_sex)
testing_feature_vectors.append(test_x)

sibsp = SibSp()
train_x = sibsp.fit_transform(train_sibsp)
training_feature_vectors.append(train_x)
test_x = sibsp.fit_transform(test_sibsp)
testing_feature_vectors.append(test_x)

parch = Parch()
train_x = parch.fit_transform(train_parch)
training_feature_vectors.append(train_x)
test_x = parch.fit_transform(test_parch)
testing_feature_vectors.append(test_x)


embarked = Embarked()
train_x = embarked.fit_transform(train_embarked)
training_feature_vectors.append(train_x)
test_x = embarked.fit_transform(test_embarked)
testing_feature_vectors.append(test_x)


pclass = Pclass()
train_x = pclass.fit_transform(train_pclass)
training_feature_vectors.append(train_x)
test_x = pclass.fit_transform(test_pclass)
testing_feature_vectors.append(test_x)


final_training_feature_vectors = np.hstack(training_feature_vectors)
final_testing_feature_vectors = np.hstack(testing_feature_vectors)

svm = svm.SVC(kernel='linear', probability=True)
rfc = RandomForestClassifier(random_state=1, n_estimators=100)
lr = LogisticRegression(random_state=1)
eclf = VotingClassifier(estimators=[
    ('svm', svm), ('lr', lr), ('rfc', rfc)
], voting='soft', weights=[0.3, 0.3, 0.4])

pg = {'svm__C': [0.1, 0.2, 0.3], 'lr__C': [1.0, 100.0],\
      'rfc__n_estimators': [20, 100]}

grid = GridSearchCV(estimator=eclf, param_grid=pg, cv=5, n_jobs=4, verbose=5)
grid.fit(final_training_feature_vectors, train_y)
print grid.best_params_
print grid.best_score_
print "grid score: ", grid.score(final_testing_feature_vectors, test_y)
predictions = grid.predict(final_testing_feature_vectors)

f = open("predictions.csv", "wt")
writer = csv.writer(f)
writer.writerow(('PassengerId', "Survived"))
for i in xrange(len(predictions)):
    writer.writerow((test_passenger_ids[i], predictions[i]))
f.close()

#eclf.fit(final_training_feature_vectors, train_y)
#print eclf.score(final_testing_feature_vectors, test_y)

logging.info(" ----------- End  ------------")
total_time = time.time() - start_time
m, s = divmod(total_time, 60)
h, m = divmod(m, 60)
print "Program run time: %d:%02d:%02d" % (h, m, s)