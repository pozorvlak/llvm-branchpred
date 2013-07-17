#!/usr/bin/env python
import csv as csv
import sys as sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from wularusclassifier import WuLarusClassifier

def read_raw_file(filename):
    # Load in the training csv file
    csv_file_object = csv.reader(open(filename, 'rb'))
    header = csv_file_object.next() # Skip the first line as it is a header
    data = []
    for row in csv_file_object:
        data.append(row)
    return data

def munge(data):
    [call, guard, loop_branch, loop_exit, loop_header, opcode, pointer, ret,
        store, file_total, name, not_taken, taken] = xrange(0, 13)
    data = np.array(data)
    data = np.array(np.delete(data, [name], 1), np.float)
    not_taken -=1
    taken -= 1
    total = data[:, taken] + data[:, not_taken]
    zeroes = np.where(total == 0)
    data = np.delete(data, zeroes, 0)
    total = np.delete(total, zeroes, 0)
    probs = data[:, taken] / total
    total = total / data[:, file_total]
    # XXX we actually want to minimise E(lost cycles)
    # The best threshold may not be at 50%
    predict_taken = data[:, taken] > data[:, not_taken]
    data = np.hstack([data, np.column_stack((total, probs, predict_taken))])
    data = np.delete(data, [not_taken, taken, file_total], 1)
    return np.array(data, np.float)

def read_files(filenames):
    data = []
    for filename in filenames:
        data += read_raw_file(filename)
    return munge(data)

def write_munged():
    data = read_files(sys.argv[1:])
    open_file_object = csv.writer(open("munged.csv", "wb"))
    header = ("call guard loop_branch loop_exit loop_header opcode " +
        "pointer ret store total prob prediction").split()
    open_file_object.writerow(header)
    for row in data:
        open_file_object.writerow(row)

class PredictTrueClassifier:
    def fit(self, features, target):
        return self

    def predict(self, features):
        return np.ones(features.shape[0])

def classifiers():
    return {
        "Logistic regression" : LogisticRegression(),
        "3-nearest neighbours" : KNeighborsClassifier(3, p=1),
        "Ada boost" : AdaBoostClassifier(),
        "Gradient boost" :
            GradientBoostingClassifier(n_estimators=50, learning_rate=1.0),
        "Random forest" :
            RandomForestClassifier(n_estimators=10, n_jobs=-1), # no. cores
        "Always predict true" : PredictTrueClassifier(),
        "Decision tree" : DecisionTreeClassifier(),
        "Bernoulli Naive Bayes" : BernoulliNB(),
        "Wu-Larus" : WuLarusClassifier() }

if __name__=="__main__":
    write_munged()
