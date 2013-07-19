#!/usr/bin/env python
from sklearn import cross_validation
import numpy as np
import sys as sys
from analysis import read_files, classifiers

[call, guard, loop_branch, loop_exit, loop_header, opcode, pointer, ret,
        store, total, prob, prediction] = xrange(0, 12)

def accuracy(actual, predictions, weights):
    return float(sum((actual == predictions) * weights)/sum(weights))

def xval_score(cfr, target, train, weights):
    """Cross-validate a single classifier"""
    #Simple K-Fold cross validation. 5 folds.
    cv = cross_validation.KFold(len(train), n_folds=5, indices=False)
    #iterate through the training and test cross validation segments and
    #run the classifier on each one, aggregating the results into a list
    results = []
    for traincv, testcv in cv:
        trained = cfr.fit(train[traincv], target[traincv])
        predictions = trained.predict(train[testcv])
        score = accuracy(target[testcv], predictions, weights[testcv])
        results.append(score)
    return np.array(results).mean()

def xval(target, train, weights):
    """Cross-validate all classifiers, and print out their scores"""
    cs = classifiers()
    max_name_length = max([len(name) for name in cs.keys()])
    print "Percentage accuracy: higher numbers are better"
    best_score = 0
    best_name = "Always wrong"
    for name in cs.keys():
        score = xval_score(cs[name], target, train, weights)
        #print out the mean of the cross-validated results
        print "{}  {}".format(name.ljust(max_name_length), score)
        if score > best_score:
            best_score = score
            best_name = name
    print "Best: {} {}".format(best_name, best_score)

def excess_mispredictions(dataset):
    excess_misprediction_frequency = abs(2 * dataset[:, prob] - 1)
    return dataset[:, total] * excess_misprediction_frequency

def main():
    #read in  data, parse into training and target sets
    dataset = read_files(sys.argv[1:])
    target = dataset[:, prediction]
    train = dataset[:, call:total]
    print "\nResults weighted equally"
    xval(target, train, np.ones(dataset.shape[0]))
    print "\nResults weighted by branch frequency"
    xval(target, train, dataset[:, total])
    print "\nResults weighted by number of mispredictions"
    xval(target, train, excess_mispredictions(dataset))

if __name__=="__main__":
    main()
