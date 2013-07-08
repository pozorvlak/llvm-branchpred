#!/usr/bin/env python
from sklearn import cross_validation
import numpy as np
import sys as sys
from analysis import read_files, classifiers

def accuracy(actual, predictions, weights):
    return float(sum((actual == predictions) * weights)/sum(weights))

def main():
    #read in  data, parse into training and target sets
    dataset = read_files(sys.argv[1:])
    target = dataset[:, 11]
    train = dataset[:, 0:9]
    weights = dataset[:, 9]

    cs = classifiers()
    max_name_length = max([len(name) for name in cs.keys()])
    print "Percentage accuracy: higher numbers are better"
    for name in cs.keys():
        cfr = cs[name]

        #Simple K-Fold cross validation. 5 folds.
        cv = cross_validation.KFold(len(train), n_folds=5, indices=False)

        #iterate through the training and test cross validation segments and
        #run the classifier on each one, aggregating the results into a list
        results = []
        for traincv, testcv in cv:
            trained = cfr.fit(train[traincv], target[traincv])
            predictions = trained.predict(train[testcv])
            # print predictions
            score = accuracy(target[testcv], predictions, weights[testcv])
            results.append(score)

        #print out the mean of the cross-validated results
        print "{}  {}".format(name.ljust(max_name_length),
                np.array(results).mean())

if __name__=="__main__":
    main()
