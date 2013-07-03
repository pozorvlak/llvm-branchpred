from sklearn import cross_validation
import numpy as np
import sys as sys
from analysis import read_files, classifiers

def accuracy(actual, predictions):
    return float(sum(actual == predictions))/len(actual)

def main():
    #read in  data, parse into training and target sets
    dataset = read_files(sys.argv[1:])
    target = dataset[0::,10]
    train = dataset[0::,0::8]

    cs = classifiers()
    for name in cs.keys():
        print "Performing 5-fold cross validation with " + name
        cfr = cs[name]

        #Simple K-Fold cross validation. 5 folds.
        cv = cross_validation.KFold(len(train), n_folds=5, indices=False)

        #iterate through the training and test cross validation segments and
        #run the classifier on each one, aggregating the results into a list
        results = []
        for traincv, testcv in cv:
            trained = cfr.fit(train[traincv], target[traincv])
            predictions = trained.predict(train[testcv])
            results.append( accuracy(target[testcv], predictions) )

        #print out the mean of the cross-validated results
        print "Results: " + str( np.array(results).mean() )

if __name__=="__main__":
    main()
