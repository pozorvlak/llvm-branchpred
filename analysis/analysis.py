import csv as csv
import sys as sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

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
        store, name, not_taken, taken] = xrange(0, 12)
    data = np.array(data)
    data = np.array(np.delete(data, [name], 1), np.float)
    not_taken -=1
    taken -= 1
    total = (data[:, taken] + data[:, not_taken] + 1)
    probs = data[:, taken] / total
    data[:, not_taken] = probs # now contains probabilities
    data = np.delete(data, [taken], 1)
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
        "pointer ret store prob").split()
    open_file_object.writerow(header)
    for row in data:
        open_file_object.writerow(row)

def classifiers():
    return { "Logistic regression" : LogisticRegression(),
             "3-nearest neighbours" : KNeighborsClassifier(3, p=1),
             "Ada boost" : AdaBoostClassifier(),
             "Gradient boost" :
                 GradientBoostingClassifier(n_estimators=50, learning_rate=1.0)}

if __name__=="__main__":
    write_munged()
