#!/usr/bin/env python

from analysis import *
from pylab import *
import os

data = read_all_files()

[call, guard, loop_branch, loop_exit, loop_header, opcode, pointer, ret,
        store, total, prob, prediction] = xrange(0, 12)

def read_all_files():
    csvs = ["../csv/" + i for i in os.listdir("../csv/")]
    return read_files(csvs)

def hist_by_attribute(attr):
    taken = data[data[:, prediction] == 1, attr]
    not_taken = data[data[:, prediction] == 0, attr]
    cla()
    hist([taken, not_taken], label=['taken', 'not taken'],
            normed=False, bins = 2)
    legend()
    xticks([0, 1])

def show_prob_dist():
    hist(data[:, prob])

def print_percentage(name, count, total_count):
    print "{}: {}, ({}%)".format(name, count, 100*(count + 0.0)/total_count)

def tail_size():
    pred_low = data[:, prob] < 0.1
    pred_high = data[:, prob] > 0.7
    low_count = data[pred_low, :].shape[0]
    high_count = data[pred_high, :].shape[0]
    tail_count = low_count + high_count
    total_count = data.shape[0]
    print_percentage("Low tail", low_count, total_count)
    print_percentage("High tail", high_count, total_count)
    print_percentage("Both tails", tail_count, total_count)
    print_percentage("Neither tail", total_count - tail_count, total_count)
