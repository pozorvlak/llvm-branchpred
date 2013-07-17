#!/usr/bin/env python

from analysis import *
from pylab import *
from matplotlib.mlab import PCA
from mpl_toolkits.mplot3d import Axes3D
import os

def read_all_files():
    return read_files(["training.csv"])

def hist_by_attribute(attr):
    fig = figure()
    ax = fig.add_subplot(111)
    taken = data[data[:, prediction] == 1, attr]
    not_taken = data[data[:, prediction] == 0, attr]
    ax.hist([taken, not_taken], label=['taken', 'not taken'],
            normed=False, bins = 2)
    ax.legend()
    ax.set_xticks([0, 1])

def show_prob_dist(num_bars):
    fig = figure()
    ax = fig.add_subplot(111)
    ax.hist(data[:, prob], bins=num_bars)
    xlabel("Probability branch will be taken")
    ylabel("Count of branches with probabilities in this range")

def show_prob_fn(num_bars, cost_fn):
    fig = figure()
    ax = fig.add_subplot(111)
    impact = zeros(num_bars)
    bar_width = 1.0/num_bars
    for i in xrange(0, num_bars):
        lower = i * bar_width
        upper = lower + bar_width
        indices = logical_and(data[:, prob] >= lower, data[:, prob] < upper)
        impact[i] = cost_fn(indices)
    ax.bar(arange(num_bars), impact)
    ax.set_xticks(arange(0, num_bars+1, num_bars/10))
    ax.set_xticklabels(arange(0, 1.1, 0.1))
    xlabel("Probability branch will be taken")

def impact(indices):
    return sum(data[indices, total])

def show_prob_impact(num_bars):
    show_prob_fn(num_bars, impact)
    ylabel("Number of branch events with probabilities in this range")

def excess_mispredictions(indices):
    excess_misprediction_frequency = abs(2 * data[indices, prob] - 1)
    return sum(data[indices, total] * excess_misprediction_frequency)

def show_prob_cost(num_bars):
    show_prob_fn(num_bars, excess_mispredictions)
    ylabel("Number of extra mispredictions if you mispredict these branches")

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

def jitter(data):
    """Scaled random jitter for 2D arrays"""
    jitter = zeros(data.shape)
    [num_rows, num_cols] = data.shape
    for i in xrange(0, num_cols):
        jitter[:, i] = (rand(num_rows) - 0.5) / (var(data[:, i]) * 2)
    return jitter

def pca(dim):
    pca = PCA(data[:, 0:9])
    return pca.project(data[:, 0:9])[:, 0:dim]

def scatter2d(data, color):
    scatter(data[:, 0], data[:, 1], color=color, marker="x")

def scatter2d_with_jitter():
    proj2 = pca(2)
    jittered = proj2 + jitter(proj2)
    strong_taken = jittered[data[:, prob] > 0.9, :]
    strong_not_taken = jittered[data[:, prob] < 0.1, :]
    weak = jittered[logical_and(data[:, prob] >= 0.1, data[:, prob] <= 0.9), :]
    scatter2d(strong_taken, "red")
    scatter2d(strong_not_taken, "blue")
    scatter2d(weak, "green")

def scatter3d(axis, data, color):
    axis.scatter(data[:, 0], data[:, 1], data[:, 2], color=color, marker="x")

def scatter3d_with_jitter():
    proj3 = pca(3)
    fig = figure()
    ax = Axes3D(fig)
    jittered = proj3 + jitter(proj3)
    taken = jittered[data[:, prediction] == 1, :]
    not_taken = jittered[data[:, prediction] == 0, :]
    scatter3d(ax, taken, "red")
    scatter3d(ax, not_taken, "blue")

data = read_all_files()

[call, guard, loop_branch, loop_exit, loop_header, opcode, pointer, ret,
        store, total, prob, prediction] = xrange(0, 12)
