import pandas as pd
import matplotlib.pyplot as plt

from evaluate_trackerLT import evaluate_tracker
from performance_evaluation_help import evaluate_performance

dataset_path = "/home/lema/Documents/RV/proj5/dataset"
network_path = "/home/lema/Documents/RV/proj5/material/siamfc_net.pth"
results_dir ="/home/lema/Documents/RV/proj5/resultsLT"

import os
from random import randint


def set_sequence(sequence):
    with open("/home/lema/Documents/RV/proj5/dataset/list.txt","w") as f:
        f.write(sequence)
        

def delete_sequence(sequence):
    old = os.path.join("/home/lema/Documents/RV/proj5/resultsLT", sequence+"_bboxes.txt")
    os.remove(old)
    
    old = os.path.join("/home/lema/Documents/RV/proj5/resultsLT", sequence+"_scores.txt")
    os.remove(old)
    
# visualize = False
# verbose = False
# treshold_stop = 3
# treshold_start = 3
# sigma = 5
# N = 50
# locality = 10
# sequence = "car9"
# set_sequence(sequence)

visualize = False
verbose = False
treshold_stop = 5
treshold_start = 3
sigma = 10
N = 30
locality = 100
sequence = "car9"
set_sequence(sequence)


# evaluate_tracker(dataset_path, network_path, results_dir, visualize, verbose, treshold_stop, treshold_start, sigma, N, locality)
# pr, re, f = evaluate_performance(dataset_path, results_dir)
# delete_sequence(sequence)


#############################3


prs = []
res = []
fs =  []

vals_multi = []

# ################################33


# name = "treshold_stop"
# # vals = [2.5, 3, 3.5]
# vals = [4,5,6]
# for treshold_stop in vals:
    
name = "treshold_start"
# vals = [2.5, 3, 3.5]
vals = [2,3,4,5]
for treshold_start in vals:
    
# Device: cuda:0
# Processing sequence: car9
# [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# fails: min: 0, max: 1, mean: 0.9444444444444444, sum: 17
# --------------------------
# Precision: 0.5329651664138192
# Recall: 0.5122592286021062
# F-score: 0.5224071047725142
# --------------------------
# Device: cuda:0
# Processing sequence: car9
# [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# fails: min: 0, max: 1, mean: 0.9375, sum: 15
# --------------------------
# Precision: 0.4010748724806244
# Recall: 0.324018976277588
# F-score: 0.35845255014753485
# --------------------------
    
# name = "N"
# vals = [10,30,50]
# for N in vals:

# name = "locality"
# vals = [0,2,10]
# for locality in vals:
    
# name = "sigma"
# vals = [2,5,10]
# for sigma in vals:
    
# ####################################3 
    
    for i in range(2):
        evaluate_tracker(dataset_path, network_path, results_dir, visualize, verbose, treshold_stop, treshold_start, sigma, N, locality)
        pr, re, f = evaluate_performance(dataset_path, results_dir)
        prs.append(pr)
        res.append(re)
        fs.append(f)
        delete_sequence(sequence)
        
        #####################
        vals_multi.append(treshold_start)
        ####################
        
# # Plot

# prs = [3,3.5,4,4.5]
# vals_multi = [2,2,5,5]

plt.figure(figsize=(10, 6))


plt.scatter(vals_multi, prs, marker='o', label='Precision')
plt.scatter(vals_multi, res, marker='o', label='Recall')
plt.scatter(vals_multi, fs, marker='o', label='F-score')

plt.title('Performance Metrics vs '+name)
plt.xlabel(name)
plt.ylabel('Score')
plt.xticks(vals)
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(sequence+"_"+name+".png")
plt.show()
        
