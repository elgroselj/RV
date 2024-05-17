import pandas as pd

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
    
visualize = False
verbose = False
treshold_stop = 3.5
treshold_start = 3
sigma = 5
N = 50
locality = 10
sequence = "dog"
set_sequence(sequence)


evaluate_tracker(dataset_path, network_path, results_dir, visualize, verbose, treshold_stop, treshold_start, sigma, N, locality)
pr, re, f = evaluate_performance(dataset_path, results_dir)
print(pr, re, f)
delete_sequence(sequence)


# for treshold_stop in [2.5, 3, 3.5]:
# for treshold_start in [2.5, 3, 3.5]:
for N in [10,30,50]:
# for locality, sigma in [(10,5),(2,5),(10,5)]:
    evaluate_tracker(dataset_path, network_path, results_dir, visualize, verbose, treshold_stop, treshold_start, sigma, N, locality)
    pr, re, f = evaluate_performance(dataset_path, results_dir)
    print(pr, re, f)
    delete_sequence(sequence)
    # evaluate_tracker(dataset_path, network_path, results_dir, visualize, verbose, treshold_stop, treshold_start, sigma, N, locality)
    # pr, re, f = evaluate_performance(dataset_path, results_dir)
    # print(pr, re, f)
    # delete_sequence(sequence)