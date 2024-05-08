import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from run_part_tracker_fun import run
from part_tracker import PartParams

def plot2(x, y1, y2, x_label, y1_label="n_failures", y2_label="avg_overlap"):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel(x_label)
    # ax1.set_xscale("log")
    
    ax1.set_ylabel(y1_label, color=color)
    ax1.plot(x, y1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel(y2_label, color=color)
    ax2.plot(x, y2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(ax2.get_ylim()[::-1])

    plt.savefig(x_label+".png")
    # plt.show()

def run_tests(params=PartParams(),verbose=False):
    
    # tests = ["bicycle",  "car",  "david",   "face",        "hand",       "juice", "sunshade",  "woman",
    # "bolt",     "cup",  "diving",  "gymnastics", "iceskater",  "jump",   "singer",    "torus"]
    
    tests = ["car",   "face",     "juice",  "woman",
       "cup",   "gymnastics",   "jump",      "torus"]
    
    n_fails = 0
    overlaps = 0
    for test in tests:
        print(test,"...")
        FPS, n_fail, overlap = run(sequence=test,parameters=PartParams(),plot=False,video_delay=15,verbose=verbose)
        n_fails += n_fail
        overlaps += overlaps
    return n_fails, overlaps/len(tests)

def d2obj(d):
    parameters = PartParams()
    if "enlarge_factor" in d.keys():
        parameters.enlarge_factor = d["enlarge_factor"]
    if "nbins" in d.keys():
        parameters.nbins = d["nbins"]
    if "hist_mode" in d.keys():
        parameters.hist_mode = d["hist_mode"]
    if "alpha" in d.keys():
        parameters.alpha = d["alpha"]
    if "q" in d.keys():
        parameters.q = d["q"]
    if "r" in d.keys():
        parameters.r = d["r"]
    if "n" in d.keys():
        parameters.n = d["n"]
    if "sigma" in d.keys():
        parameters.sigma = d["sigma"]
    if "sigma2" in d.keys():
        parameters.sigma2 = d["sigma2"]
    return parameters
    


# parameters = PartParams()
# n_fails = []
# avg_overlaps = []
# ###############################vvv
# x_label = "enlarge_factor"
# values = [1,1.5,2,2.5]
# for val in values:
#     parameters.enlarge_factor = val
#     ##############################^^^
#     print(x_label,val)
#     n_fail, avg_overlap = run_tests(parameters)
#     n_fails.append(n_fail)
#     avg_overlaps.append(avg_overlap)
# plot2(values,n_fails, avg_overlaps, x_label)

# parameters = PartParams()

###############################vvv



def neki(x_label, values):
    n_fails = []
    avg_overlaps = []
    for val in values:
        parameters = d2obj({x_label:val})
        print(x_label,val)
        n_fail, avg_overlap = run_tests(parameters)
        n_fails.append(n_fail)
        avg_overlaps.append(avg_overlap)
    plot2(values,n_fails, avg_overlaps, x_label)
    

for x_label, values in [
                        ("enlarge_factor",[2,2.5,3])
                        # ("alpha",[0,0.01,0.1]),
                        ("alpha",[0.1,0.15,0.2]),
                        ("q",[0.1,0.25,0.5,1,2]),
                        ("r",[0.1,0.5,1,2]),
                        ("n",[50,100,150]),
                        ("sigma2", [0.8,1,1.5])
                        ]:
    try:
        neki(x_label,values)
        plt.clf()
    except Exception:
        print("exception at x_label, you need to retry")


        
# self.enlarge_factor = 2
        
# self.nbins = 16
# self.hist_mode = "BGR"
# self.alpha = 0.1 # spreminjanje tergeta

# self.q = 1 # dependent on target size # kolko naprej skočijo pikice v dinamičnem modelu
# self.r = 1 # koliko je šuma v podatkih

# self.n = 100 # število partiklov
# self.sigma = 1 # koliko se razmečejo okoli ground truth (samo pri init)
# self.sigma2 = 1 # kolko se kaznuje različnost od targeta (v obliki zmanjšanja uteži)

        