from run_corr_tracker_fun import run
from corr_tracker import CorrParams
import matplotlib.pyplot as plt



    
def run_tests(params=CorrParams(),verbose=False):
    
    tests = ["bicycle",  "car",  "david",   "face",        "hand",       "juice", "sunshade",  "woman",
    "bolt",     "cup",  "diving",  "gymnastics", "iceskater",  "jump",   "singer",    "torus"]

    sum_fails = 0
    sum_overlap = 0
    init_speeds = []
    track_speeds = []
    for test in tests:
        fps, n_fail, overlap, init_times, track_times = run(sequence=test,plot=False,video_delay=40,parameters=params,verbose=verbose)
        init_speed = round(len(init_times)/sum(init_times)) # 1/avg(time)        
        track_speed = round(len(track_times)/sum(track_times))
        init_speeds.append(init_speed)
        track_speeds.append(track_speed)
        
        sum_fails+= n_fail
        sum_overlap+= overlap
    
    if False:
        st_pikslov = [20*50, 57*36, 80*100, 85*99, 50*60, 25*60, 37*52, 30*100, 25*60, 45*60, 50*30, 50*110, 90*150, 60*57, 70*270, 47*50]

        scatter3(tests,init_speeds,track_speeds,st_pikslov)
        
    print("n_fails: ", sum_fails)
    print("avg_overlap: ", sum_overlap/len(tests))
    return sum_fails, sum_overlap/len(tests)

def plot2(x, y1, y2, x_label, y1_label="n_failures", y2_label="avg_overlap"):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel()
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
    plt.show()
    
def scatter3(x, y1, y2, z):
    fig, ax1 = plt.subplots()

    color1 = 'tab:red'
    color2 = 'tab:blue'
    # ax1.set_xticklabels(ax1.get_xticks(), rotation = 50)
    for tick in ax1.get_xticklabels():
        tick.set_rotation(50)
    
    ax1.set_ylabel("FPS", color=color1)
    ax1.scatter(x, y1, color=color1, marker="x")
    ax1.scatter(x, y2, color=color1, marker="o")
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
    
    ax2.set_ylabel("n_pixels", color=color2)
    ax2.scatter(x, z, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(ax2.get_ylim()[::-1])

    plt.savefig("initvtrack.png")
    plt.show()
    




# parameters = CorrParams()
# sigmas = [1,2,3,4]
# lmbds = [0,0.1,1,10,30,70,100,500,1000]
# alphas = [0,0.01,0.03,0.05,0.06,0.07,0.1]
# Fs = []
# Os = []
# for sigma in sigmas:
#     parameters.sigma=sigma
# # for lmbd in lmbds:
# #     parameters.lmbd=lmbd 
# # for alpha in alphas:
# #     parameters.alpha = alpha
    
#     F, O = run_tests(parameters)
#     Fs.append(F)
#     Os.append(O)
# print(lmbds,Fs,Os)
    
# plot2(sigmas,Fs,Os, x_label="sigma")
# # plot2(lmbds,Fs,Os, x_label="lambda")
# # plot2(alphas,Fs,Os, x_label="alpha")
params = CorrParams()
params.enlarge_factor = 1.1
run_tests(verbose=True, params=params)


        

