import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import pandas as pd


import sys
sys.path.append(".")
from ex2_utils import generate_responses_1, generate_responses_2, generate_responses_3, create_epanechnik_kernel, get_patch, extract_histogram, backproject_histogram
from ms_tracker import MeanShiftTracker
from run_ms_tracker import run_ms_tracker_fun



# tart = 50,40, velikost=21

# for pdf in [generate_responses_1()]:
#         for kernel_size in [9,15,21,33,55]:
#             for x0 in [[30,25]]:
#                 x1,_ = MeanShiftTracker.mean_shift_epanechnik(pdf,x0,100,kernel_size,kernel_size,plot=True,anneling_correction=False)
#                 err = np.linalg.norm((np.array(x1)-np.array([50,70])),2)
#                 print(kernel_size,x0,x1,err)

# for pdf in [generate_responses_1()]:
#         for kernel_size in [9,15,21,33,55]:
#             for x0 in [[30,25]]:
#                 pdf_ = cv.resize(pdf, (0,0), fx = 0.5, fy = 0.5)
#                 x0 = [x0[0]*0.5,x0[1]*0.5]
#                 print(pdf_.shape)
#                 x1,_ = MeanShiftTracker.mean_shift_epanechnik(pdf_,x0,100,kernel_size,kernel_size,plot=True,anneling_correction=False)
#                 x1 = [x1[0]*2,x1[1]*2]
#                 err = np.linalg.norm((np.array(x1)-np.array([70,50])),2)
#                 print(kernel_size,x0,x1,err)
                

# for pdf in [generate_responses_2()]:
#     for kernel_size in [9,15,21,33,55]:
#         for x0 in [[30,25]]:
#             x1 = MeanShiftTracker.mean_shift_epanechnik(pdf,x0,100,kernel_size,kernel_size,plot=True,anneling_correction=False)
#             err = np.linalg.norm((np.array(x1)-np.array([70,50])),2)
#             print(kernel_size,x0,x1,err)


# for pdf in [generate_responses_3()]:
#     for kernel_size in [9]:
#         for x0 in [[20,20]]:
#             x1 = MeanShiftTracker.mean_shift_epanechnik(pdf,x0,100,kernel_size,kernel_size,plot=True,anneling_correction=False,verbose=False)
#             err = np.linalg.norm((np.array(x1)-np.array([70,50])),2)
#             print(kernel_size,x0,x1,err)





# Compare different starting
# points, kernel sizes and termination criteria and report where do they converge.
# Write your observations of the convergence speed (in number of steps needed) in
# different setups. Can you speed up the convergence?

# pdf = generate_responses_1()

# for x0 in [(30,30),(80,40),(80,80),(40,80),(60,60)]:
#     for kernel_size in [11,21,31]:
#         x1, n_iter = MeanShiftTracker.mean_shift_epanechnik(pdf,x0,100,kernel_size,kernel_size,plot=True,anneling_correction=False)
#         err = np.linalg.norm((np.array(x1)-np.array([50,70])),2)
#         print(x0,kernel_size,x1,n_iter)

# for x0 in [(30,30),(80,40),(80,80),(40,80),(60,60)]:
#         for kernel_size in [11,21,31]:
#             kernel_size_ = int(kernel_size/2)
#             if kernel_size_ % 2 == 0:
#                 kernel_size_ +=1
#             pdf_ = cv.resize(pdf, (0,0), fx = 0.5, fy = 0.5)
#             x0_ = (x0[0]*0.5,x0[1]*0.5)
            
#             x1_, n_iter = MeanShiftTracker.mean_shift_epanechnik(pdf_,x0_,100,kernel_size_,kernel_size_,plot=True,anneling_correction=False)
#             x1 = (x1_[0]*2,x1_[1]*2)
#             err = np.linalg.norm((np.array(x1)-np.array([50,70])),2)
#             print(x0,kernel_size,x1,n_iter)



# pdf = generate_responses_2()

# for x0 in [(30,30),(40,80)]:
#     for kernel_size in [11,31]:
#         x1, n_iter = MeanShiftTracker.mean_shift_epanechnik(pdf,x0,100,kernel_size,kernel_size,plot=True,anneling_correction=False)
#         err = np.linalg.norm((np.array(x1)-np.array([50,70])),2)
#         print(x0,kernel_size,x1,n_iter)
        
        
# pdf = generate_responses_3()
# for x0 in [(20,20)]:
#     for kernel_size in [5,9]:
#         x1, n_iter = MeanShiftTracker.mean_shift_epanechnik(pdf,x0,100,kernel_size,kernel_size,plot=True,anneling_correction=False)
#         err = np.linalg.norm((np.array(x1)-np.array([50,70])),2)
#         print(x0,kernel_size,x1,n_iter)


# df = None
# d = {}
# for sequence in ['basketball', 'bicycle', 'fish1', 'sphere', 'surfing']:
# # for sequence in ['bicycle']:
#     # for enlarge_factor in [1,2,3,5]:
#     # for epsi in [10e-3, 10e-7, 10e-10]:
#     # for mean_shift_kernel_shape in [(5,5),(11,11),(21,21),(31,31),(51,51)]:
#     # for mean_shift_n_iter in [5,10,20,100]:
#     # for alpha in [0,0.1,0.3,0.5,0.8,1]:
#     # for nbins in [8,16,32]:
#     for hist_mode in ["BGR","HSV","HS","YCbCr","Lab"]:
#         d = {}
#         d["sequence"] = sequence
#         # d["enlarge_factor"] = enlarge_factor
#         # d["epsi"] = epsi
#         # d["mean_shift_kernel_shape"] = mean_shift_kernel_shape
#         # d["mean_shift_n_iter"] = mean_shift_n_iter
#         # d["alpha"] = alpha
#         # d["nbins"] = nbins
#         d["hist_mode"] = hist_mode
#         if sequence == "fish1":
#             print(d)
#             d["n_failures"] = run_ms_tracker_fun(sequence,d,plot=False)
#         else:
#             try:
#                 d["n_failures"] = run_ms_tracker_fun(sequence,d)
#             except:
#                 d["n_failures"] = 2
        
#         if df is None:
#             df = pd.DataFrame(columns=d.keys())
        
#         # df2 = {'Name': 'Amy', 'Maths': 89, 'Science': 93} 
#         df.loc[len(df.index)] = d.values()
#         # df = df.append(d, ignore_index = True) 
# # print(df)
# df.to_csv("params.csv")

df=pd.read_csv("params.csv")
print(df)

param = "hist_mode"

for sequence in df["sequence"].unique():
    df_ = df[df["sequence"] == sequence]
    plt.plot(df_[param],df_["n_failures"],'--o',label=sequence)
    plt.xlabel(param)
    plt.ylabel("n_failures")
    # plt.xscale("log")
    # plt.legend()
neki = df[["hist_mode","n_failures"]].groupby("hist_mode").mean().reset_index()
plt.scatter(neki["hist_mode"],neki["n_failures"],c="black")
plt.savefig(param+".png")
plt.show()




# for sequence in ['basketball', 'bicycle', 'fish1','sphere', 'surfing']:
    
# sequence = "basketball"
# d={"alpha":0.1}
# run_ms_tracker_fun(sequence,d,plot=True,video_delay=15,show_gt=True)

# sequence = "bicycle"
# run_ms_tracker_fun(sequence,{},plot=True)

# sequence = "fish1"
# d={"alpha":0,
# "nbins":32}
# run_ms_tracker_fun(sequence,d,plot=True,video_delay=30,show_gt=True)

# sequence = "fish1"
# d={"alpha":0.1,
# "nbins":32}
# run_ms_tracker_fun(sequence,d,plot=True,video_delay=30,show_gt=True)

# sequence = "sphere"
# d={
#     # "enlarge_factor":3,
#    "alpha":0.3,
#     "nbins":8, #32
#     "mean_shift_kernel_shape":(31,31),
#     # "mean_shift_n_iter": 10
# }
# run_ms_tracker_fun(sequence,d,plot=True)

# sequence = "surfing"
# d={
#     # "enlarge_factor":3,
#    "alpha":0,
#     "nbins":8, #32
#     "mean_shift_kernel_shape":(11,11),
#     "mean_shift_n_iter": 4
# }
# run_ms_tracker_fun(sequence,d,plot=True)






# sequence = "fish1"
# d={"alpha":0,
# "nbins":32,
# "hist_mode":"HSV"}
# run_ms_tracker_fun(sequence,d,plot=True,video_delay=30,show_gt=True)




