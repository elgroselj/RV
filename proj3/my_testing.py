from run_corr_tracker_fun import run
from corr_tracker import CorrParams

parameters = CorrParams()
# for sigma in [0.2,0.5,1,1.5,2]:
#     parameters.sigma=sigma
# for lmbd in [0,0.1,0.5,1,2,10,20,200,500,1000]: # ne vpliva?
#     parameters.lmbd = lmbd
# for alpha in [0,0.01,0.03,0.05,0.06,0.07]:
#     parameters.alpha  =alpha
    # run(parameters=parameters, plot=False)
run(sequence="bicycle",plot=True,video_delay=40)