import numpy as np
import matplotlib . pyplot as plt
from ex1_utils import rotate_image, show_flow
import sys
sys.path.append(".")
from my import lucaskanade, hornschunck

#examples1

# im_name, im1= ("rnd",np.random.rand(200 ,200).astype( np.float32 ))
#### im_name, im1 = ("avtocek", np.array(plt.imread("/home/lema/Documents/RV/proj1/collision/00000001.jpg")))
# im_name, im1 = ("grad", np.array(plt.imread("/home/lema/Documents/RV/proj1/disparity/cporta_left.png")))

# sig = 0.9
# tresh = 10**(-5)
    
# im2 = rotate_image(im1.copy() , -1)


#examples2

# im_name="cporta"
# im1=np.array(plt.imread("/home/lema/Documents/RV/proj1/disparity/cporta_left.png"))
# im2=np.array(plt.imread("/home/lema/Documents/RV/proj1/disparity/cporta_right.png"))

# sig = 0.9
# tresh = 10**(-5)

#examples3

# im_name="office"
# im1=np.array(plt.imread("/home/lema/Documents/RV/proj1/disparity/office_left.png"))
# im2=np.array(plt.imread("/home/lema/Documents/RV/proj1/disparity/office_right.png"))

# sig = 0.9
# tresh = 10**(-5)

#examples4

im_name="office2"
im1=np.array(plt.imread("/home/lema/Documents/RV/proj1/disparity/office2_left.png"))
im2=np.array(plt.imread("/home/lema/Documents/RV/proj1/disparity/office2_right.png"))

sig = 0.9
tresh = 10**(-5)



U_lk, V_lk = lucaskanade(im1, im2, 3, SIG=sig, TRESH=tresh)


fig1, ((ax1_11, ax1_12), (ax1_21, ax1_22)) = plt.subplots(2, 2)
ax1_11.imshow(im1)
ax1_12.imshow(im2)
show_flow(U_lk, V_lk, ax1_21, type='angle')
show_flow(U_lk, V_lk, ax1_22, type='field', set_aspect=True)
fig1.suptitle('Lucas−Kanade Optical Flow')

plt.savefig(im_name+"_lk.svg")


# U_hs , V_hs = hornschunck( im1 , im2 , 1000 , 0.5 )

# fig2, ((ax2_11, ax2_12), (ax2_21, ax2_22)) = plt.subplots(2, 2)

# ax2_11.imshow( im1 )
# ax2_12.imshow( im2 )
# show_flow(U_hs , V_hs , ax2_21 , type='angle')
# show_flow(U_hs, V_hs, ax2_22, type='field',set_aspect=True)
# fig2.suptitle('Horn−Schunck OpticalFlow')
# plt.savefig(im_name+"_hs.svg")