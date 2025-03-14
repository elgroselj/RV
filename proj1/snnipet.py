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
U_hs , V_hs = hornschunck( im1 , im2 , 1000 , 0.5 )



fig1, ((ax1_11, ax1_12), (ax1_21, ax1_22), (ax1_31, ax1_32)) = plt.subplots(3, 2)
for tup in ((ax1_11, ax1_12), (ax1_21, ax1_22), (ax1_31, ax1_32)):
    for ax in tup:
        ax.set_axis_off()
# fig1.tight_layout()
fig1.subplots_adjust(wspace=0, hspace=0)
ax1_11.imshow(im1)
ax1_12.imshow(im2)
show_flow(U_lk, V_lk, ax1_21, type='angle')
show_flow(U_lk, V_lk, ax1_22, type='field', set_aspect=True)
show_flow(U_hs , V_hs , ax1_31 , type='angle')
show_flow(U_hs, V_hs, ax1_32, type='field',set_aspect=True)

# fig1.suptitle('Lucas−Kanade and Horn−Schunck Optical Flow')

plt.savefig(im_name+".svg",bbox_inches="tight",pad_inches=0)