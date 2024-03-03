import cv2
from matplotlib import pyplot as plt
import numpy as np
import sys
sys.path.append(".")
from proj1.ex1_utils import gaussderiv, show_flow


def lucaskanade(im1,im2,N,SIG=0.5):
    #im1−first image matrix(grayscale)
    #im2−second image matrix(grayscale)
    #n−size of the neighborhood(NxN)
    
    Ix,Iy = gaussderiv(im1,SIG)
    It = im2-im1
    Ixy = np.dot(Ix,Iy)
    Ixx = np.dot(Ix,Ix)
    Iyy = np.dot(Iy,Iy)
    Ixt = np.dot(Ix,It)
    Iyt = np.dot(Iy,It)
    
    kernel1 = np.ones((N,N), np.float32)
    suIxx = cv2.filter2D(src=Ixx, ddepth=-1, kernel=kernel1)
    suIxy = cv2.filter2D(src=Ixy, ddepth=-1, kernel=kernel1)
    suIyy = cv2.filter2D(src=Iyy, ddepth=-1, kernel=kernel1)
    suIxt = cv2.filter2D(src=Ixt, ddepth=-1, kernel=kernel1)
    suIyt = cv2.filter2D(src=Iyt, ddepth=-1, kernel=kernel1)
    

    D = np.dot(suIxx,suIyy) - np.dot(suIxy,suIxy)
    U_denom = np.dot(suIyy,suIxt)-np.dot(suIxy,suIyt)
    U = -np.divide(U_denom,D)
    V_denom = np.dot(suIxx,suIyt)-np.dot(suIxy,suIxt)
    V = -np.divide(V_denom,D)
    
    return U,V
    
    
    
    # u = 
    # plt.imshow(suIxx)
    # plt.show()
    
    
# im1 = np.ones((9,9), np.float32)/2
# im2 = np.ones((9,9), np.float32)/3

# im1 = np.eye(9,9)+np.eye(9,9,k=1)
# im2 = np.eye(9,9,k=1)+np.eye(9,9,k=2)
# print(im1,im2)


# U,V = lucaskanade(im1,im2,3)

# show_flow(U,V,plt.gca())
# plt.show()

    
