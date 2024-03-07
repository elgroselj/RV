import cv2
from matplotlib import pyplot as plt
import numpy as np
import sys
sys.path.append(".")
from proj1.ex1_utils import gaussderiv, show_flow


def lucaskanade(im1,im2,N,SIG=0.5,TRESH=10**(-5)):
    #im1−first image matrix(grayscale)
    #im2−second image matrix(grayscale)
    #n−size of the neighborhood(NxN)
    
    Ix,Iy = gaussderiv(im1,SIG)
    It = im2-im1
    Ixy = np.multiply(Ix,Iy)
    Ixx = np.multiply(Ix,Ix)
    Iyy = np.multiply(Iy,Iy)
    Ixt = np.multiply(Ix,It)
    Iyt = np.multiply(Iy,It)
    
    kernel1 = np.ones((N,N), np.float32)
    suIxx = cv2.filter2D(src=Ixx, ddepth=-1, kernel=kernel1)
    suIxy = cv2.filter2D(src=Ixy, ddepth=-1, kernel=kernel1)
    suIyy = cv2.filter2D(src=Iyy, ddepth=-1, kernel=kernel1)
    suIxt = cv2.filter2D(src=Ixt, ddepth=-1, kernel=kernel1)
    suIyt = cv2.filter2D(src=Iyt, ddepth=-1, kernel=kernel1)
    

    D = np.multiply(suIxx,suIyy) - np.multiply(suIxy,suIxy)
    mask = np.abs(D) < TRESH
    D[mask & (D > 0)] = TRESH
    D[mask & (D == 0)] = TRESH
    D[mask & (D < 0)] = -TRESH
    
    
    U_denom = np.multiply(suIyy,suIxt)-np.multiply(suIxy,suIyt)
    U = -np.divide(U_denom,D)
    V_denom = np.multiply(suIxx,suIyt)-np.multiply(suIxy,suIxt)
    V = -np.divide(V_denom,D)
    
    
    
    return U,V
    

def hornschunck(im1,im2,n_iters,lmbd,SIG = 0.9):
    #im1−first imagematrix(grayscale)
    #im2−second imagematrix(grayscale)
    #n_iters−number of iterations(tryseveralhundred)
    #lmbd−parameter
    
    u = np.zeros(im1.shape)
    v = np.zeros(im1.shape)
    Ld = np.array([[0,1,0],[1,0,1],[0,1,0]]) * 0.25
    
    Ix,Iy = gaussderiv(im1,SIG)
    Ix2 = np.multiply(Ix,Ix)
    Iy2 = np.multiply(Iy,Iy)
    D = lmbd + Ix2 + Iy2
    It = im2-im1
    
    for _ in range(n_iters):
        ua = cv2.filter2D(src=u, ddepth=-1, kernel=Ld)
        va = cv2.filter2D(src=v, ddepth=-1, kernel=Ld)
        
        P = np.multiply(Ix,ua) + np.multiply(Iy,va) + It
        
        u = ua - np.multiply(Ix,np.divide(P,D))
        v = va - np.multiply(Iy,np.divide(P,D))
        
    return u,v
        
    

