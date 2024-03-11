import cv2
from matplotlib import pyplot as plt
import numpy as np
import sys
sys.path.append(".")
from proj1.ex1_utils import gaussderiv, show_flow, gausssmooth


def lucaskanade(im1,im2,N,SIG=0.9,TRESH=10**(-5),R=None,R_tresh=0.6):
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
    
    assert(N%2==1)    
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
    
    if R is not None:
        U[R<R_tresh]=0
        V[R<R_tresh]=0
    
    # if R is not None:
    #     U[R==0]=0
    #     V[R==0]=0
    
    
    
    return U,V
    

def hornschunck(im1,im2,n_iters,lmbd,SIG = 0.9,u0=None,v0=None):
    #im1−first imagematrix(grayscale)
    #im2−second imagematrix(grayscale)
    #n_iters−number of iterations(tryseveralhundred)
    #lmbd−parameter
    
    u = np.zeros(im1.shape) if u0 is None else u0
    v = np.zeros(im1.shape) if v0 is None else v0
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
    
    
def harris_response(im,SIG=1,kernel_size=6,alpha=0.06,tresh=0.95):
    Ix,Iy = gaussderiv(im,sigma=SIG)
    a = gausssmooth(Ix*Ix,sigma=SIG)
    b = gausssmooth(Iy*Iy,sigma=SIG)

    det = np.multiply(a,b)
    trace = a+b

    R = det - alpha * np.multiply(trace,trace)
    R = R-np.amin(R)
    R = R/np.amax(R)
          

    def points_nonmax_tresh(hp,tresh=tresh,box_size=2):
        def nonmaxima_suppression_box(A,box_size=2):
            
            def loc_max_box(A,y,x,box_size=1):#9,25
                if box_size == 1:
                    return (A[y,x] >= max(A[y+1,x-1],A[y+1,x+1],A[y+1,x],A[y,x+1]) and A[y,x] > max(A[y,x-1],A[y-1,x-1],A[y-1,x],A[y-1,x+1]))
                elif box_size == 2:
                    return (A[y,x] > max(A[y-2,x-2], A[y-2,x-1], A[y-2,x], A[y-2,x+1], A[y-2,x+2],
                                        A[y-1,x-2], A[y-1,x-1], A[y-1,x], A[y-1,x+1], A[y-1,x+2],
                                        A[y,x-2],   A[y,x-1]) and
                            A[y,x] >= max(A[y,x+1], A[y,x+2],
                                        A[y+1,x-2], A[y+1,x-1], A[y+1,x], A[y+1,x+1], A[y+1,x+2],
                                        A[y+2,x-2], A[y+2,x-1], A[y+2,x], A[y+2,x+1], A[y+2,x+2]))
                    
            A_ = np.copy(A)
            for x in range(box_size,len(A[0])-box_size):
                for y in range(box_size,len(A)-box_size):
                    if not loc_max_box(A_,y,x,box_size):
                        A_[y,x] = 0
            return(A_)
        
        hp_ = nonmaxima_suppression_box(hp,box_size)
        hp_ = np.where(hp_>tresh,1,0)
        return(hp_)
    
    
    # R_ = points_nonmax_tresh(R,tresh=tresh)
    # return(R_)
    return R
    
    
        
    

