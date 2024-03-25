import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


import sys
sys.path.append(".")
from ms_material.ex2_utils import generate_responses_1, create_epanechnik_kernel, get_patch



# plt.imshow(generate_responses_1())
# plt.show()

# im = generate_responses_1()

# w = 100
# h = 100
# sigma = 1
# kernel = create_epanechnik_kernel(w, h, sigma)

# plt.imshow(kernel)
# plt.show()

# k = lambda y: 1-y
# g = lambda y: 1

def mean_shift_epanechnik(pdf, x0, n_iter, h, hs= None, plot=False):
    # pdf           pdf matrix
    # x0 = (a,b)    start position
    # n_iter        number of iterations
    # h region size
    print("x0: ", x0)
    x = x0
    xs = [tuple(x)]
    if hs is None:
        hs = [h]
        
    if plot:
        plt.imshow(pdf)
    
    COLORS = "rgb"
    for hi, h in enumerate(hs):
        assert h % 2 == 1
        pol = int(h/2)
        gradnik = np.arange(-pol,pol+1).reshape(h,1)
        indexy = np.repeat(gradnik, h, axis=1)
        indexx = indexy.transpose()
        for i in range(n_iter):
            patch, _ = get_patch(pdf, x, (h,h))
            # print(patch)
            # plt.imshow(patch)
            # plt.show()
            xx = np.sum(np.multiply(patch, indexx))
            yy = np.sum(np.multiply(patch, indexy))
            norm = np.sum(patch)
            # print(xx,yy,norm)
            x_ = np.array([xx, yy]) / norm
            # print(x_)
            x_ = [int(round(x_[0])),int(round(x_[1]))]
            # print(x_)
            x[0] += x_[0]
            x[1] += x_[1]
            xs.append(tuple(x))
            if x_ == [0,0]:
                print("Break: close enough n_iter: ", i)
                break
            print(x)
        
        if plot:
            a = np.array(xs)
            plt.scatter(a[:,0],a[:,1],s=5,c=COLORS[hi%len(COLORS)],marker="o")
            xs = []
    plt.show()      
    return x
        


# im = generate_responses_1()

# mean_shift_epanechnik(im, [20,20], n_iter = 10, h =9,plot=True)



# r = mean_shift_epanechnik(im, [60,60], n_iter = 10, h =21,plot=True)
# print("err : ",np.sqrt((r[0]-50)**2+(r[1]-70)**2))
# r = mean_shift_epanechnik(im, [60,60], n_iter = 10, h ="krneki", hs =[61,51,21,15,11,3],plot=True)
# print("err : ",np.sqrt((r[0]-50)**2+(r[1]-70)**2))
# mean_shift_epanechnik(im, [60,60], n_iter = 10, h =19,plot=True)
# mean_shift_epanechnik(im, [60,60], n_iter = 10, h =11,plot=True)

# r = mean_shift_epanechnik(im, [40,40], n_iter = 100, h =21,plot=True)
# print("err : ",np.sqrt((r[0]-50)**2+(r[1]-70)**2))
# r = mean_shift_epanechnik(im, [40,40], n_iter = 100, h =11,plot=True)
# print("err : ",np.sqrt((r[0]-50)**2+(r[1]-70)**2))
# r = mean_shift_epanechnik(im, [40,40], n_iter = 100, h =3,plot=True)
# print("err : ",np.sqrt((r[0]-50)**2+(r[1]-70)**2))

# mean_shift_epanechnik(im, [70,40], n_iter = 10, h =3,plot=True)
# mean_shift_epanechnik(im, [70,40], n_iter = 10, h =21,plot=True)
# mean_shift_epanechnik(im, [70,40], n_iter = 10, h =51,plot=True)
# mean_shift_epanechnik(im, [70,40], n_iter = 10, h = "kr neki", hs = [51,21,11,3], plot=True)

def weighted_color_histogram_epanechnik(region,m,sigma=1, plot=False):
    region = np.array(region)
    for ch in range(3):
        region[:,:,ch] = np.floor(region[:,:,ch] / 255 * (m-1)).astype(np.int32)
        
        
    w, h = region.shape
    kernel = create_epanechnik_kernel(w, h, sigma)
    kernel = kernel[:h,:w].transpose()
    
    
    hist = np.zeros((16,16,16))
    for i in range(region.shape[0]):
        for j in range(region.shape[1]):
            r,g,b = region[i,j,:]
            hist[r,g,b]+=kernel[i,j]
        
        



def binarize(region, m):
    return np.floor(region / 255 * (m-1)).astype(np.int32)
    

def weighted_color_histogram_epanechnik(region,m,sigma=1, plot=False):
    # region
    # m         number of distinct colors (bins)
    # print(region)
    region_binarized = binarize(region, m)
    # print(region_binarized)
    # plt.imshow(region_binarized)
    # plt.show()
    
    
    w, h = region.shape
    kernel = create_epanechnik_kernel(w, h, sigma)
    kernel = kernel[:h,:w].transpose()
    
    region_binarized_flat = region_binarized.reshape(-1)
    kernel_flat = kernel.reshape(-1)
    
    hist = [0]*m
    for c, v in zip(region_binarized_flat,kernel_flat):
        hist[c] += v
        
    s = sum(hist)
    hist = [x/s for x in hist]
        
    if plot:
        plt.bar(range(m),hist)
        plt.show()
        
    return hist

def weighted_color_histogram_epanechnik_RGB(region,m,sigma=1, plot=False):
    hists = []
    for ch in range(region.shape[2]):
        hist = weighted_color_histogram_epanechnik(region[:,:,ch],m,sigma=1, plot=plot)
        hists.append(hist)
    hists = np.array(hists)
    s = np.sum(hists)
    hists /= s
    
    return hists
        

# im = np.array([[1,0,1],[0,2,0],[1,0,1]]) * 255/2
# plt.imshow(im)
# plt.show()
# weighted_color_histogram_epanechnik(im,m=3,plot=True)

# im = plt.imread("/home/lema/Documents/RV/proj1/spegli1.jpg")[:,:,0]
# weighted_color_histogram_epanechnik(im,m=30,plot=True)
# im = plt.imread("/home/lema/Documents/RV/proj1/spegli2.jpg")[:,:,0]
# weighted_color_histogram_epanechnik(im,m=30,plot=True)


# im1 = plt.imread("/home/lema/Documents/RV/proj1/spegli1.jpg")
# qs = weighted_color_histogram_epanechnik_RGB(im,m=16,plot=True)
# im2 = plt.imread("/home/lema/Documents/RV/proj1/spegli2.jpg")
# ps = weighted_color_histogram_epanechnik_RGB(im,m=16,plot=True)


# target je q (hist_old), candidati so p

            
def main(imgs,m,x0,h,n_iter,plot =False, plot_hists=False, plot_mean_shift=False):
    x = x0
    xs = [tuple(x)]
    for i in range(len(imgs)-1):
        im1 = imgs[i]
        qs = weighted_color_histogram_epanechnik_RGB(im1,m=m,plot=plot_hists)
        im2 = imgs[i+1]
        ps = weighted_color_histogram_epanechnik_RGB(im2,m=m,plot=plot_hists)
        W = np.zeros(im1.shape[:2])
        for ch in range(3):
            B = binarize(im2[:,:,ch],m) # b(x_i)
            for r in range(B.shape[0]):
                for c in range(B.shape[1]):
                    W[r,c] += np.sqrt(qs[ch][int(B[r,c])] / ps[ch][int(B[r,c])])
        x = mean_shift_epanechnik(W, x, n_iter=n_iter, h=h, plot=plot_mean_shift)
        if plot:
            plt.imshow(im2)
            plt.scatter(x[0],x[1])
        xs.append(tuple(x))
    return xs


im1 = plt.imread("/home/lema/Documents/RV/proj1/spegli1.jpg")
im1 = cv.resize(im1, (0, 0), fx = 0.1, fy = 0.1)
# qs = weighted_color_histogram_epanechnik_RGB(im,m=16,plot=True)
im2 = plt.imread("/home/lema/Documents/RV/proj1/spegli2.jpg")
im2 = cv.resize(im2, (0, 0), fx = 0.1, fy = 0.1)
# ps = weighted_color_histogram_epanechnik_RGB(im,m=16,plot=True)
main([im1,im2],m=16,x0=[75,75],h=201,n_iter=200,plot=True,plot_hists=False,plot_mean_shift=True)
    
        
    
    
  
