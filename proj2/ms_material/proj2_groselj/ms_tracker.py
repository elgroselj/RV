import numpy as np
import cv2
import matplotlib.pyplot as plt


import sys
sys.path.append(".")
from ex2_utils import Tracker
import sys
sys.path.append("..")  
from ex2_utils import create_epanechnik_kernel, \
            get_patch, extract_histogram, backproject_histogram
from ohlajanje import simulated_annealing_local_maxima

import random

       
# from my_functions import mean_shift_epanechnik



class MeanShiftTracker(Tracker):
    @staticmethod
    def extract_background_histogram(patch,nbins,k,mode="BGR"):
        mask = np.zeros((patch.shape[0],patch.shape[1]))
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if i < mask.shape[0] * k or mask.shape[0] * (1-k) < i or \
                    j < mask.shape[1] * k or mask.shape[1]* (1-k) < j :
                        mask[i,j] = 1
        hist = extract_histogram(patch,nbins,weights=mask,mode=mode)
        neki = min(hist[hist > 0])
        hist = [1 if x == 0 else min(neki/x,1) for x in hist]
        hist = list(np.array(hist)/sum(hist))
        return hist
    
    @staticmethod
    def mean_shift_epanechnik(pdf, x0, n_iter, width, height, plot=False, anneling_correction=False, verbose=False):
        # pdf           pdf matrix
        # x0 = (a,b)    start position
        # n_iter        number of iterations
        # width, height kernel size
        # print("x0: ", x0)
        x = list(x0)
        xs = [tuple(x)]
        

            

        assert width % 2 == 1
        assert height % 2 == 1
        
        
        #sestavim indeksni matriki
        pol_height = int(height/2)
        gradnik = np.arange(-pol_height,pol_height+1).reshape(height,1)
        indexy = np.repeat(gradnik, width, axis=1)
        
        pol_width = int(width/2)
        gradnik = np.arange(-pol_width,pol_width+1).reshape(width,1)
        indexx = np.repeat(gradnik, height, axis=1)
        
        indexx = indexx.transpose()
        
                
        
        for i in range(n_iter):
            # patch = pdf[(x[0]-pol_height):(x[0]+pol_height+1),(x[1]-pol_width):(x[1]+pol_width+1)]
            patch,_ = get_patch(pdf,x,(height,width))
             # print(patch)
            if verbose:
                plt.imshow(patch)
                plt.colorbar()
                plt.show()
            if patch.shape != indexx.shape:
                # print("Break: zdivergiralo na rob okolice")
                break
            
            xx = np.sum(np.multiply(patch, indexx))

            yy = np.sum(np.multiply(patch, indexy))
            norm = np.sum(patch)
            if verbose:
                print(xx,yy,norm)
            x_ = [0,0] if norm < 10e-7 else np.array([xx, yy]) / norm
            # print(x_)
            # x_ = [int(round(x_[0])),int(round(x_[1]))]
            if verbose:
                print(x_)
            if [int(round(x_[0])),int(round(x_[1]))] == [0,0]:
                # print("Break: meanshift is zero n_iter: ", i)
                break
            
            x[0] += x_[0]
            x[1] += x_[1]
            xs.append(tuple([int(round(x[0])),int(round(x[1]))]))
            
            # print(x)
            
        if anneling_correction:
            x_anneling = simulated_annealing_local_maxima(pdf,x,500,0.5,0.95,plot=plot)
            if pdf[tuple(x_anneling)] > pdf[tuple(map(round,x))]:
                x = x_anneling
        
        if plot or verbose:
            xx = x0[0] # column
            yy = x0[1] # row
            plt.scatter(xx,yy,marker="x",c="black",s=10)
            plt.imshow(pdf)
            a = np.array(xs)
            xx = a[:,0] # columns
            yy = a[:,1] # rows
            plt.scatter(xx,yy,s=5,c='r',marker="o")
            plt.savefig("druga{}".format(random.randint(1,1000)))
            plt.show()      
        return (x, i)

    def initialize(self, image, region):
        # region tuple dolzine 4 (oz. 8)  
        #  0: col , 1: row , 2: delta col, 3: delta row  !!!!!!!!!!!!!!  
        
        # reshape
        # self.k0 = 50 / region[3]
        # self.k1 = 50 / region[2]  
        # image = cv2.resize(image, (0, 0), fx = self.k0, fy = self.k1)
        # region[0] = int(region[0]*self.k1)
        # region[1] = int(region[1]*self.k0)
        # region[2] = 50
        # region[3] = 50

        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]

        # window = veckratnik dolzine vecje od stranic
        self.window = max(region[2], region[3]) * self.parameters.enlarge_factor

        left = max(region[0], 0)
        top = max(region[1], 0)

        right = min(region[0] + region[2], image.shape[1] - 1)
        bottom = min(region[1] + region[3], image.shape[0] - 1)

        self.template = image[int(top):int(bottom), int(left):int(right)]
        # lahko NEcelo stevilo
        # sredina regiona
        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)
        self.size = (region[2], region[3])
        
        # zracunam histogram q z Epanechnikovim jedrom
        height, width = self.template.shape[:2]
        kernel = create_epanechnik_kernel(width, height, sigma=1)
        # self.template = self.template*255 if max(image) <= 1 else self.template
        hist = extract_histogram(self.template, nbins=self.parameters.nbins, weights=kernel[:height,:width],mode=self.parameters.hist_mode)
        self.q = hist if np.sum(hist) == 0 else hist/np.sum(hist)
        if self.parameters.background == True:
            c = MeanShiftTracker.extract_background_histogram(self.template,self.parameters.nbins,self.parameters.background_k,mode=self.parameters.hist_mode)
            self.q = np.multiply(self.q,c)

    def track(self, image):
        
        # image = cv2.resize(image, (0, 0), fx = self.k0, fy = self.k1)
        # if self.k0 > 1 or self.k1 > 1:
        #     print("vecji od 1")


        # krajisca preiskovane okolice (vecji bounding box)
        left = max(round(self.position[0] - float(self.window) / 2), 0)
        top = max(round(self.position[1] - float(self.window) / 2), 0)

        right = min(round(self.position[0] + float(self.window) / 2), image.shape[1] - 1)
        bottom = min(round(self.position[1] + float(self.window) / 2), image.shape[0] - 1)

        # ce je template manjsi od okolice , se prestavim za pol velikosti desno, dol
        # ?? al je to sam rezlika v def pos in bbox
        if right - left < self.template.shape[1] or bottom - top < self.template.shape[0]:
            return [self.position[0] + self.size[0] / 2, self.position[1] + self.size[1] / 2, self.size[0], self.size[1]]

        # okolico izrezem iz slike
        cut = image[int(top):int(bottom), int(left):int(right)]
        
        assert cut.shape == (-int(top)+int(bottom), -int(left)+int(right),3)
        
        ########################vstavi MS##########################

        # matches = cv2.matchTemplate(cut, self.template, cv2.TM_CCOEFF_NORMED)
        # _, _, _, max_loc = cv2.minMaxLoc(matches)
        
        # zracunam histogram p z Epanechnikovim jedrom
        height, width = self.template.shape[:2]
        r1 = max(int(round(self.position[1]-float(self.size[1])/2)), 0)
        # r2 = int(self.position[0]+self.size[0]/2)
        c1 = max(int(round(self.position[0]-float(self.size[0])/2)), 0)
        # c2 = int(self.position[1]+self.size[1]/2)
        region_im = image[r1:(r1+height),c1:(c1+width)]
        
        if region_im.shape != (height,width,3):
            return [self.position[0] + self.size[0] / 2, self.position[1] + self.size[1] / 2, self.size[0], self.size[1]]

        kernel = create_epanechnik_kernel(width, height, sigma=1)
        # self.template = self.template*255 if max(image) <= 1 else self.template
        hist = extract_histogram(region_im, nbins=self.parameters.nbins, weights=kernel[:height,:width],mode=self.parameters.hist_mode)
        self.p = hist if np.sum(hist) == 0 else hist/np.sum(hist)
        if self.parameters.background == True:
            c = MeanShiftTracker.extract_background_histogram(region_im,self.parameters.nbins,self.parameters.background_k,mode=self.parameters.hist_mode)
            self.p = np.multiply(self.p,c)
        
        #TODO tuki ne vem ker hist uporabit od q al od p (tj b(xi) je bin barve od xi v im1 al im2)
        backprojection_q = backproject_histogram(cut, self.q, nbins=self.parameters.nbins)
        backprojection_p = backproject_histogram(cut, self.p, nbins=self.parameters.nbins)
        if np.any(np.isnan(backprojection_q)):
            pass
    
        
        W = np.sqrt(np.divide(backprojection_q , (backprojection_p+ self.parameters.epsi)))
        
        # x0 je center, x1 je center
        # x0 = [int(round(self.position[1])),int(round(self.position[0]))]
        x0 = [int((bottom-top)/2),int((right-left)/2)]
        x1, _ = MeanShiftTracker.mean_shift_epanechnik(pdf=W, x0=x0,
                                        n_iter=self.parameters.mean_shift_n_iter,
                                        width=self.parameters.mean_shift_kernel_shape[1],
                                        height=self.parameters.mean_shift_kernel_shape[0],
                                        plot=False,
                                        anneling_correction=self.parameters.anneling_correction)
        
        #max_loc = zg levo krajisce templata v cut-u
        max_loc = [x1[0] - float(self.size[0]) / 2, x1[1] - float(self.size[1]) / 2]
        
        ##########################################################3

        # left + max_loc[0] + float(self.size[0]) / 2
        # sredina_regiona <- (krajicse_okolice + max) + pol vel regiona
        # + pol vel regiona, pretvorba iz krajisca templata (kjer najboljse ujemanje) v sredisce
        self.position = (left + max_loc[0] + float(self.size[0]) / 2, top + max_loc[1] + float(self.size[1]) / 2)
        
        # popravi q
 
        height, width = self.template.shape[:2]
        r1 = max(int(round(self.position[1]-float(self.size[1])/2)), 0)
        c1 = max(int(round(self.position[0]-float(self.size[0])/2)), 0)

        region_im = image[r1:(r1+height),c1:(c1+width)]
        
        kernel = create_epanechnik_kernel(width, height, sigma=1)
        hist = extract_histogram(region_im, nbins=self.parameters.nbins, weights=kernel[:height,:width],mode=self.parameters.hist_mode)
        q_new = hist if np.sum(hist) == 0 else hist/np.sum(hist)
        if self.parameters.background == True:
            c = MeanShiftTracker.extract_background_histogram(region_im,self.parameters.nbins,self.parameters.background_k,mode=self.parameters.hist_mode)
            q_new = np.multiply(q_new,c)
        self.q = self.q * (1 - self.parameters.alpha) + q_new * self.parameters.alpha
        
        

        # koordinati krajisca zgoraj levo, velikosti regiona
        return [left + max_loc[0], top + max_loc[1], self.size[0], self.size[1]]
    
    def set_params(self,d):
        for k in d.keys():
            if k == "enlarge_factor":
                self.parameters.enlarge_factor = d[k]
            elif k == "epsi":
                self.parameters.epsi = d[k]
            elif k == "mean_shift_kernel_shape":
                self.parameters.mean_shift_kernel_shape = d[k]
            elif k == "mean_shift_n_iter":
                self.parameters.mean_shift_n_iter = d[k]
            elif k == "alpha":
                self.parameters.alpha = d[k]
            elif k == "nbins":
                self.parameters.nbins = d[k]
            elif k == "hist_mode":
                self.parameters.hist_mode = d[k]
            elif k == "background":
                self.parameters.background = d[k]
            elif k == "background_k":
                self.parameters.background_k = d[k]
        

class MSParams():
    def __init__(self):
        self.enlarge_factor = 2
        self.epsi = 1e-3
        self.mean_shift_kernel_shape = (21,21)
        self.mean_shift_n_iter = 20
        self.alpha = 0.1
        self.anneling_correction = False
        self.nbins = 16
        
        self.hist_mode = "BGR"
        self.background = False
        self.background_k = 0.2

