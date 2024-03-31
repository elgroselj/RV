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

       
# from my_functions import mean_shift_epanechnik



class MeanShiftTracker(Tracker):
    
    def mean_shift_epanechnik(self, pdf, x0, n_iter, width, height, plot=False):
        # pdf           pdf matrix
        # x0 = (a,b)    start position
        # n_iter        number of iterations
        # width, height kernel size
        print("x0: ", x0)
        x = x0
        xs = [tuple(x)]
        
        
            
        if plot:
            plt.imshow(pdf)

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
            # padded_patch, _ = get_patch(pdf, x, (width,height))
            # plt.imshow(padded_patch)
            # plt.show()
            # a = (int(padded_patch.shape[0]/2)-pol_height)
            # b = (int(padded_patch.shape[0]/2)+pol_height)+1
            # c = (int(padded_patch.shape[1]/2)-pol_width)
            # d = (int(padded_patch.shape[1]/2)+pol_width)+1
            # patch = padded_patch[a:b,c:d]
            patch = pdf[(x[0]-pol_height):(x[0]+pol_height+1),(x[1]-pol_width):(x[1]+pol_width+1)]
             # print(patch)
            # plt.imshow(patch)
            # plt.show()
            if patch.shape != indexx.shape:
                print("Break: zdivergiralo na rob okolice")
                break

            xx = np.sum(np.multiply(patch, indexx))

            yy = np.sum(np.multiply(patch, indexy))
            norm = np.sum(patch)
            # print(xx,yy,norm)
            x_ = [0,0] if norm == 0 else np.array([xx, yy]) / norm
            # print(x_)
            x_ = [int(round(x_[0])),int(round(x_[1]))]
            # print(x_)
            if x_ == [0,0]:
                print("Break: close enough n_iter: ", i)
                break
            
            x[0] += x_[0]
            x[1] += x_[1]
            xs.append(tuple(x))
            
            print(x)
        
        if plot:
            a = np.array(xs)
            plt.scatter(a[:,0],a[:,1],s=5,c='r',marker="o")
            xs = []
        plt.show()      
        return x

    def initialize(self, image, region):
        # region tuple dolzine 4 (oz. 8)  
        #  0: col , 1: row , 2: delta col, 3: delta row  !!!!!!!!!!!!!!  

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
        hist = extract_histogram(self.template, nbins=16, weights=kernel[:height,:width])
        self.q = hist/np.sum(hist)

    def track(self, image):

        # krajisca preiskovane okolice (vecji bounding box)
        left = max(round(self.position[0] - float(self.window) / 2), 0)
        top = max(round(self.position[1] - float(self.window) / 2), 0)

        right = min(round(self.position[0] + float(self.window) / 2), image.shape[1] - 1)
        bottom = min(round(self.position[1] + float(self.window) / 2), image.shape[0] - 1)
        print("image shape ", image.shape)
        print("left,top,right,bottom ",left,top,right,bottom)

        # ce je template manjsi od okolice , se prestavim za pol velikosti desno, dol
        # ?? al je to sam rezlika v def pos in bbox
        if right - left < self.template.shape[1] or bottom - top < self.template.shape[0]:
            return [self.position[0] + self.size[0] / 2, self.position[1] + self.size[1] / 2, self.size[0], self.size[1]]

        # okolico izrezem iz slike
        cut = image[int(top):int(bottom), int(left):int(right)]
        
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
        
        print("image shape ", image.shape)
        print("template shape ", self.template.shape)
        print("position ", self.position)
        print("size ", self.size)
        print("r1, c1 ", r1, c1)
        if region_im.shape != (height,width,3):
            return [self.position[0] + self.size[0] / 2, self.position[1] + self.size[1] / 2, self.size[0], self.size[1]]

        kernel = create_epanechnik_kernel(width, height, sigma=1)
        # self.template = self.template*255 if max(image) <= 1 else self.template
        hist = extract_histogram(region_im, nbins=16, weights=kernel[:height,:width])
        self.p = hist/np.sum(hist)
        
        #TODO tuki ne vem ker hist uporabit od q al od p (tj b(xi) je bin barve od xi v im1 al im2)
        backprojection_q = backproject_histogram(cut, self.q, nbins=16)
        backprojection_p = backproject_histogram(cut, self.p, nbins=16)
        
        W = np.sqrt(np.divide(backprojection_q , (backprojection_p+ self.parameters.epsi)))
        
        # x0 je center, x1 je center
        # x0 = [int(round(self.position[1])),int(round(self.position[0]))]
        x0 = [int((bottom-top)/2),int((right-left)/2)]
        x1 = self.mean_shift_epanechnik(pdf=W, x0=x0,
                                        n_iter=self.parameters.mean_shift_n_iter,
                                        width=self.parameters.mean_shift_kernel_shape[1],
                                        height=self.parameters.mean_shift_kernel_shape[0],
                                        plot=False)
        
        #max_loc = zg levo krajisce templata v cut-u
        max_loc = [x1[0] - float(self.size[0]) / 2, x1[1] - float(self.size[1]) / 2]
        
        ##########################################################3

        # left + max_loc[0] + float(self.size[0]) / 2
        # sredina_regiona <- (krajicse_okolice + max) + pol vel regiona
        # + pol vel regiona, pretvorba iz krajisca templata (kjer najboljse ujemanje) v sredisce
        self.position = (left + max_loc[0] + float(self.size[0]) / 2, top + max_loc[1] + float(self.size[1]) / 2)

        # koordinati krajisca zgoraj levo, velikosti regiona
        return [left + max_loc[0], top + max_loc[1], self.size[0], self.size[1]]

class MSParams():
    def __init__(self):
        self.enlarge_factor = 2
        self.epsi = 1e-3
        self.mean_shift_kernel_shape = (21,21)
        self.mean_shift_n_iter = 100

