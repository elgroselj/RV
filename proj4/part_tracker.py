import numpy as np
from numpy.random import choice
import matplotlib.pyplot as plt

from ex2_utils import Tracker, create_epanechnik_kernel, extract_histogram, get_patch
from ex4_utils import sample_gauss


class PartTracker(Tracker):
    def visualize(bag, image):
        plt.imshow(image)
        points = np.array([pair["state"][:2] for pair in bag])
        if points.shape[0] > 1:
            plt.scatter(points[:,0],points[:,1],c="r",s=10)
        else:
            plt.scatter(points[0],points[1],c="r",s=10)
        plt.show()
    def visualize1(location, image):
        plt.imshow(image)
        plt.scatter(int(location[0]),int(location[1]),c="orange",s=10)
        plt.show()

    def initialize(self, image, region):

        assert len(region) == 4

        self.window = max(region[2], region[3]) * self.parameters.enlarge_factor

        left = max(region[0], 0)
        top = max(region[1], 0)

        right = min(region[0] + region[2], image.shape[1] - 1)
        bottom = min(region[1] + region[3], image.shape[0] - 1)

        self.template = image[int(top):int(bottom), int(left):int(right)]
        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)
        self.size = (region[2], region[3])
        
        # zracunam histogram q z Epanechnikovim jedrom
        height, width = self.template.shape[:2]
        self.epanechnik_kernel = create_epanechnik_kernel(width, height, sigma=1)
        # self.template = self.template*255 if max(image) <= 1 else self.template
        hist = extract_histogram(self.template,
                                 nbins=self.parameters.nbins,
                                 weights=self.epanechnik_kernel[:height,:width],
                                 mode=self.parameters.hist_mode)
        self.q = hist if np.sum(hist) == 0 else hist/np.sum(hist)
        # noise = sample_gauss(self.parameters.mu, self.parameters.sigma, self.parameters.n)
        # self.bag = [(self.position[0] + x, self.position[1] + y, 1) for x,y in noise]
        samples = sample_gauss(self.position, np.eye(2) * self.parameters.sigma, self.parameters.n)
        self.bag = [{"state":np.array([x,y,0,0]), "weight": 1} for x,y in samples]
        PartTracker.visualize(self.bag, image)

    def track(self, image):

        left = max(round(self.position[0] - float(self.window) / 2), 0)
        top = max(round(self.position[1] - float(self.window) / 2), 0)

        right = min(round(self.position[0] + float(self.window) / 2), image.shape[1] - 1)
        bottom = min(round(self.position[1] + float(self.window) / 2), image.shape[0] - 1)

        if right - left < self.template.shape[1] or bottom - top < self.template.shape[0]:
            return [self.position[0] + self.size[0] / 2, self.position[1] + self.size[1] / 2, self.size[0], self.size[1]]

        # cut = image[int(top):int(bottom), int(left):int(right)]
        
        ###########################################################
        # resample
        p = [pair["weight"] for pair in self.bag]
        p = np.array(p)/sum(p)
        self.bag = choice(self.bag, self.parameters.n, p=p, replace=True)
        
        PartTracker.visualize(self.bag, image)
        
        # dynamic model
        
        for pair in self.bag:
            pair["state"] = np.dot(self.parameters.A, np.array(pair["state"]))
            pair["state"] += sample_gauss([0,0,0,0], self.parameters.Q, 1).flatten()
            
        # ce gre ven iz search region
        # 1 ga zavrzem
        # self.bag = [pair for pair in self.bag if pair["state"][0] >= left and pair["state"][0] <= right 
        #                                         and pair["state"][1] >= top and pair["state"][1] <= bottom]
        
        # 2 mu dam utež na 0
        for pair in self.bag:
            if pair["state"][0] >= left and pair["state"][0] <= right \
                and pair["state"][1] >= top and pair["state"][1] <= bottom:
                    pair["weight"] = 0
            
            
        PartTracker.visualize(self.bag, image)
            
        def hell_dist(p,q):
            P = p.flatten()
            Q = q.flatten()
            return np.sqrt(1/2) * np.sqrt(  np.sum((np.sqrt(P) - np.sqrt(Q))**2) )
        
        # visual similarity
        # hells = []
        # probs = []
        for pair in self.bag:
            height, width = self.template.shape[:2]
            cut,_ = get_patch(image, pair["state"][:2], (height,width))
            hist = extract_histogram(cut,
                                    nbins=self.parameters.nbins,
                                    weights=self.epanechnik_kernel[:height,:width],
                                    mode=self.parameters.hist_mode)
            p = hist if np.sum(hist) == 0 else hist/np.sum(hist)
            hell = hell_dist(p,self.q)
            # hells.append(hell)
            prob = np.exp(-1/2 * (hell**2) / (self.parameters.sigma2**2))
            # probs.append(prob)
            
            pair["weight"] = prob
            
        # probs = np.array(probs) / sum(probs)
        # for pair, prob in zip(self.bag, probs):
        #     pair["weight"] = prob
        
        self.position = np.sum([ pair["state"][:2] * pair["weight"] for pair in self.bag ],axis=0)
        print(self.position)
        PartTracker.visualize1(self.position, image)
        max_loc = (float(self.position[0]) - left - float(self.size[0]) / 2, float(self.position[1]) - top - float(self.size[1])/2)
        ############################################3
        # update q
        height, width = self.template.shape[:2]
        center = (int(self.position[0]),int(self.position[1]))
        cut,_ = get_patch(image, center, (height,width))
        hist = extract_histogram(cut,
                                nbins=self.parameters.nbins,
                                weights=self.epanechnik_kernel[:height,:width],
                                mode=self.parameters.hist_mode)
        p = hist if np.sum(hist) == 0 else hist/np.sum(hist)
        self.q = (1 - self.parameters.alpha) * self.q + self.parameters.alpha * p
        
        #############################################33

        assert (self.position [0] == left + max_loc[0] + float(self.size[0]) / 2 and
                self.position[1] == top + max_loc[1] + float(self.size[1]) / 2)

        return [left + max_loc[0], top + max_loc[1], self.size[0], self.size[1]]

class PartParams():
    def __init__(self):
        self.enlarge_factor = 2
        
        self.nbins = 16
        self.hist_mode = "BGR"
        self.alpha = 0.1
        
        self.q = 1
        self.r = 1
        
        self.n = 100
        self.sigma = 1
        self.sigma2 = 1
        
        self.A = np.array([[1,0,1,0],
                  [0,1,0,1],
                  [0,0,1,0],
                  [0,0,0,1]])
        self.Q = self.q * 1/6 * np.array([[2,0,3,0],
                                          [0,2,0,3],
                                          [3,0,6,0],
                                          [0,3,0,6]])
        
        
        

