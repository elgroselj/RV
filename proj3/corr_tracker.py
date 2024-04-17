import numpy as np
import sys
# sys.path.append("/home/lema/Documents/RV/proj3/toolkit/pytracking-toolkit-lite/")
try:
    from utils.tracker import Tracker
except Exception:
    from ex2_utils import Tracker
from my import construct_Hfc, localization_step, exponential_forgetting


class CorrTracker(Tracker):
    
    def __init__(self):
        self.parameters = CorrParams()
    
    def name(self):
        return 'corr'
    
    def gray_clip(cut,shape):
        shape = (shape[1],shape[0])
        F = (cut[:,:,0]+cut[:,:,1]+cut[:,:,2])/3 
        Z = np.zeros(shape[:2])
        Z[:cut.shape[0],:cut.shape[1]] = F[:shape[0],:shape[1]]
        return Z

    def initialize(self, image, region):
        
        assert len(region) == 4

        # bom probala kr 2D window
        # self.window = max(region[2], region[3]) * self.parameters.enlarge_factor
        # self.window = (region[2], region[3]) * self.parameters.enlarge_factor
        self.window = (region[2]*self.parameters.enlarge_factor, region[3]*self.parameters.enlarge_factor)
        self.window_shape = (int(self.window[0]),int(self.window[1]))

        left = max(region[0], 0)
        top = max(region[1], 0)

        right = min(region[0] + region[2], image.shape[1] - 1)
        bottom = min(region[1] + region[3], image.shape[0] - 1)

        self.template = image[int(top):int(bottom), int(left):int(right)]
        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)
        self.size = (region[2], region[3])
        
        
        # krajisca preiskovane okolice (vecji bounding box)
        left = max(round(self.position[0] - float(self.window[0]) / 2), 0)
        top = max(round(self.position[1] - float(self.window[1]) / 2), 0)

        right = min(round(self.position[0] + float(self.window[0]) / 2), image.shape[1] - 1)
        bottom = min(round(self.position[1] + float(self.window[1]) / 2), image.shape[0] - 1)

        cut = image[int(top):int(bottom), int(left):int(right)]
        F = CorrTracker.gray_clip(cut,self.window_shape)
        
        self.Hfc = construct_Hfc(F,sigma=self.parameters.sigma,
                             lmbd=self.parameters.lmbd)
        

    def track(self, image):

        # krajisca preiskovane okolice (vecji bounding box)
        left = max(round(self.position[0] - float(self.window[0]) / 2), 0)
        top = max(round(self.position[1] - float(self.window[1]) / 2), 0)

        right = min(round(self.position[0] + float(self.window[0]) / 2), image.shape[1] - 1)
        bottom = min(round(self.position[1] + float(self.window[1]) / 2), image.shape[0] - 1)

        # ce je template manjsi od okolice , se prestavim za pol velikosti desno, dol
        # ?? al je to sam rezlika v def pos in bbox
        if right - left < self.template.shape[1] or bottom - top < self.template.shape[0]:
            return [self.position[0] + self.size[0] / 2, self.position[1] + self.size[1] / 2, self.size[0], self.size[1]]

        # okolico izrezem iz slike
        cut = image[int(top):int(bottom), int(left):int(right)]
        
        ######################corr tracking################
        # to grayscale
        F = CorrTracker.gray_clip(cut,self.window_shape)

        new_locF = localization_step(self.Hfc,F)
        
        self.position = (left + new_locF[0], top + new_locF[1])
        # max_loc: zg levo krajisce templata v cut-u
        max_loc = (float(self.position[0]) - left - float(self.size[0]) / 2, float(self.position[1]) - top - float(self.size[1])/2)
        #####################################################3

        assert self.position == (left + max_loc[0] + float(self.size[0]) / 2, top + max_loc[1] + float(self.size[1]) / 2)

        #################### update ###############################
        
        # krajisca preiskovane okolice (vecji bounding box)
        left = max(round(self.position[0] - float(self.window[0]) / 2), 0)
        top = max(round(self.position[1] - float(self.window[1]) / 2), 0)

        right = min(round(self.position[0] + float(self.window[0]) / 2), image.shape[1] - 1)
        bottom = min(round(self.position[1] + float(self.window[1]) / 2), image.shape[0] - 1)

        # okolico izrezem iz slike
        cut = image[int(top):int(bottom), int(left):int(right)]
        
        F = CorrTracker.gray_clip(cut,self.window_shape)
        
        Hfc_calc = construct_Hfc(F,sigma=self.parameters.sigma,
                             lmbd=self.parameters.lmbd)
        self.Hfc = exponential_forgetting(self.Hfc,Hfc_calc,self.parameters.alpha)
        
        ##############################################################
        

        return [left + max_loc[0], top + max_loc[1], self.size[0], self.size[1]]

class CorrParams():
    def __init__(self):
        self.enlarge_factor = 1.5
        
        self.sigma = 3
        self.lmbd = 10
        
        self.alpha = 0.05

