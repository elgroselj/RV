import numpy as np
import cv2
import sys
sys.path.append(".")
from proj3.toolkit_dir.pytracking_toolkit_lite.utils.tracker import Tracker
from my import construct_H, localization_step, exponential_forgetting


class CorrTracker(Tracker):
    
    def __init__(self):
        self.parameters = CorrParams()
    
    def name(self):
        return 'corr'
    
    def gray_clip(cut,shape):
        F = (cut[:,:,0]+cut[:,:,1]+cut[:,:,2])/3 
        Z = np.zeros(shape[:2])
        Z[:cut.shape[0],:cut.shape[1]] = F[:shape[0],:shape[1]]
        return Z
        
        

    def initialize(self, image, region):
        
        assert len(region) == 4

        # bom probala kr 2D window
        # self.window = max(region[2], region[3]) * self.parameters.enlarge_factor
        # self.window = (region[2], region[3]) * self.parameters.enlarge_factor
        self.window = (region[2], region[3]) * self.parameters.enlarge_factor

        left = max(region[0], 0)
        top = max(region[1], 0)

        right = min(region[0] + region[2], image.shape[1] - 1)
        bottom = min(region[1] + region[3], image.shape[0] - 1)

        self.template = image[int(top):int(bottom), int(left):int(right)]
        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)
        self.size = (region[2], region[3])

        
        F = CorrTracker.gray_clip(self.template,self.template.shape)
        
        self.H = construct_H(F,sigma=self.parameters.sigma,
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
        F = CorrTracker.gray_clip(cut,self.template.shape)

        new_locF =localization_step(self.H,F)
        
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
        
        F = CorrTracker.gray_clip(cut,self.template.shape)
        
        Hcalc = construct_H(F,sigma=self.parameters.sigma,
                             lmbd=self.parameters.lmbd)
        self.H = exponential_forgetting(self.H,Hcalc,self.parameters.alpha)
        
        ##############################################################
        

        return [left + max_loc[0], top + max_loc[1], self.size[0], self.size[1]]

class CorrParams():
    def __init__(self):
        self.enlarge_factor = 1 # TODO lahko to dodas
        
        self.sigma = 0.2 # TODO finetune
        self.lmbd = 0.5 # TODO finetune
        
        self.alpha = 0.05 # TODO finetune

