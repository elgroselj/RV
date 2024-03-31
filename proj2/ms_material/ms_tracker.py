import numpy as np
import cv2

import sys
sys.path.append(".")
from ex2_utils import Tracker
from ms_material.ex2_utils import create_epanechnik_kernel, \
            get_patch, extract_histogram, backproject_histogram



class MeanShiftTracker(Tracker):

    def initialize(self, image, region):
        # region tuple dolzine 4 (oz. 8)  
        #  0: col , 1: row , 2: delta col, 3: delta row    

        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]

        # window?? = veckratnik dolzine vecje od stranic
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
        width = abs(top-bottom)
        height = abs(right-left)
        kernel = create_epanechnik_kernel(width, height, sigma=1)
        # self.template = self.template*255 if max(image) <= 1 else self.template
        hist = extract_histogram(self.template, nbins=16, weights=kernel)
        self.q = hist/np.sum(hist)

    def track(self, image):

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

        matches = cv2.matchTemplate(cut, self.template, cv2.TM_CCOEFF_NORMED)
        _, _, min_loc, max_loc = cv2.minMaxLoc(matches)

        # ????????????????????????????????????????
        # left + max_loc[0] + float(self.size[0]) / 2
        # sredina_regiona <- (krajicse_okolice + max) + pol vel regiona
        # + pol vel regiona, pretvorba iz krajisca templata (kjer najboljse ujemanje) v sredisce
        self.position = (left + max_loc[0] + float(self.size[0]) / 2, top + max_loc[1] + float(self.size[1]) / 2)

        # koordinati krajisca zgoraj levo, velikosti regiona
        return [left + max_loc[0], top + max_loc[1], self.size[0], self.size[1]]

class MSParams():
    def __init__(self):
        self.enlarge_factor = 2

