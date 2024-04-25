import numpy as np
import matplotlib.pyplot as plt

from ex3_utils import create_gauss_peak, create_cosine_window


def fur(M): return np.fft.fft2(M)
def ifur(M): return np.fft.ifft2(M)

def construct_Hfc(F, sigma, lmbd, mode="notsep"):
    # F: grayscale image patch around object location
    # lmbd: kako pomembno je, da je H majhen (napram temu, da se dobro nafitta)
    G = create_gauss_peak(target_size=(F.shape[1],F.shape[0]), sigma=sigma)
    G = G[:F.shape[0],:F.shape[1]]
    Gf = fur(G)
    
    coswin = create_cosine_window((F.shape[1],F.shape[0]))
    Fw = np.multiply(F,coswin)
    Ff = fur(Fw)
    Ffc = np.conjugate(Ff)
    if mode=="sep":
        return (np.multiply(Gf,Ffc), (np.multiply(Ff,Ffc) + lmbd))
    else:
        Hfc = np.divide(np.multiply(Gf,Ffc), (np.multiply(Ff,Ffc) + lmbd))
        return Hfc

def localization_step(Hfc,F,mode="notsep"):
    # R correlation response
    coswin = create_cosine_window((F.shape[1],F.shape[0]))
    Fw = np.multiply(F,coswin)
    Ff = fur(Fw)
    
    Rf = np.multiply(Hfc,Ff)
    R = ifur(Rf)
    detected_location = np.unravel_index(np.argmax(R), R.shape)
    r,c = detected_location
    
    height = F.shape[0]
    width = F.shape[1]
    
    if r > height/2:
        r = r - height
    if c > width/2:
        c = c - width     
    
    old_location = (height/2, width/2)
    r_old, c_old = old_location
    
    r_new = r_old + r
    c_new = c_old + c
    # return (r_new, c_new)
    return (c_new, r_new)

def exponential_forgetting(Hfc_prev, Hfc_calc, alpha):
    # H: updated Hprev with new observation Hcalc
    # alpha: update speed: small alpha for conservative behaviour (0.02 ali 0.1)
    Hfc = (1-alpha)*Hfc_prev + alpha*Hfc_calc
    return Hfc
    
    
    