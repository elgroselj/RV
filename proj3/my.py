import numpy as np
import matplotlib.pyplot as plt

from ex3_utils import create_gauss_peak, create_cosine_window

def fur(M, mode = "basic"):
    if mode == "basic":
        Mf = np.fft.fft2(M)
        return Mf
    else:
        Ml = np.log(M)
        v = Ml - np.min(Ml)
        Mln = np.divide(v, np.max(v))
        Mlnf = np.fft.fft2(Mln)
        return Mlnf

def ifur(Mlnf, mode = "basic"):
    Mln = np.fft.ifft2(Mlnf)
    assert np.max(Mln.imag) < 1e-14
    Mln = Mln.real
    if mode == "basic":
        return Mln
    else:
        Mn = np.exp(Mln) # TODO pomaga?
        return Mn

# init
def construct_H(F, sigma=1, lmbd=1e-3):
    # F: grayscale image patch around object location
    # lmbd: kako pomembno je, da je H majhen (napram temu, da se dobro nafitta)
    G = np.transpose(create_gauss_peak(target_size=F.shape, sigma=sigma))
    G = G[:F.shape[0],:F.shape[1]]
    # Gf = np.fft.fft2(G)
    Gf = fur(G)
    # Ff = np.fft.fft2(F)
    Ff = fur(F)
    Ffc = np.conjugate(Ff) # TODO ali meč tudi transponira
    Hfc = np.divide(np.multiply(Gf,Ffc), (np.multiply(Ff,Ffc) + lmbd))
    Hf = np.conjugate(Hfc)
    # H = np.fft.ifft2(Hf)
    H = ifur(Hf)
    # plt.imshow(H)
    # plt.show()
    return H

def localization_step(H, F):
    # R correlation response
    coswin = create_cosine_window((F.shape[1],F.shape[0]))
    Fw = np.multiply(F,coswin)
    # Hf = np.fft.fft2(H)
    Hf = fur(H)
    Hfc = np.conjugate(Hf)
    # Ff = np.fft.fft2(Fw)
    Ff = fur(Fw)
    Rf =np.multiply(Hfc,Ff)
    # R = np.fft.ifft2(Rf)
    R = ifur(Rf)
    detected_location = np.unravel_index(np.argmax(R), R.shape)
    return detected_location

def exponential_forgetting(Hprev, Hcalc, alpha):
    # H: updated Hprev with new observation Hcalc
    # alpha: update speed: small alpha for conservative behaviour (0.02 ali 0.1)
    H = (1-alpha)*Hprev + alpha*Hcalc
    return H
    
    
    