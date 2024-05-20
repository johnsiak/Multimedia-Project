import numpy as np
import cv2

def blockDCT(block):
    #initialize variables 
    P = 8
    dctBlock = cv2.dct(np.float64(block) - 2**(P - 1)) 
    return dctBlock