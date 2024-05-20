import numpy as np
import cv2

def iBlockDCT(dctBlock):
    #initialize variables
    P = 8
    block = cv2.idct(np.float64(dctBlock)) + 2**(P - 1)
    return block