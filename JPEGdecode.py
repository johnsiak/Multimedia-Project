import numpy as np
from convert2rgb import *
from iBlockDCT import *
from dequantizeJPEG import *
from irunLength import *
from huffDec import *

def JPEGdecode(JPEGenc, subimg, qScale):
    #initialize variables
    encodeInfoObject = JPEGenc[0]
    qTableL = encodeInfoObject.qTableL
    qTableC = encodeInfoObject.qTableC
    DCL = encodeInfoObject.DCL
    DCC = encodeInfoObject.DCC
    ACL = encodeInfoObject.ACL
    ACC = encodeInfoObject.ACC

    #last object so it has the info about the image size
    lastObject = JPEGenc[-1]
    block_size = 8
    M = (lastObject.indHor + 1)*block_size
    N = (lastObject.indVer + 1)*block_size
  
    Y = np.zeros((M,N))
    Cr = np.zeros((M,N))
    Cb = np.zeros((M,N))

    DC_predY = 0
    DC_predCr = 0
    DC_predCb = 0
    for object in JPEGenc[1:]:
        if object.blkType == "Y":
            #get info from tuple
            huffStreamY = object.huffStream
            indHor = object.indHor
            indVer = object.indVer
            runSymbolsY = huffDec(huffStreamY,1)
            qYblock = irunLength(runSymbolsY, DC_predY)
            #DC pred for the next block
            DC_predY = qYblock[0,0]
            deqDctYblock = dequantizeJPEG(qYblock, qTableL, qScale)
            idctBlockY = iBlockDCT(deqDctYblock)
            Y[indHor*block_size:indHor*block_size + block_size, indVer*block_size:indVer*block_size + block_size] = idctBlockY
        elif object.blkType == "Cr":
            #get info from tuple
            huffStreamCr = object.huffStream
            indHor = object.indHor
            indVer = object.indVer
            runSymbolsCr = huffDec(huffStreamCr,0)
            qCrBlock = irunLength(runSymbolsCr, DC_predCr)
            #DC pred for the next block
            DC_predCr = qCrBlock[0,0]
            deqDctCrBlock = dequantizeJPEG(qCrBlock, qTableC, qScale)
            idctBlockCr = iBlockDCT(deqDctCrBlock)
            Cr[indHor*block_size:indHor*block_size + block_size, indVer*block_size:indVer*block_size + block_size] = idctBlockCr
        elif object.blkType == "Cb":
            #get info from tuple
            huffStreamCb = object.huffStream
            indHor = object.indHor
            indVer = object.indVer
            runSymbolsCb = huffDec(huffStreamCb,0)
            qCbBlock = irunLength(runSymbolsCb, DC_predCb)
            #DC pred for the next block
            DC_predCb = qCbBlock[0,0]
            deqDctCbBlock = dequantizeJPEG(qCbBlock, qTableC, qScale)
            idctBlockCb = iBlockDCT(deqDctCbBlock)
            Cb[indHor*block_size:indHor*block_size + block_size, indVer*block_size:indVer*block_size + block_size] = idctBlockCb
        
    #find subimg based on number of zeros on Cr
    #if np.sum(idctBlockCr == 0) == (block_size**2)/2:
    #    subimg = [4,2,2]
    #elif np.sum(idctBlockCr == 0) == (block_size**2)/4:
    #    subimg = [4,0,0]
    #else:
    #    subimg = [4,4,4]

    imgRec = convert2rgb(Y, Cr, Cb, subimg)
    return imgRec