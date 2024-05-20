from PIL import Image
from scipy.stats import entropy
import matplotlib.pyplot as plt
import numpy as np
from JPEGencode import *
from JPEGdecode import *
from huffEnc import *
from huffDec import *
from convert2ycrcb import *

file_path1 = "baboon.png"
file_path2 = "lena_color_512.png"

image1RGB = np.array(Image.open(file_path1))
image2RGB = np.array(Image.open(file_path2))

qTableL = huffman_tables.luminance_quantization_table
qTableC = huffman_tables.chrominance_quantization_table

#Image 1
#Y
subimg = [4,2,2]
qScale = 0.6
imageY, imageCb, imageCr = convert2ycrcb(image1RGB, subimg)
RowNumberY, ColumnNumberY = imageY.shape

allQuantY = []
allRunlengthY = []
DC_PredY = 0

for row in range(0, RowNumberY, 8):
    for column in range(0, ColumnNumberY, 8):
        blockY = imageY[row:row + 8, column:column + 8]
        dctYBlock = blockDCT(blockY)
        qYblock = quantizeJPEG(dctYBlock, qTableL, qScale)
        runSymbolsY = runLength(qYblock, DC_PredY)
        DC_PredY = qYblock[0, 0]
        allRunlengthY.extend(runSymbolsY)
        allQuantY.append(qYblock)

#Cr & Cb
RowNumberC4Cb, ColumnNumberCrCb = imageCb.shape
allQuantCb = []
allQuantCr = []
allRunlengthCb = []
allRunlengthCr = []
DC_PredCr = 0
DC_PredCb = 0

for row in range(0, RowNumberC4Cb, 8):
    for column in range(0, ColumnNumberCrCb, 8):
        blockCr = imageCr[row:row + 8, column:column + 8]
        blockCb = imageCb[row:row + 8, column:column + 8]

        dctblockCr = blockDCT(blockCr)
        dctblockCb = blockDCT(blockCb)

        qCrBlock = quantizeJPEG(dctblockCr, qTableC, qScale)
        qCbBlock = quantizeJPEG(dctblockCb, qTableC, qScale)

        runSymbolsCr = runLength(qCrBlock, DC_PredCr)
        runSymbolsCb = runLength(qCbBlock, DC_PredCb)

        DC_PredCr = qCrBlock[0, 0]
        DC_PredCb = qCbBlock[0, 0] 
        
        allRunlengthCr.extend(runSymbolsCr)
        allRunlengthCb.extend(runSymbolsCb)

        allQuantCr.append(qCrBlock)
        allQuantCb.append(qCbBlock)

red = image1RGB[:, :, 0]
green = image1RGB[:, :, 1]
blue = image1RGB[:, :, 2]

#calculate entropy of Spatial Domain
entropySpatialRed = entropy(red.flatten(), base=2)
entropySpatialGreen = entropy(green.flatten(), base=2)
entropySpatialBlue = entropy(blue.flatten(), base=2)
entropySpatialAll = entropySpatialRed + entropySpatialGreen + entropySpatialBlue

print("[First Image] The entropy for the Spatial Domain is: {:.6f} per symbol.".format(entropySpatialAll))
print("[First Image] The entropy for the Spatial Domain is: {:.6e}.".format(entropySpatialAll*len(image1RGB.flatten())))

#calculate entropy for the quantized values
allQuantY = np.concatenate(allQuantY)
allQuantCb = np.concatenate(allQuantCb)
allQuantCr = np.concatenate(allQuantCr)

#check for zeros and add epsilon
epsilon = 1e-10
allQuantY = np.maximum(allQuantY, epsilon)
allQuantCb = np.maximum(allQuantCb, epsilon)
allQuantCr = np.maximum(allQuantCr, epsilon)

entropyQuantY = entropy(allQuantY.flatten(), base=2)
entropyQuantCb = entropy(allQuantCb.flatten(), base=2)
entropyQuantCr = entropy(allQuantCr.flatten(), base=2)
entropyQuantAll = entropyQuantY + entropyQuantCb + entropyQuantCr

print("\n[First Image] The entropy for the Quantize DCT Coefficients is: {:.6f} per symbol.".format(entropyQuantAll))
print("[First Image] The entropy for the Quantize DCT Coefficients is: {:.6e}.".format(entropyQuantAll*len(allQuantY.flatten())))

#calculate entropy for run length
allRunlengths = np.vstack((allRunlengthY, allRunlengthCb, allRunlengthCr))

#check for zeros and add epsilon
allRunlengths = np.maximum(allRunlengths, epsilon)

entropyRunlength = entropy(allRunlengths.flatten(), base=2)

print("\n[First Image] The entropy for the Runlength is: {:.6f} per Symbol".format(entropyRunlength))
print("[First Image] The entropy for the Runlength is: {:.6e}.".format(entropyRunlength*len(allRunlengths.flatten())))

#Image 2
#Y
subimg = [4,4,4]
qScale = 5
imageY, imageCb, imageCr = convert2ycrcb(image2RGB, subimg)
RowNumberY, ColumnNumberY = imageY.shape

allQuantY = []
allRunlengthY = []
DC_PredY = 0

for row in range(0, RowNumberY, 8):
    for column in range(0, ColumnNumberY, 8):
        blockY = imageY[row:row + 8, column:column + 8]
        dctYBlock = blockDCT(blockY)
        qYblock = quantizeJPEG(dctYBlock, qTableL, qScale)
        runSymbolsY = runLength(qYblock, DC_PredY)

        DC_PredY = qYblock[0, 0]
        allRunlengthY.extend(runSymbolsY)
        allQuantY.append(qYblock)

#Cr & Cb
RowNumberCrCb, ColumnNumberCrCb = imageCb.shape
allQuantCb = []
allQuantCr = []
allRunlengthCb = []
allRunlengthCr = []
DC_PredCr = 0
DC_PredCb = 0

for row in range(0, RowNumberCrCb, 8):
    for column in range(0, ColumnNumberCrCb, 8):
        blockCr = imageCr[row:row + 8, column:column + 8]
        blockCb = imageCb[row:row + 8, column:column + 8]

        dctCrBlock = blockDCT(blockCr)
        dctCbBlock = blockDCT(blockCb)

        qCrBlock = quantizeJPEG(dctCrBlock, qTableC, qScale)
        qCbBlock = quantizeJPEG(dctCbBlock, qTableC, qScale)

        runSymbolsCr = runLength(qCrBlock, DC_PredCr)
        runSymbolsCb = runLength(qCrBlock, DC_PredCb)

        DC_PredCr = qCrBlock[0, 0]
        DC_PredCb = qCbBlock[0, 0]
        
        allRunlengthCr.extend(runSymbolsCr)
        allRunlengthCb.extend(runSymbolsCb)
        allQuantCr.append(qCrBlock)
        allQuantCb.append(qCbBlock)

red = image2RGB[:, :, 0]
green = image2RGB[:, :, 1]
blue = image2RGB[:, :, 2]

#calculate entropy of Spatial Domain
entropySpatialRed = entropy(red.flatten(), base=2)
entropySpatialGreen = entropy(green.flatten(), base=2)
entropySpatialBlue = entropy(blue.flatten(), base=2)
entropySpatialAll = entropySpatialRed + entropySpatialGreen + entropySpatialBlue

print("\n[Second Image] The entropy for the Spatial Domain is: {:.6f} per symbol.".format(entropySpatialAll))
print("[Second Image] The entropy for the Spatial Domain is: {:.6e}.".format(entropySpatialAll*len(image2RGB.flatten())))

#calculate entropy for the quantized values
allQuantY = np.concatenate(allQuantY)
allQuantCb = np.concatenate(allQuantCb)
allQuantCr = np.concatenate(allQuantCr)

#check for zeros and add epsilon
epsilon = 1e-10
allQuantY = np.maximum(allQuantY, epsilon)
allQuantCb = np.maximum(allQuantCb, epsilon)
allQuantCr = np.maximum(allQuantCr, epsilon)

entropyQuantY = entropy(allQuantY.flatten(), base=2)
entropyQuantCb = entropy(allQuantCb.flatten(), base=2)
entropyQuantCr = entropy(allQuantCr.flatten(), base=2)
entropyQuantAll = entropyQuantY + entropyQuantCb + entropyQuantCr

print("\n[Second Image] The entropy for the Quantize DCT Coefficients is: {:.6f} per symbol.".format(entropyQuantAll))
print("[Second Image] The entropy for the Quantize DCT Coefficients is: {:.6e}.".format(entropyQuantAll*len(allQuantY.flatten())))

#calculate entropy for run length
allRunlengths = np.vstack((allRunlengthY, allRunlengthCb, allRunlengthCr))

#check for zeros and add epsilon
allRunlengths = np.maximum(allRunlengths, epsilon)

entropyRunlength = entropy(allRunlengths.flatten(), base=2)

print("\n[Second Image] The entropy for the Runlength is: {:.6f} per Symbol".format(entropyRunlength))
print("[Second Image] The entropy for the Runlength is: {:.6e}.".format(entropyRunlength*len(allRunlengths.flatten())))