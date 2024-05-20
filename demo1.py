import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import huffman_tables
from convert2ycrcb import *
from convert2rgb import *
from blockDCT import *
from iBlockDCT import *
from quantizeJPEG import *
from dequantizeJPEG import *
from runLength import *
from irunLength import *
from huffEnc import *
from huffDec import *

#1
file_path1 = "baboon.png"
file_path2 = "lena_color_512.png"

image1RGB = np.array(Image.open(file_path1))
image2RGB = np.array(Image.open(file_path2))

#convert to YCRCB
Y1, Cr1, Cb1 = convert2ycrcb(image1RGB, [4,2,2])
Y2, Cr2, Cb2 = convert2ycrcb(image2RGB, [4,4,4])

#convert back to RGB
image1 = convert2rgb(Y1,Cr1,Cb1,[4,2,2])
image2 = convert2rgb(Y2,Cr2,Cb2,[4,4,4])

#plots and comparisons
plt.figure()
plt.subplot(1,2,1)
plt.imshow(image1RGB)
plt.title('baboon original')
plt.subplot(1,2,2)
plt.imshow(image1)
plt.title('baboon after conversions')
plt.show()

plt.figure()
plt.subplot(1,2,1)
plt.imshow(image2RGB)
plt.title('lena original')
plt.subplot(1,2,2)
plt.imshow(image2)
plt.title('lena after conversions')
plt.show()

#2
for file_path, qScale, subimg in [("baboon.png", 0.6, [4,2,2]), ("lena_color_512.png", 5, [4,4,4])]:
    imageRGB = np.array(Image.open(file_path))
    #convert to YCRCB
    Y, Cr, Cb = convert2ycrcb(imageRGB, subimg)
    M, N, _ = imageRGB.shape
    #initialize variables 
    Y_new = np.zeros((M,N))
    Cr_new = np.zeros((M,N))
    Cb_new = np.zeros((M,N))

    block_size = 8
    qTable = huffman_tables.luminance_quantization_table
    #iterate 8x8 Y blocks
    for i in range(0, M, block_size):
        for j in range(0, N, block_size):
            #encoding
            blockY = Y[i:i + block_size, j:j + block_size]
            dctYBlock = blockDCT(blockY)
            qYblock = quantizeJPEG(dctYBlock, qTable, qScale)
            #decoding
            deqDctYblock = dequantizeJPEG(qYblock, qTable, qScale)
            idctBlockY = iBlockDCT(deqDctYblock)
            Y_new[i:i + block_size, j:j + block_size] = idctBlockY

    qTable = huffman_tables.chrominance_quantization_table
    #iterate 8x8 blocks Cb and Cr blocks
    for i in range(0, M, block_size):
        for j in range(0, N, block_size):
            #encoding
            blockCr = Cr[i:i + block_size, j:j + block_size]
            blockCb = Cb[i:i + block_size, j:j + block_size]
            dctCrBlock = blockDCT(blockCr)
            dctCbBlock = blockDCT(blockCb)
            qCrBlock = quantizeJPEG(dctCrBlock, qTable, qScale)            
            qCbBlock = quantizeJPEG(dctCbBlock, qTable, qScale)
            #decoding
            deqDctCrBlock = dequantizeJPEG(qCrBlock, qTable, qScale)
            deqDctCbBlock = dequantizeJPEG(qCbBlock, qTable, qScale)
            idctBlockCr = iBlockDCT(deqDctCrBlock)
            idctBlockCb = iBlockDCT(deqDctCbBlock)
            Cr_new[i:i + block_size, j:j + block_size] = idctBlockCr
            Cb_new[i:i + block_size, j:j + block_size] = idctBlockCb

    image = convert2rgb(Y_new, Cr, Cb, subimg)
    #plots and comparisons
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(imageRGB)
    plt.title('Original')
    plt.subplot(1,2,2)
    plt.imshow(image)
    plt.title('After DCT conversions')
    plt.show()