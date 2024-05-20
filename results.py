from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from JPEGencode import *
from JPEGdecode import *
from huffEnc import *
from huffDec import *

file_path1 = "baboon.png"
file_path2 = "lena_color_512.png"

image1RGB = np.array(Image.open(file_path1))
image2RGB = np.array(Image.open(file_path2))

#change quantize tables and remove high frequency coefficients
num = [20, 40, 50, 60, 63]
#Image 1
subimg = [4,4,4]
qScale = 1
for n in num:
    JPEGenc = JPEGencode(image1RGB, subimg, qScale)
    imgRec = JPEGdecode(JPEGenc, subimg, qScale)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image1RGB)
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(imgRec)
    plt.title(f"Reconstructed Image - Zeroing the last {n} High Frequency Coefficients.")
    plt.show()

#Image 2 
for n in num:
    JPEGenc = JPEGencode(image2RGB, subimg, qScale)
    imgRec = JPEGdecode(JPEGenc, subimg, qScale)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image2RGB)
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(imgRec)
    plt.title(f"Reconstructed Image - Zeroing the last {n} High Frequency Coefficients.")
    plt.show()

#Statistics
#Image 1 
subimg = [4,2,2]
qScale = [0.1, 0.3, 0.6, 1, 2, 5, 10]
MSE = np.zeros(len(qScale))
bitNumber = np.zeros(len(qScale))
compressRatio = np.zeros(len(qScale))
for i, scale in enumerate(qScale):
    JPEGenc = JPEGencode(image1RGB, subimg, scale)
    imgRec = JPEGdecode(JPEGenc, subimg, scale)
    MSE[i] = np.sum((image1RGB - imgRec)**2)/np.prod(image1RGB.shape)

    for j in range(1, len(JPEGenc)):
        currStruct = JPEGenc[j]
        bitNumber[i] += len(currStruct.huffStream)*8

    compressRatio[i] = (np.prod(image1RGB.shape)*8)/bitNumber[i]

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(image1RGB)
    plt.title('Original Image')

    plt.subplot(1, 3, 2)
    plt.imshow(imgRec)
    plt.title(f"Reconstructed Image - Subsampling {subimg[0]}:{subimg[1]}:{subimg[2]}, qScale = {scale}")

    plt.subplot(1, 3, 3)
    plt.imshow(image1RGB - imgRec)
    plt.title('Error on reconstruction')
    plt.show()

plt.figure()
plt.plot(qScale, MSE, '-o')
plt.title('First Image - Mean Square Error')
plt.xlabel('qScale')
plt.ylabel('MSE')
plt.grid(True)
plt.show()

plt.figure()
plt.plot(qScale, compressRatio, '-o')
plt.title('First Image - Compression Ratio')
plt.xlabel('qScale')
plt.ylabel('Compression Ratio')
plt.grid(True)
plt.show()

plt.figure()
plt.plot(qScale, bitNumber, '-o')
plt.title('First Image - Number of bits [Encoded Image]')
plt.xlabel('qScale')
plt.ylabel('Number of bits')
plt.grid(True)
plt.show()

plt.figure()
plt.plot(bitNumber, MSE, '-o')
plt.title('First Image - Mean Square Error and Number of bits')
plt.xlabel('Number of bits')
plt.ylabel('MSE')
plt.grid(True)
plt.show()

#Image 2
subimg = [4,4,4]
qScale = [0.1, 0.3, 0.6, 1, 2, 5, 10]
MSE = np.zeros(len(qScale))
bitNumber = np.zeros(len(qScale))
compressRatio = np.zeros(len(qScale))
for i, scale in enumerate(qScale):
    JPEGenc = JPEGencode(image2RGB, subimg, scale)
    imgRec = JPEGdecode(JPEGenc, subimg, scale)
    MSE[i] = np.sum((image2RGB - imgRec)**2)/np.prod(image2RGB.shape)

    for j in range(1, len(JPEGenc)):
        currStruct = JPEGenc[j]
        bitNumber[i] += len(currStruct.huffStream)*8

    compressRatio[i] = (np.prod(image2RGB.shape)*8)/bitNumber[i]

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(image2RGB)
    plt.title('Original Image')

    plt.subplot(1, 3, 2)
    plt.imshow(imgRec)
    plt.title(f"Reconstructed Image - Subsampling {subimg[0]}:{subimg[1]}:{subimg[2]}, qScale = {scale}")

    plt.subplot(1, 3, 3)
    plt.imshow(image2RGB - imgRec)
    plt.title('Error on reconstruction')
    plt.show()

plt.figure()
plt.plot(qScale, MSE, '-o')
plt.title('Second Image - Mean Square Error')
plt.xlabel('qScale')
plt.ylabel('MSE')
plt.grid(True)
plt.show()

plt.figure()
plt.plot(qScale, compressRatio, '-o')
plt.title('Second Image - Compression Ratio')
plt.xlabel('qScale')
plt.ylabel('Compression Ratio')
plt.grid(True)
plt.show()

plt.figure()
plt.plot(qScale, bitNumber, '-o')
plt.title('Second Image - Number of bits [Encoded Image]')
plt.xlabel('qScale')
plt.ylabel('Number of bits')
plt.grid(True)
plt.show()

plt.figure()
plt.plot(bitNumber, MSE, '-o')
plt.title('Second Image - Mean Square Error and Number of bits')
plt.xlabel('Number of bits')
plt.ylabel('MSE')
plt.grid(True)
plt.show()