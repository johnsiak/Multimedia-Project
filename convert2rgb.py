import numpy as np

def convert2rgb(imageY, imageCr, imageCb, subimg):
    #initialize variables    
    M, N = imageY.shape
    imageRGB = np.zeros((M,N,3))
    #sampling from nearest neighbors
    if subimg == [4,2,2] or subimg == [4,2,0]:
        for i in range(M):
            for j in range(N):
                #odd columns
                if j % 2 == 1:
                    imageCr[i, j] = imageCr[i, j-1]
                    imageCb[i, j] = imageCb[i, j-1]
                #odd rows and subimg = 4:2:0
                if i % 2 == 1 and subimg == [4,2,0]:
                    imageCr[i, j] = imageCr[i-1, j]
                    imageCb[i, j] = imageCb[i-1, j]
    
    R = imageY + 1.402*(imageCr - 128)
    G = imageY - 0.344136*(imageCb - 128) - 0.714136*(imageCr - 128)
    B = imageY + 1.772*(imageCb - 128)

    # Stack the R, G, and B channels along the third dimension
    imageRGB[:, :, 0] = np.clip(R, 0, 255)
    imageRGB[:, :, 1] = np.clip(G, 0, 255)
    imageRGB[:, :, 2] = np.clip(B, 0, 255)
    return imageRGB.astype(np.uint8)