def convert2ycrcb(imageRGB, subimg):
    #making sure that my image can be perfectly divided to 8x8 blocks
    M, N, _ = imageRGB.shape
    M = (M // 8) * 8
    N = (N // 8) * 8
    imageRGB = imageRGB[0:M,0:N,:]

    #initialize variables    
    R = imageRGB[:,:,0]
    G = imageRGB[:,:,1]
    B = imageRGB[:,:,2]
    
    imageY = 0.299*R + 0.587*G + 0.114*B
    imageCr = 128 + 0.5*R - 0.418688*G - 0.081312*B
    imageCb = 128 - 0.168736*R - 0.331264*G + 0.5*B

    if subimg == [4,2,2] or subimg == [4,2,0]:
        for i in range(M):
            #odd rows and subimg = 4:2:0
            if i % 2 == 1 and subimg == [4,2,0]:
                imageCr[i, :] = 128
                imageCb[i, :] = 128
                continue
            for j in range(N):
                #odd columns
                if j % 2 == 1:
                    imageCr[i, j] = 128
                    imageCb[i, j] = 128
    return imageY, imageCr, imageCb