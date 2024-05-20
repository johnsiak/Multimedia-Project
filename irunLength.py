import numpy as np

def irunLength(runSymbols, DCpred):
    if len(runSymbols) == 0:
        return np.zeros((8, 8), dtype=int)
    #DC coeff decoding
    newList = [DCpred + runSymbols[0, 1]]
    #remove first row
    runSymbols = np.delete(runSymbols, 0, axis=0)
    #AC coeff decoding
    for value in runSymbols:
        #add preceding zeros
        if value[0] != 0:
            for _ in range(value[0]):
                newList.append(0)
        #add non zero values
        newList.append(value[1])
        #if we are at the end
        if value[0] == 0 and value[1] == 0:
            length = len(newList)
            #add the number of zeros that have not been decoded to create the 8x8 block
            numOfZeros = 63 - length
            for _ in range(numOfZeros):
                newList.append(0)
    
    qBlock = np.zeros((8, 8))
    #zigzag pattern of an 8x8 array
    zigzagPattern = [[1, 2, 6, 7, 15, 16, 28, 29],
                      [3, 5, 8, 14, 17, 27, 30, 43],
                      [4, 9, 13, 18, 26, 31, 42, 44],
                      [10, 12, 19, 25, 32, 41, 45, 54],
                      [11, 20, 24, 33, 40, 46, 53, 55],
                      [21, 23, 34, 39, 47, 52, 56, 61],
                      [22, 35, 38, 48, 51, 57, 60, 62],
                      [36, 37, 49, 50, 58, 59, 63, 64]]

    #create the block from the values of zigzagPattern
    for i, value in enumerate(newList):
        for row in range(8):
            for col in range(8):
                if zigzagPattern[row][col] == i + 1:
                    qBlock[row, col] = value
    return qBlock.astype(int)