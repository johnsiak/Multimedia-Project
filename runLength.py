import numpy as np

def runLength(qBlock, DCpred):
    flat = qBlock.flatten()
    #reshape the flattened array to a 2D array
    reshape = flat.reshape(qBlock.shape)
    #create zigzag pattern
    qBlock = np.concatenate([reshape[::-1, :].diagonal(i)[::(i % 2)*2 - 1] for i in range(1 - reshape.shape[0], reshape.shape[0])])
    
    #DC coeff encoding
    runSymbols = np.array([[0, qBlock[0] - DCpred]])
    
    #AC coeff encoding
    precedingZeros = 0
    for i in range(1, len(qBlock)):
        #add the symbol to the array with the correct number of previous zeros
        if qBlock[i] != 0:
            runSymbols = np.vstack([runSymbols, [precedingZeros, qBlock[i]]])
            precedingZeros = 0
        #remove the last encoded bits if they are 0
        elif i == len(qBlock) - 1:
            while (runSymbols[-1] == [15, 0]).all():
                runSymbols = np.delete(runSymbols, -1, axis=0)
            runSymbols = np.vstack([runSymbols, [0, 0]])
        #if symbol == 0 and not the last bit 
        else:
            precedingZeros += 1
            #we stop encoding more than 16 zeros in a row
            if precedingZeros == 16:
                runSymbols = np.vstack([runSymbols, [15, 0]])
                precedingZeros = 0
    return runSymbols