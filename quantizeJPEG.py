import numpy as np

def quantizeJPEG(dctBlock, qTable, qScale):
    #inputs must be numpy matrices
    qTableNew = qTable*qScale
    qBlock = np.round(dctBlock/qTableNew).astype(int)
    return qBlock