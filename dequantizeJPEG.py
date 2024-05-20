def dequantizeJPEG(qBlock, qTable, qScale):
    #inputs must be numpy matrices
    qTableNew = qTable*qScale
    dctBlock = qBlock*qTableNew
    return dctBlock