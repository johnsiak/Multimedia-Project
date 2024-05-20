import huffman_tables
from convert2ycrcb import *
from blockDCT import *
from quantizeJPEG import *
from runLength import *
from huffEnc import *

class EncodeInfo:
    def __init__(self, qTableL, qTableC, DCL, DCC, ACL, ACC):
        self.qTableL = qTableL
        self.qTableC = qTableC
        self.DCL = DCL
        self.DCC = DCC
        self.ACL = ACL
        self.ACC = ACC

class BlockEncodeInfo:
    def __init__(self, blkType, indHor, indVer, huffStream):
        self.blkType = blkType
        self.indHor = indHor
        self.indVer = indVer
        self.huffStream = huffStream

def JPEGencode(img, subimg, qScale):
    #initialize first element of tuple
    JPEGenc = [EncodeInfo(huffman_tables.luminance_quantization_table, huffman_tables.chrominance_quantization_table,
               huffman_tables.DC_L, huffman_tables.DC_C, huffman_tables.AC_L, huffman_tables.AC_C)]
    Y, Cr, Cb = convert2ycrcb(img, subimg)
    M, N, _ = img.shape
    
    #initialize variables
    block_size = 8
    DC_predY = 0
    DC_predCr = 0
    DC_predCb = 0
    
    #iterate 8x8 Y blocks
    indHor = 0
    for i in range(0, M, block_size):
        indVer = 0
        for j in range(0, N, block_size):
            #encoding
            blockY = Y[i:i + block_size, j:j + block_size]
            blockCr = Cr[i:i + block_size, j:j + block_size]
            blockCb = Cb[i:i + block_size, j:j + block_size]
            
            dctYBlock = blockDCT(blockY)
            dctCrBlock = blockDCT(blockCr)
            dctCbBlock = blockDCT(blockCb)

            qTable = huffman_tables.luminance_quantization_table
            qYblock = quantizeJPEG(dctYBlock, qTable, qScale)
            
            qTable = huffman_tables.chrominance_quantization_table
            qCrBlock = quantizeJPEG(dctCrBlock, qTable, qScale)
            qCbBlock = quantizeJPEG(dctCbBlock, qTable, qScale)

            runSymbolsY = runLength(qYblock, DC_predY)
            runSymbolsCr = runLength(qCrBlock, DC_predCr)
            runSymbolsCb = runLength(qCbBlock, DC_predCb)

            #DC pred for the next block
            DC_predY = qYblock[0,0]
            DC_predCr = qCrBlock[0,0]
            DC_predCb = qCbBlock[0,0]
            huffStreamY = huffEnc(runSymbolsY,1)
            huffStreamCr = huffEnc(runSymbolsCr,0)
            huffStreamCb = huffEnc(runSymbolsCb,0)

            JPEGenc.append(BlockEncodeInfo("Y", indHor, indVer, huffStreamY))
            JPEGenc.append(BlockEncodeInfo("Cr", indHor, indVer, huffStreamCr))
            JPEGenc.append(BlockEncodeInfo("Cb", indHor, indVer, huffStreamCb))
            indVer += 1
        indHor += 1
    return tuple(JPEGenc)