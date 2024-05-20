import numpy as np
import huffman_tables

#helper function to calculate the binary representation of an integer
def calcBinary(num, category):
    #binary value of zero is zero
    if category == 0:
        return '0'

    binary = np.binary_repr(abs(num), width=category)
    if num < 0:
        #calculate complement
        binary = ''.join('1' if bit == '0' else '0' for bit in binary).zfill(category)
    return binary

def huffEnc(runSymbols, type):
    #initialize variables 
    if type:
        #luminance
        DC_Huff = huffman_tables.DC_L
        AC_Huff = huffman_tables.AC_L
    else:
        #chrominance
        DC_Huff = huffman_tables.DC_C
        AC_Huff = huffman_tables.AC_C

    R = runSymbols.shape[0]
    strHuff = []
    
    #DC coeff Huffman encoding
    category = 0
    dcAdditionalBits = ''
    if runSymbols[0, 1]:
        #calculate category
        category = int(np.floor(np.log2(np.abs(runSymbols[0, 1]))) + 1)
        #calculate additional bits
        dcAdditionalBits = calcBinary(runSymbols[0, 1], category)

    #Huffman code of DC Coeff
    index = category
    strHuff.extend([DC_Huff[index], dcAdditionalBits])

    #AC coeff Huffman encoding
    AC_Magn = np.zeros(R - 1, dtype=int)
    acAdditionalBits = ["" for _ in range(R - 1)]

    for i in range(1, R):
        #if ZRL or EOB
        if np.array_equal(runSymbols[i, :], [15, 0]) or np.array_equal(runSymbols[i, :], [0, 0]) or runSymbols[i,1] == 0:
            AC_Magn[i - 1] = 0
            acAdditionalBits[i-1] = ''
        else:
            #calculate category
            category = int(np.floor(np.log2(np.abs(runSymbols[i, 1]))) + 1)
            AC_Magn[i - 1] = category
            #calculate additional bits
            acAdditionalBits[i - 1] = calcBinary(runSymbols[i, 1], category)

    #Huffman code of AC Coeff
    for i in range(1, R):
        index = runSymbols[i, 0]*10 + AC_Magn[i - 1]
        #if ZRL
        if np.array_equal(runSymbols[i, :], [15, 0]):
            index += 1
        strHuff.extend([AC_Huff[index], acAdditionalBits[i - 1]])

    huffStream = ''.join(strHuff)
    #1 byte is 8 bits
    bytes = [huffStream[i:i+8] for i in range(0, len(huffStream), 8)]

    #convert each byte to decimal
    huffStream = [int(byte, 2) for byte in bytes]
    return huffStream