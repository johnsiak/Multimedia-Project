import numpy as np
import huffman_tables

#helper function to calculate the decimal representation of a binary number
def calcDecimal(currChar):
    if len(currChar) > 0:
        if currChar[0] == '0':
        #calculate complement and convert to decimal
            decValue = int(currChar, 2) - 2**len(currChar) + 1
            return decValue
        elif currChar[0] == '1':
            #convert to decimal
            decValue = int(currChar, 2)
            return decValue
    return 0  #default value

#helper function to return the index of the input char based on the Huffman tables
def finder(list, char):
    for i, value in enumerate(list):
        if value == char:
            return i
    return None

def huffDec(huffStream, type):
    #initialize variables 
    if type:
        #luminance
        DC_Huff = huffman_tables.DC_L
        AC_Huff = huffman_tables.AC_L
    else:
        #chrominance
        DC_Huff = huffman_tables.DC_C
        AC_Huff = huffman_tables.AC_C

    binary_strings = []

    for i, byte in enumerate(huffStream):
        #check if it's the last byte
        if i == len(huffStream) - 1:
            #we don't want zero-padding in the last bits
            binary_strings.append(f'{byte:b}')
        else:
            binary_strings.append(f'{byte:08b}')

    strHuff = ''.join(binary_strings)
    totalSize = len(strHuff)
    currChar = ""
    runSymbols = np.empty((0, 2), int)

    #DC coeff Huffman decoding
    for i in range(totalSize):
        currChar += strHuff[i]
        index = finder(DC_Huff, currChar)
        if index is not None:
            #calculate category
            category = index
            if category != 0:
                #calculate additional bits
                currChar = strHuff[i + 1:i + 1 + category]
                dcAdditionalBits = calcDecimal(currChar)
                endOfDC = i + 1 + category
            else:
                dcAdditionalBits = 0
                endOfDC = i + 1
            #run is always 0 for DC coeff
            runSymbols = np.vstack([runSymbols, [0, dcAdditionalBits]])
            break

    #AC coeff Huffman decoding
    currChar = ""
    i = endOfDC if 'endOfDC' in locals() else 0

    while i < totalSize:
        currChar += strHuff[i]
        index = finder(AC_Huff, currChar)
        if index is not None:
            #EOB
            if index == 0:
                runSymbols = np.vstack([runSymbols, [0, 0]])
                break
            #ZRL
            elif index == 151:
                runSymbols = np.vstack([runSymbols, [15, 0]])
                #reset currChar after ZRL
                currChar = ""
            else:
                remInd = index % 10
                if remInd == 0:
                    category = 10
                    run = index // 10 - 1
                else:
                    category = remInd
                    run = index // 10
                currChar = strHuff[i + 1:i + 1 + category]
                acAdditionalBits = calcDecimal(currChar)
                runSymbols = np.vstack([runSymbols, [run, acAdditionalBits]])
                #next AC coeff
                i += category
                currChar = ""
        i += 1
    return runSymbols