# Compress a file using Block Vector Quantization Coding, Output a binary file with the encoding and a restored image
# Calculate the Mean Square Error and the Peak-to-Peak Signal-to-Noise Ratio
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image # PIL is used to directly export Grayscale images
import hashlib # Used to generate SHA256 checksums of images

# Codebook Translation function
# takes the index, g0, g1 as parameters
# returns the matching codebook

def codebookTranslate(index, g0, g1):

    match index:

        case 0:
            return [
                [g0, g0, g1],
                [g0, g1, g1],
                [g1, g1, g1]
            ]

        case 1:
            return [
                [g1, g0, g0],
                [g1, g1, g0],
                [g1, g1, g1]
            ]

        case 2:
            return [
                [g1, g1, g1],
                [g1, g1, g0],
                [g1, g0, g0]
            ]

        case 3:
            return [
                [g1, g1, g1],
                [g0, g1, g1],
                [g0, g0, g1]
            ]

        case 4:
            return [
                [g0, g0, g0],
                [g1, g1, g1],
                [g1, g1, g1]
            ]

        case 5:
            return [
                [g1, g1, g1],
                [g1, g1, g1],
                [g0, g0, g0]
            ]

        case 6:
            return [
                [g1, g1, g0],
                [g1, g1, g0],
                [g1, g1, g0]
            ]

        case 7:
            return [
                [g0, g1, g1],
                [g0, g1, g1],
                [g0, g1, g1]
            ]

        case 8:
            return [
                [g1, g1, g0],
                [g1, g0, g0],
                [g0, g0, g0]
            ]

        case 9:
            return [
                [g0, g1, g1],
                [g0, g0, g1],
                [g0, g0, g0]
            ]

        case 10:
            return [
                [g0, g0, g0],
                [g0, g0, g1],
                [g0, g1, g1]
            ]

        case 11:
            return [
                [g0, g0, g0],
                [g1, g0, g0],
                [g1, g1, g0]
            ]

        case 12:
            return [
                [g1, g1, g1],
                [g0, g0, g0],
                [g0, g0, g0]
            ]

        case 13:
            return [
                [g0, g0, g0],
                [g0, g0, g0],
                [g1, g1, g1]
            ]

        case 14:
            return [
                [g0, g0, g1],
                [g0, g0, g1],
                [g0, g0, g1]
            ]

        case 15:
            return [
                [g1, g0, g0],
                [g1, g0, g0],
                [g1, g0, g0]
            ]
        
# encoding function
def BVQC3encode(in_image_filename, out_encoding_result_filename):

    # read the image
    # try reading file into numpy array with intensity value bound in [0, 255]
    try:
        image = (mpimg.imread(in_image_filename) * 255).astype(np.uint8)

    # exception handling: file not found
    except FileNotFoundError:
        print("File not found.")
        sys.exit("Exiting program...")

    # exception handling: no input
    if in_image_filename == '':
        print("Exiting program...")
        sys.exit("Exiting program...")

    # exception handling: not png
    if os.path.splitext(in_image_filename)[1] != '.png':
        print("Not a PNG file.")
        sys.exit("Exiting program...")

    # exception handling: input is not grayscale
    if image.ndim != 2:
        print("Image color channel is not 2. The image is not a grayscale image.")
        sys.exit("Exiting program...")

    else:

        # display the image
        plt.imshow(image, cmap='gray')
        plt.title("Original Image")
        plt.show()

        # generate SHA256 checksum of original image (to be stored in the binary file and for comparison)
        chksum = hashlib.sha256(image).digest()

        # check if file had been processed before and binary file still exists, skip if yes
        if os.path.exists(out_encoding_result_filename) == True:

            # read the SHA256 checksum from the binary file
            with open(out_encoding_result_filename, 'rb') as f:
                readChksum = f.read(32)
            f.close() # be a good programmer and close the file when you're done :)

            # if the checksums match, skip encoding
            if chksum == readChksum:
                print("File had been processed before.\nSkipping encoding...")
                return

        # set the amount of rows and columns of the blocks
        row = image.shape[0] // 3
        column = image.shape[1] // 3
        
        # initialization of arrays to store encoding output
        meanArray = np.zeros([row, column], dtype = np.uint8) # Mean array, referred to as M in the assignment
        stdDeviationArray = np.zeros([row, column], dtype = np.uint8) # Standard Deviation array, referred to as Sd in the assignment
        indexArray = np.zeros([row, column], dtype = np.uint8) # Index array, referred to as Idx in the assignment

        # top to bottom, left to right
        for x in range(0, row * 3, 3): # left-to-right later

            for y in range(0, column * 3, 3): # top-to-bottom first
                
                block = image[x:x+3, y:y+3] # get 3x3 blocks

                # calculate the mean and standard deviation of the block
                mean = np.mean(block)
                stdDeviation = np.std(block)

                # save the mean and standard deviation into the arrays
                meanArray[(x // 3), (y // 3)] = min(255, (mean // 2) * 2) # mean prime = min(255, 2 * round-down[mean / 2])
                stdDeviationArray[(x // 3), (y // 3)] = min(127, (stdDeviation // 4) * 4) # std prime = min(127, 4 * round-down[std / 4])

                # calculate g0 and g1 for distance calculation
                g0 = np.uint8(max(0, np.int16(meanArray[(x // 3), (y // 3)]) - stdDeviationArray[(x // 3), (y // 3)])) # g0 = max(0, mean prime - std prime)
                g1 = np.uint8(min(255, np.int16(meanArray[(x // 3), (y // 3)]) + stdDeviationArray[(x // 3), (y // 3)])) # g1 = min(255, mean prime + std prime)

                # calculate the distances between the block and codewords
                # I'm using int32 here because maximum total square distance is 255 ^ 2 * 3 * 3 = 585,255, which takes 20 bits
                distance = np.zeros(16, dtype = np.uint32)

                # find the total square distance between the block and each codeword
                # I'm using int16 here because a distance before being square can be negative, and using uint8 would cause overflow after the square operation
                for i in range(16):
                    distance[i] = np.sum( np.square( (block.astype(np.uint16) - codebookTranslate(i, g0, g1)) ) )
                
                # save the index of the closest codeblock
                indexArray[(x // 3), (y // 3)] = np.argmin(distance)

        # output the data structure into a binary file
        with open(out_encoding_result_filename, 'wb') as f:

            # write the header

            f.write(chksum) # 32 bytes: SHA256 checksum

            f.write(np.uint8(8)) # byte 1: header length
            f.write(np.uint8(3)) # byte 2: block size
            f.write(np.uint16(column)) # byte 3-4: number of columns
            f.write(np.uint8(image.shape[1] - (column * 3))) # byte 5: number of skipped columns
            f.write(np.uint16(row)) # byte 6-7: number of rows
            f.write(np.uint8(image.shape[0] - (row * 3))) # byte 8: number of skipped rows

            # write the data

            for x in range(row):
                for y in range(column):
                    # pack mean divide by 2 into 7 bits
                    # pack std divide by 4 into 5 bits
                    # pack index into 4 bits

                    twobyte = np.uint16(
                        np.uint8(meanArray[x, y] // 2) << (5 + 4) |
                        np.uint8(stdDeviationArray[x, y] // 4) << 4 |
                        np.uint8(indexArray[x, y])
                        ).byteswap() # byteswap is done to match the endianness of the task sample file

                    f.write(twobyte)

        f.close() # be a good programmer and close the file when you're done :)
        print("A binary file " + out_encoding_result_filename + " encoded in BVQC has been created.")
    
    # return data struction of encoding output info
    return {"M": meanArray, "Sd": stdDeviationArray, "Idx": indexArray}


# decoding function
def BVQC3decode(in_encoding_result_filename, out_reconstructed_image_filename):
    
    # unpack the binary file
    with open(in_encoding_result_filename, 'rb') as f:

        # parse the header

        # read the SHA256 checksum (we don't need the result, we just need to skip the lines)
        np.frombuffer(f.read(32))

        # read the header length from first byte
        headerLength = np.frombuffer(f.read(1), dtype = np.uint8)[0]

        # read the block size
        blockSize = np.frombuffer(f.read(1), dtype = np.uint8)[0]

        # read the number of columns
        column = np.frombuffer(f.read(2), dtype = np.uint16)[0]

        # read the number of skipped columns
        skippedColumn = np.frombuffer(f.read(1), dtype = np.uint8)[0]

        # read the number of rows
        row = np.frombuffer(f.read(2), dtype = np.uint16)[0]

        # read the number of skipped rows
        skippedRow = np.frombuffer(f.read(1), dtype = np.uint8)[0]

        # parse the content

        meanArray = np.zeros([row, column], dtype = np.uint8) # Mean array, referred to as M in the assignment
        stdDeviationArray = np.zeros([row, column], dtype = np.uint8) # Standard Deviation array, referred to as Sd in the assignment
        indexArray = np.zeros([row, column], dtype = np.uint8) # Index array, referred to as Idx in the assignment

        for x in range(row):

            for y in range(column):
                
                # read next two bytes in big endian
                data = (np.frombuffer(f.read(1), dtype = np.uint8)[0] << 8) + np.frombuffer(f.read(1), dtype = np.uint8)[0]

                # unpack mean from 7 bits
                # unpack std from 5 bits
                # unpack index from 4 bits
                meanArrayMask = 0b1111111000000000
                stdDeviationArrayMask = 0b0000000111110000
                indexArrayMask = 0b0000000000001111

                meanArray[x, y] = ( (data & meanArrayMask) >> (5 + 4) ) * 2
                stdDeviationArray[x, y] = ( ( data & stdDeviationArrayMask ) >> 4 ) * 4
                indexArray[x, y] = data & indexArrayMask
                            
    f.close() # be a good programmer and close the file when you're done :)

    # init image
    image = np.zeros([row * blockSize + skippedRow, column * blockSize + skippedColumn], dtype = np.uint8)

    for x in range(row):

        for y in range(column):

            # get the mean, std and codeword of the block
            mean = meanArray[x, y]
            std = stdDeviationArray[x, y]
            index = indexArray[x, y]

            # get the codeword
            g0 = np.uint8(max(0, mean - np.int16(std))) # g0 = max(0, mean prime - std prime)
            g1 = np.uint8(min(255, mean + np.int16(std))) # g1 = min(255, mean prime + std prime)
            
            # put the codeword into the image
            image[(x * blockSize) : ((x * blockSize) + blockSize), (y * blockSize) : ((y * blockSize) + blockSize)] = codebookTranslate(index, g0, g1)

    # deal with the skipped columns and rows by duplicating the last row and column
    if skippedRow > 0: # row
        for x in range(skippedRow):
            image[(row * blockSize) + x, : ] = image[(row * blockSize) - 1, : ]
   
    if skippedColumn > 0: # column
        for y in range(skippedColumn):
            image[ :, (column * blockSize) + y] = image[ : , (column * blockSize) - 1]

    # show the image
    plt.imshow(image, cmap = 'gray')
    plt.title("Reconstructed Image")
    plt.show()

    # save the image
    im = Image.fromarray(image)
    im.save(out_reconstructed_image_filename)
    print("A reconstructed image file " + out_reconstructed_image_filename + " has been created.")

    return image

# main program

# ask for input for input file name & set the output file name
print("Please input the file name of an image file.")
print("The image file should be a grayscale image in PNG file format. (Type the whole file name including the file extension)")
print("The image file should be in the same directory as the Python program file.")
print("Output binary file will have the extension .bvqc3.")

inFileName = input("Image file name: ")
outFileName = os.path.splitext(inFileName)[0] + ".bvqc3"
inRestoredFileName = os.path.splitext(outFileName)[0] + "-R.png"

BVQC3encode(inFileName, outFileName)
BVQC3decode(outFileName, inRestoredFileName)

# calculate mean square error
originalImage = ( plt.imread(inFileName) * 255 ).astype(np.uint8)
restoredImage = ( plt.imread(inRestoredFileName) * 255).astype(np.uint8)

mse = np.mean( np.square( originalImage - restoredImage.astype(np.uint16) ) )
print("Estimated Mean Square Error:", mse)

# calculate peak-to-peak signal-to-noise ratio
ppsnr = 10 * np.log10( ( 255 ** 2 ) / mse)
print("Estimated Peak-to-Peak Signal-to-Noise Ratio:", ppsnr)