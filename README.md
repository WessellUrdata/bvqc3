# <img src="logo.svg" width="48px" height="48px" alt="logo"> BVQC3 in Python

An implementation of Block Vector Quantization Coding in Python (and my Year 2 assignment)

## What this is

This is just a simple Python program that takes a PNG file as input and then does lossy compression onto it using Block Vector Quantization Coding (BVQC).

Since the initial commit, the project may have deviated from the task requirements from my assignment. This is done mostly for optimization.

The code quality is probably bad and not very *Pythonic* since I've only had a few tens of hours of Python experience prior to making this. *The result image may also be very lossy*

## Requirements

The program uses a new feature from Python 3.10 `match/case` and thus requires Python >= 3.10.

### Required Python modules

* [Pillow][1]
* [NumPy][2]
* [Matplotlib][3]

## Usage
```
git clone https://github.com/WessellUrdata/bvqc3 && cd ./bvqc3 && python ./bvqc3.py
```

## Example Output

```
$ python ./bvqc3.py 
Please input the file name of an image file.
The image file should be a grayscale image in PNG file format. (Type the whole file name including the file extension)
The image file should be in the same directory as the Python program file.
Output binary file will have the extension .bvqc3.
Image file name: ./img/boats.png
A binary file ./img/boats.bvqc3 encoded in BVQC has been created.
A reconstructed image file ./img/boats-R.png has been created.
Estimated Mean Square Error: 128.1510533807829
Estimated Peak-to-Peak Signal-to-Noise Ratio: 27.053581805030813
```

<img src="./img/boats-R.png" alt="result-image">

[//]: # (Links Reference)
[1]: https://github.com/python-pillow/Pillow/
[2]: https://github.com/numpy/numpy
[3]: https://github.com/matplotlib/matplotlib
