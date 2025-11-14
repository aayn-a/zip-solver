'''
Get Puzzle
Make it into a grid
use an algorithm to solve
'''



import pytesseract
import cv2
import numpy as np

image = cv2.imread("puzzle.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)


