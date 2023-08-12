from Project_dir.Sudoku import extract_digit
from Project_dir.Sudoku import find_puzzle
from sudoku import Sudoku
import numpy as np
import argparse
import imutils
import cv2
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.image import img_to_array

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="output/model.h5")
ap.add_argument("-i", "--image", required=True, help="Sudoku.jpg");
ap.add_argument("-d", "--debug", type=int, default=-1, help="whether or not we are visualizing each step of the "
                                                            "pipeline")
args = vars(ap.parse_args())

print("[INFO] loading digit classifier...")
model = load_model(args["model"])

print("[INFO] processing image...")
image = cv2.imread(args["image"])
image = imutils.resize(image, width=600)

(puzzleImage, warped) = find_puzzle(image, debug=args["debug"] > 0)

board = np.zeros((9, 9), dtype="int")

stepX = warped.shape[1]//9
stepY = warped.shape[0]//9

cellLocs = []

for y in range(0,9):
    row = []

    for x in range(0,9):
        startX = x*stepX
        startY = y*stepY
        endX = (x+1)*stepX
        endY = (y+1)*stepY
        row.append((startX,startY,endX,endY))
        cell = warped[startY:endY,startX:endX]
        digit = extract_digit(cell,debug=args["debug"] > 0)
        if digit is not None:
            roi = cv2.resize(digit,(28,28))
            roi = roi.astype("float")/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis = 0)
            pred = model.predict(roi).argmax(axis = 1)[0]
            board[y,x] = pred

    cellLocs.append(row)

print("[INFO] OCR'd sudoku board:")
puzzle = Sudoku(3,3,board = board.tolist())
puzzle.show()

print("[INFO] solving sudoku puzzle...")
solution = puzzle.solve()
solution.show_full()

for (cellRow,boardRow) in zip(cellLocs,solution.board):
    for (box,digit) in zip(cellRow,boardRow):
        startX,startY,endX,endY = box
        textX = int((endX - startX)*0.33)
        textY = int((endY - startY)*-0.2)
        textX += startX
        textY += endY
        cv2.putText(puzzleImage,str(digit),(textX,textY),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,255),2)

cv2.imshow("Sudoku Result",puzzleImage)
cv2.waitKey(0)