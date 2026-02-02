'''
Get Puzzle
Make it into a grid
use an algorithm to solve
'''


import cv2
import numpy as np
image = cv2.imread("puzzle.png")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
print(contours)

board_quad = None
print(f"Found {len(contours)} external contours")
for idx, c in enumerate(contours[:10]):
    area = cv2.contourArea(c)
    peri = cv2.arcLength(c, True)
    print(f"  contour[{idx}]: area={area:.1f}, perimeter={peri:.1f}, points={len(c)}")

for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02*peri, True)
    if len(approx) == 4:                              # if polygon has 4 vertices -> quad
        board_quad = approx.reshape(4, 2).astype(np.float32)
        break



def order_points(pts):
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

rect = order_points(board_quad)
tl, tr, br, bl = rect
widthA = np.linalg.norm(br - bl)
widthB = np.linalg.norm(tr - tl)
maxWidth = int(max(widthA, widthB))
heightA = np.linalg.norm(tr - br)         
heightB = np.linalg.norm(tl - bl)        
maxHeight = int(max(heightA, heightB))

dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]
], dtype=np.float32)

M = cv2.getPerspectiveTransform(rect, dst)
warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
warped_edges = cv2.Canny(warped, 50, 150)

cv2.imshow("Warped", warped_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()