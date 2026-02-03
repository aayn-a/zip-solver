'''
Get Puzzle
Make it into a grid
use an algorithm to solve
'''


import cv2
import numpy as np

''' Warping the image so that the corners don't mess things up '''
image = cv2.imread("Screenshot 2026-02-01 at 9.00.29 PM.png")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Converts to bw
blur = cv2.GaussianBlur(gray, (5, 5), 0)  
th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 8)

cv2.imshow("thing", th)
cv2.waitKey(0)

contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

board_quad = None
print(f"Found {len(contours)} external contours")
for idx, c in enumerate(contours[:10]):
    area = cv2.contourArea(c)
    peri = cv2.arcLength(c, True)
    print(f"  contour[{idx}]: area={area:.1f}, perimeter={peri:.1f}, points={len(c)}")

for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.05*peri, True)
    if len(approx) == 4:                              
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
warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
warped_blur = cv2.GaussianBlur(warped_gray, (5, 5), 0)



# Uses another adaptive threshold to color the grid

th2 = cv2.adaptiveThreshold(
    warped_gray,
    255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY_INV,
    15,   
    5     
)
H, W = th2.shape

#Gets the horizontal and vertical stuff 


#Closes the grid lines
k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
th2c = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, k_close, iterations=1)


v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(0.40 * H)))
v_only = cv2.morphologyEx(th2c, cv2.MORPH_OPEN, v_kernel)

h_kernel =cv2.getStructuringElement(cv2.MORPH_RECT, (int(0.40 * W), 1))
h_only = cv2.morphologyEx(th2c, cv2.MORPH_OPEN, h_kernel)

vertical_sum = np.sum(v_only > 0, axis=0)       
horizontal_sum = np.sum(h_only > 0, axis=1)     

verticalThreshold = 0.45 * np.max(vertical_sum)
horizontalThreshold = 0.45 * np.max(horizontal_sum)

v_lines = np.where(vertical_sum > 0.35 * np.max(vertical_sum))[0]
h_lines = np.where(horizontal_sum > 0.35 * np.max(horizontal_sum))[0]

def clusterLines(indices, min_gap=10):
    indices = np.asarray(indices)
    if indices.size == 0:
        return []
    clusters = [[int(indices[0])]]
    for idx in indices[1:]:
        idx = int(idx)
        if idx - clusters[-1][-1] <= min_gap:
            clusters[-1].append(idx)
        else:
            clusters.append([idx])
    return [int(np.mean(c)) for c in clusters]




print(len(clusterLines(v_lines)))
print(len(clusterLines(h_lines)))

cv2.imshow("v_only (vertical lines)", v_only)
cv2.waitKey(0)
cv2.imshow("h_only (horizontal lines)", h_only)
cv2.waitKey(0)

