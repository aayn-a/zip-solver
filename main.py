'''
Get Puzzle
Make it into a grid
use an algorithm to solve
'''


import cv2
import numpy as np
image = cv2.imread("Screenshot 2026-02-01 at 9.01.12 PM.png")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
#cv2.imshow("thing", blur)
#cv2.waitKey(0)
'''edges = cv2.Canny(blur, 50, 150)
#cv2.imshow("thing", edges)
#cv2.waitKey(0)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
edges_clean = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
edges_clean = cv2.morphologyEx(edges_clean, cv2.MORPH_OPEN, kernel, iterations=1)

contours, _ = cv2.findContours(edges_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)'''
th = cv2.adaptiveThreshold(blur, 255,
                           cv2.ADAPTIVE_THRESH_MEAN_C,
                           cv2.THRESH_BINARY_INV,
                           15, 8)
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
#closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
#closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
cv2.imshow("thing", th)
cv2.waitKey(0)

contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(contours)
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
    approx = cv2.approxPolyDP(c, 0.05*peri, True)
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
warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
warped_edges = cv2.Canny(warped, 50, 150)

cv2.imshow("Warped", warped)
cv2.waitKey(0)
print(warped_edges)
vertical_sum = np.sum(warped_edges, axis=0)
horizontal_sum = np.sum(warped_edges, axis=1)

verticalThreshold = 0.45 * np.max(vertical_sum)
horizontalThreshold = 0.45 * np.max(horizontal_sum)

v_lines = np.where(vertical_sum > verticalThreshold)[0]
h_lines = np.where(horizontal_sum > horizontalThreshold)[0]

def clusterLines(indices, min_gap=10):
    clusters = [[indices[0]]]
    for idx in indices[1:]:
        if idx - clusters[-1][-1] <= min_gap:
            clusters[-1].append(idx)
        else:
            clusters.append([idx])
    return [int(np.mean(c)) for c in clusters]


print(len(clusterLines(v_lines)))
print(len(clusterLines(h_lines)))