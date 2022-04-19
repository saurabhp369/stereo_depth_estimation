import numpy as np
import cv2
import matplotlib.pyplot as plt

def find_H(img1, img2, pts1, pts2, fundamental_matrix):
    h1, w1 = img1.shape[0], img1.shape[1]
    _, H1, H2 = cv2.stereoRectifyUncalibrated(pts1, pts2, fundamental_matrix, imgSize=(w1, h1))
    return H1, H2


def drawlines(img1src, img2src, lines, pts1src, pts2src):

    line = []
    r, c = img1src.shape
    pts1src = pts1src.astype(int)
    pts2src = pts2src.astype(int)
    img1color = cv2.cvtColor(img1src, cv2.COLOR_GRAY2BGR)
    img2color = cv2.cvtColor(img2src, cv2.COLOR_GRAY2BGR)
    np.random.seed(0)
    for r, pt1, pt2 in zip(lines, pts1src, pts2src):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1color = cv2.line(img1color, (x0, y0), (x1, y1), color, 1)
        img1color = cv2.circle(img1color, tuple(pt1), 5, color, -1)
        img2color = cv2.circle(img2color, tuple(pt2), 5, color, -1)
        epiline = [[x0,y0], [x1,y1]]
        line.append(epiline)
    return img1color, img2color, np.array(line)

def epilines(img1, img2, pts1, pts2, fundamental_matrix, f):

    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, fundamental_matrix)
    lines1 = lines1.reshape(-1, 3)
    img5, img6, l1 = drawlines(img1, img2, lines1, pts1, pts2)
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, fundamental_matrix)
    lines2 = lines2.reshape(-1, 3)
    img3, img4, l2 = drawlines(img2, img1, lines2, pts2, pts1)

    plt.subplot(121), plt.imshow(img5)
    plt.axis('off')
    plt.subplot(122), plt.imshow(img3)
    plt.axis('off')
    plt.suptitle("Epilines in both images")
    plt.savefig(f +'_epilines.png')
    return l1, l2

