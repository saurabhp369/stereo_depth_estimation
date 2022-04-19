import numpy as np
import cv2
from tqdm import tqdm


def correspondence(y, x, block_left, right_array, block_size):
    search_block_size = 100
    x_min = max(0, x - search_block_size)
    x_max = min(right_array.shape[1], x + search_block_size)
    first = True
    min_ssd = None
    min_index = None
    for x in range(x_min, x_max):
        block_right = right_array[y: y+block_size,x: x+block_size]
        if block_left.shape != block_right.shape:
            ssd = -1
        else:
            ssd = np.sum(np.square(block_left - block_right))
        if first:
            min_ssd = ssd
            min_index = (y, x)
            first = False
        else:
            if ssd < min_ssd:
                min_ssd = ssd
                min_index = (y, x)

    return min_index

def disparity_map(left_img, right_img):
    block_size = 10
    
    left_img = cv2.resize(left_img, (int(left_img.shape[1]/2), int(left_img.shape[0]/2)))
    right_img = cv2.resize(right_img, (int(right_img.shape[1]/2), int(right_img.shape[0]/2)))
    h, w = left_img.shape
    d_map = np.zeros((h,w))
    for y in tqdm(range(block_size, h-block_size)):
        for x in range(block_size, w-block_size):
            block_left = left_img[y:y + block_size,x:x + block_size]
            min_index = correspondence(y, x, block_left,right_img,block_size)
            d_map[y, x] = abs(min_index[1] - x)
    
    return d_map