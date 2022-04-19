from pprint import pprint
from numpy import float32
from Utils.calibration import *
from Utils.correspondence import *
from Utils.rectification import *

def main():
    data_folders = ['curule', 'pendulum', 'octagon']

    for folder in data_folders:
        K1, K2, b, thresh = get_params(folder)
        i1 = cv2.imread('data/' + folder + '/im0.png')
        i2 = cv2.imread('data/' + folder + '/im1.png')
        image1 = i1.copy()
        image2 = i2.copy()
        img1 = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)
        points1, points2 = feature_matching(img1, img2, folder)
        points1 = np.array(points1).astype(float32)
        points2 = np.array(points2).astype(float32)
        # fundamental_matrix, inliers = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)
        # print(fundamental_matrix)
        A  = construct_A(points1, points2)
        fundamental_mat = compute_fundamental_matrix(A)
        print('********** Dataset {}**********'.format(folder))
        print('Fundamental Matrix')
        pprint(fundamental_mat)
        final_f, final_points = Ransac(points1, points2,2000, 0.002)
        print('Fundamental Matrix after RANSAC')
        pprint(final_f)
        essential_mat = compute_essential_matrix(final_f, K1,K2)
        print('Essential Matrix')
        pprint(essential_mat)
        c,r = estimate_camera_pose(essential_mat)
        r_i = np.eye(3)
        c_i = np.zeros((3,1))
        X_3d = []
        for i in range(len(r)):
            X = linear_triangulation(K1, r_i, r[i], c_i, c[i], points1, points2)
            X = X/X[:,3].reshape(-1,1)
            X_3d.append(X)
        R_final, C_final, X_final = find_correct_pose(r,c,X_3d)
        print('Rotation matrix')
        pprint(R_final)
        print('Translation matrix')
        pprint(C_final)

        ########### Rectification Part ############

        h1, h2 = find_H(image1, image2, points1, points2, final_f)
        h_1, w_1 = img1.shape[0], img1.shape[1]
        h_2, w_2 = img2.shape[0], img2.shape[1]
        # epilines(img1, img2, points1, points2, fundamental_matrix)
        img1_rectified = cv2.warpPerspective(img1, h1, (w_1, h_1))
        img2_rectified = cv2.warpPerspective(img2, h2, (w_2, h_2))
        left_img = img1_rectified.copy()
        right_img = img2_rectified.copy()
        dst1 = cv2.perspectiveTransform(points1.reshape(-1,1,2), h1).squeeze()
        dst2 = cv2.perspectiveTransform(points2.reshape(-1,1,2),h2).squeeze()
        h2_t_inv =  np.linalg.inv(h2.T)
        h1_inv = np.linalg.inv(h1)
        F_rectified = np.dot(h2_t_inv, np.dot(final_f, h1_inv))
        lines1, lines2 = epilines(img1_rectified, img2_rectified, dst1, dst2, F_rectified, folder)

        ########### Correspondence Part ############
        print('Caculating disparity map for '+ folder)
        disparityMap = disparity_map(left_img, right_img)
        plt.figure(figsize=  (10,10))
        plt.axis('off')
        plt.imshow(disparityMap, cmap=plt.cm.RdBu, interpolation='bilinear')
        plt.savefig(folder + '_disparity.png')
        plt.imshow(disparityMap, cmap='gray', interpolation='bilinear')
        plt.axis('off')
        plt.savefig(folder + '_disparity_gray.png')

        ########### Depth Map Part ############
        print('Caculating depth map for '+ folder)
        f = K1[0,0]
        depthMap = (b*f)/(disparityMap + 1e-15)
        depthMap[depthMap > thresh] = thresh
        depthMap = np.uint8(depthMap * 255 / np.max(depthMap))
        plt.figure(figsize=  (10,10))
        plt.imshow(depthMap, cmap=plt.cm.RdBu, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(folder + '_depthMap.png')
        plt.imshow(depthMap, cmap='gray', interpolation='bilinear')
        plt.axis('off')
        plt.savefig(folder + '_depthMap_gray.png')
        
if __name__ == '__main__':
    main()
