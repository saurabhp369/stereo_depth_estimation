import numpy as np
import cv2

def get_params(f):
    if f == 'curule':
        K = np.array([[1758.23, 0 , 977.42],[ 0 ,1758.23, 552.15],[0, 0 ,1]])
        K_hat = np.array([[1758.23, 0 , 977.42],[ 0 ,1758.23, 552.15],[0, 0 ,1]])
        baseline = 88.39
        d_thresh = 100000
    elif f == 'octagon':
        K = np.array([[1742.11, 0 ,804.90], [0, 1742.11, 541.22,],[0, 0, 1]])
        K_hat = np.array([[1742.11, 0 ,804.90], [0, 1742.11, 541.22,],[0, 0, 1]])
        baseline=221.76
        d_thresh = 90000
    else:
        K = np.array([[1729.05, 0, -364.24], [0 ,1729.05, 552.22], [0, 0, 1]])
        K_hat = np.array([[1729.05, 0, -364.24], [0 ,1729.05, 552.22], [0, 0, 1]])
        baseline=537.75
        d_thresh = 100000
    return K, K_hat, baseline, d_thresh

def feature_matching(img1, img2, name):
    orb = cv2.ORB_create()
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    p1 = []
    p2 = []
    for match in matches[:50]:
        p1.append([int(kp1[match.queryIdx].pt[0]),int(kp1[match.queryIdx].pt[1])])
        p2.append([int(kp2[match.trainIdx].pt[0]),int(kp2[match.trainIdx].pt[1])])
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:100],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(name + '_feature_match.png', img3)
    return p1, p2

def construct_A(p1,p2): 
    A = []
    for i in range(p1.shape[0]):
        row = [p1[i,0]*p2[i,0], p1[i,0]*p2[i,1], p1[i,0], p1[i,1]*p2[i,0], p1[i,1]*p2[i,1], p1[i,1], p2[i,0], p2[i,1], 1]
        A.append(row) 
    return A

def compute_fundamental_matrix(A):
    U, S, Vt = np.linalg.svd(np.array(A))
    f= Vt[np.argmin(S)]
    F = np.array([[f[0], f[1],f[2]],
                    [f[3],f[4],f[5]],
                    [f[6],f[7],f[8]]])
    # enforcing rank 2 constraint
    u ,s, vt = np.linalg.svd(F)
    s = np.diag(s)
    s[2,2] = 0
    F = np.dot(u, np.dot(s,vt))
    return F

def compute_error(x1, x2, f_matrix):
    x_1 = np.hstack((x1,1))
    x_2 = np.hstack((x2, 1))
    e = np.dot(x_2.T, np.dot(f_matrix, x_1))
    return np.abs(e)

def Ransac(p1,p2, n, threshold):
    inliers = []
    max_S = 0
    correct_f = None
    for i in range(n):
        points = []
        index = np.random.choice(p1.shape[0], 8, replace = 'False')
        x1_hat = p1[index,:]
        x2_hat = p2[index,:]
        S = 0
        a = construct_A(x1_hat, x2_hat)
        f = compute_fundamental_matrix(a)
        for j in range(p1.shape[0]):
            error = compute_error(p1[j,:], p2[j,:], f)
            if error < threshold:
                points.append(j)
                S+=1
        if S > max_S:
            max_S = S
            inliers = points
            correct_f = f
    if correct_f is None:
        correct_f = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC)
    
    return correct_f, inliers

def compute_essential_matrix(f,k1,k2):
    k2_t = k2.T
    e = np.dot(k2_t,np.dot(f,k1))
    U,S,Vt = np.linalg.svd(e)
    S = [1,1,0]
    E = np.dot(U,np.dot(np.diag(S),Vt))
    return E
    
def estimate_camera_pose(E):
    U, D, Vt = np.linalg.svd(E)
    W = np.array([[0,-1,0], [1,0,0], [0,0,1]])
    C = []
    R = []
    C.append(U[:,2])
    C.append(-U[:,2])
    C.append(U[:,2])
    C.append(-U[:,2])
    R.append(U.dot(W.dot(Vt)))
    R.append(U.dot(W.dot(Vt)))
    R.append(U.dot(W.T.dot(Vt)))
    R.append(U.dot(W.T.dot(Vt)))
    # correct the camera pose
    for i in range(len(R)):
        if(np.linalg.det(R[i]) < 0):
            R[i] = -R[i]
            C[i] = -C[i]

    return C, R

def linear_triangulation(K, R1, R2, C1, C2, x1, x2):
    X = []
    #By general mapping of a pinhole camera
    I = np.eye(3)
    C1 = C1.reshape((3,1))
    C2 = C2.reshape((3,1))
    P1 = K.dot(R1.dot(np.hstack((I, -C1))))
    P2 = K.dot(R2.dot(np.hstack((I, -C2))))

    p1T = P1[0,:].reshape(1,4)
    p2T = P1[1,:].reshape(1,4)
    p3T = P1[2,:].reshape(1,4)

    p_1T = P2[0,:].reshape(1,4)
    p_2T = P2[1,:].reshape(1,4)
    p_3T = P2[2,:].reshape(1,4)

    # constructing the A matrix of AX = 0
    for i in range(x1.shape[0]):
        x = x1[i,0]
        y = x1[i,1]
        x_dash = x2[i,0]
        y_dash = x2[i,1]
        A = [(y * p3T) -  p2T, p1T -  (x * p3T), (y_dash * p_3T) -  p_2T, p_1T -  (x_dash * p_3T) ]
        A = np.array(A).reshape(4,4)
        # Solving for A_triangulation*X = 0
        U , S, Vt = np.linalg.svd(A)
        x = Vt[np.argmin(S)]
        X.append(x)
    return np.array(X)
   
def cheirality_check(r3, X, c):
    # returns the number of points infront of the camera with positive depth
    count = 0
    for x in X:
        x = x.reshape((-1,1)) 
        if (r3.dot(x-c) > 0 and x[2]>0):
            count += 1
    return count

def find_correct_pose(R, C, X_points):
    correct_pose = 0
    max = 0
    for i in range(len(R)):
        R3 = R[i][2,:].reshape((1,-1))
        p_count = cheirality_check(R3, X_points[i][:,0:3], C[i].reshape((-1,1)))
        if p_count > max:
            correct_pose = i
            max = p_count
    
    return R[correct_pose], C[correct_pose], X_points[correct_pose]