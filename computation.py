import cv2
import numpy as np



def getVij(H, i,j):
    i,j = i-1,j-1
    v_ij = np.array([H[0, i]*H[0, j],
                    H[0, i]*H[1, j] + H[1, i]*H[0, j],
                    H[1, i]*H[1, j],
                    H[2, i]*H[0, j] + H[0, i]*H[2, j],
                    H[2, i]*H[1, j] + H[1, i]*H[2, j],
                    H[2, i]*H[2, j] 
                    ])
    return v_ij


def getIntrinsicMat(H_matrices):

    V = []

    for h in H_matrices:
        V.append(getVij(h,1,2))
        V.append(getVij(h,1,1)-getVij(h,2,2))
    V = np.array(V)
    
    U, S, Vt = np.linalg.svd(V)

    b11, b12, b22, b13, b23, b33 = Vt[np.argmin(S)]

    v0 = (b12*b13 - b11*b23)/(b11*b22 - b12**2)
    lamda = b33 - (b13**2 + v0*(b12*b13 - b11*b23))/b11
    alpha = np.sqrt(lamda/b11)
    beta = np.sqrt(lamda*b11 /(b11*b22 - b12**2))
    gamma = -1*b12*(alpha**2)*beta/lamda
    u0 = gamma*v0/beta -b13*(alpha**2)/lamda

    K = np.array([[alpha, gamma, u0],
                  [0,     beta,  v0],
                  [0,     0,      1]])
    
    return K


def getExtrinsicMat(H, K):

    h1,h2,h3 = H.T 
    #eliminating the scale factor and estimating R & T
    K_inv = np.linalg.inv(K)
    lamda = np.linalg.norm(K_inv.dot(h1),ord =2 )
    r1 = lamda*K_inv.dot(h1)
    r2 = lamda*K_inv.dot(h2)
    r3 = np.cross(r1,r2)
    t = lamda*K_inv.dot(h3)
    
    return np.stack((r1,r2,r3,t), axis=1)

