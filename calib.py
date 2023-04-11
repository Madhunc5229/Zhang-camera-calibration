from utils import *
from computation import *
import glob
from scipy import optimize


def main():

    data_path = 'Calibration_Imgs/'
    img_paths = sorted(glob.glob(data_path+'*.jpg'))
    
    M_pts, m_pts, H_matrices = list(), list(), list()
    f = open("results/results.txt", "w")
    for img_path in img_paths:
        M, m, H = getCorrespondces(img_path)
        M_pts.append(M)
        m_pts.append(m)
        H_matrices.append(H)
    K_init = getIntrinsicMat(H_matrices)
    f.write('Initially estimated K matrix : \n '+ str(np.matrix.round(K_init,3))+ '\n\n')

    E_init = list()
    for h in H_matrices:
        RT = getExtrinsicMat(h,K_init)
        E_init.append(RT)

    kC = (0,0)
    f.write('Distortion Coordinates before optimization: '+ str(np.round(kC,5))+ '\n')
    reprojection_error = estimateReprojectionError(K_init, kC, (M_pts, m_pts, E_init))
    print("Projection error before optimization, : " + str(reprojection_error))
    f.write('Projection error before optimization : '+ str(np.round(reprojection_error,5))+ '\n\n\n')

    alpha, beta, gamma = K_init[0, 0], K_init[1, 1], K_init[0, 1]
    u0,v0 = K_init[0, 2], K_init[1, 2]
    k1,k2 = 0, 0

    init_params = [alpha, beta, gamma, u0, v0, k1, k2]

    out = optimize.least_squares(fun = loss, x0 = init_params, method="lm", args = [M_pts, m_pts, E_init])
    optimal_params = out.x

    kC = (optimal_params[-2], optimal_params[-1])
    K = K_matrix(optimal_params)
    f.write('optimized K matrix : \n '+ str(np.matrix.round(K,3))+ '\n\n')


if __name__ == "__main__":
    main()