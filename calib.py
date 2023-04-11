from utils import *
from computation import *
import glob



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

    


if __name__ == "__main__":
    main()