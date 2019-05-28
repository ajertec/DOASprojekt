import numpy as np
import matplotlib.pyplot as plt
from numba import jit

def ssa_2d(img, u, v, l, verbose=False):
    """
    2D-SSA as described in paper:
    http://ssa.cf.ac.uk/zhigljavsky/pdfs/SSA/Rodriguez-Aragon.pdf
    
    # Parameters:
        img: grayscale image, 2d numpy array
        u: windows height, integer
        v: window width, integer
        l: number of eigenvectors used in image reconstruction, integer
        verbose: boolean, verbosity mode. False = silent, True = prints used parameters and plots eigenvalues

    # Returns:
        reconstructed grayscale image, 2d numpy array

    """

    h, w = img.shape

    if type(u) != int or type(v) != int or type(l) != int:
        raise Exception("Window size parameters must be integer.")

    p = u*v
    q = (h-u+1)*(w-v+1)

    if verbose:
        print("Img shape: ", (h, w))
        print("Window size: ", (u, v))
        print("Trajectory matrix shape: ", (p, q))
        print("Number of eigenvectors used for reconstruction: ", l)

    if p>q:
        raise Exception ("Algorithm assumes p<g, choose another window parameters (u, v).")
    if p>1000:
        raise Exception ("Choose smaller window, current parameters computationally expensive.")

    
    # 1) formation of TRAJECTORY MATRIX
    X = compute_trajectory_matrix(img, u, v)
    
    # 2) SVD of matrix X*X.T
    U, S, Vh = np.linalg.svd(np.dot(X, X.T))

    if verbose:
        plt.figure()
        plt.plot(S)
        plt.title("Eigenvalues.")

    # 3) selection of eigenvectors
    # 4) reconstruction of image
    X_approx = np.dot(np.dot(U[:, :l],Vh[:l, :]), X)

    return reconstruct_img(X_approx, h, w, u, v)


@jit
def compute_trajectory_matrix(img, u, v):
    h, w = img.shape
    trajectory_matrix = np.zeros((u*v, (h-u+1)*(w-v+1))) # shape (p, q)
    k = 0
    for i in range(h-u+1):
        for j in range(w-v+1):
            trajectory_matrix[:, k] = img[i:i+u, j:j+v].flatten()
            k += 1
    return trajectory_matrix

@jit
def reconstruct_img(X_approx, h, w, u, v):
    img_approx = np.zeros((h,w))
    img_division = np.zeros((h,w))

    # image reconstruction from trajectory matrix
    k = 0
    for i in range(h-u+1):
        for j in range(w-v+1):
            img_approx[i:i+u, j:j+v] += np.reshape(X_approx[:, k], (u, v))
            img_division[i:i+u, j:j+v] += 1
            k += 1
    return img_approx/img_division