import numpy as np
import matplotlib.pyplot as plt
from numba import jit

import utils


def ssa_2d(img, u, v, l, l_rec=None, verbose=False):
    """
    2D-SSA as described in paper:
    http://ssa.cf.ac.uk/zhigljavsky/pdfs/SSA/Rodriguez-Aragon.pdf

    # Parameters:
        img: numpy.ndarray
            2D Grayscale image.
        u: int
            Window height.
        v: int
            Window width.
        l: int
            Number of computed eigenvectors.
        l_rec: int
            Number of eigenvectors used in image reconstruction.
        verbose: int
            Verbosity mode. 
            0 = silent, 
            1 = prints used parameters and plots eigenvalues, 
            2 = 1 + plots img reconstruction for each eigenvector,
            3 = 2 + plots w-correlation matrix.

    # Returns:
        numpy.ndarray
        Reconstructed grayscale image.

    """

    if len(img.shape) != 2:
        raise Exception("Function accepts only grayscale 2D image.")

    if l_rec:
        if type(l_rec) != int:
            raise Exception("Parameter 'l_rec' must be integer.")
        if l_rec > l:
            raise Exception("Parameter 'l_rec' must be smaller than 'l'.")

    if l_rec == None:
        l_rec = l

    if type(u) != int or type(v) != int or type(l) != int:
        raise Exception(
            "Window size parameters 'u', 'v' and number of eigenvectors 'l' must be integers.")

    h, w = img.shape

    p = u*v
    q = (h-u+1)*(w-v+1)

    if l > p:
        raise Exception(
            "Maximum number of eigenvalues for current window parameters is {}. Choose smaller parameter 'l'.".format(p))

    params = {
        "h": h,
        "w": w,
        "u": u,
        "v": v,
        "p": p,
        "q": q,
        "l": l,
        "l_rec": l_rec
    }

    if verbose:
        print("Img shape: ", (h, w))
        print("Window size: ", (u, v))
        print("Trajectory matrix shape: ", (p, q))
        print("Number of eigenvectors: ", l)
        print("Number of eigenvectors used for reconstruction: ", l_rec)

    if p > q:
        raise Exception(
            "Algorithm assumes p<g, choose another window parameters 'u' and 'v'.")
    if p > 1000:
        raise Exception(
            "Choose smaller window, current parameters computationally expensive.")

    # 1) formation of TRAJECTORY MATRIX
    if verbose:
        print("Computing trajectory matrix ...")
    X = compute_trajectory_matrix(img, params)

    # 2) SVD of matrix X*X.T
    if verbose:
        print("Computing SVD ...")
    U, S, Vh = svd(np.dot(X, X.T))

    # if verbose:
        # plt.figure()
        # plt.subplot(121)
        # plt.semilogy(S)
        # plt.title("All Eigenvalues.")
        # plt.ylabel("log")

        # plt.subplot(122)
        # plt.stem(S[1:l])
        # plt.title("First {} Eigenvalues (w/o 1.).".format(l))

    # 3) selection of eigenvectors
    # 4) reconstruction of image
    if verbose:
        print("Computing approx trajectory matrix ...")
    X_approx = np.dot(np.dot(U[:, :l_rec], Vh[:l_rec, :]), X)

    if verbose >= 2:
        utils.plot_reconstruct_eigenvectors_img(U, Vh, X, params, verbose)

    if verbose:
        print("Reconstructing image ...")
    return reconstruct_img(X_approx, params)



def compute_trajectory_matrix(img, params):
    u, v = params["u"], params["v"]
    h, w = img.shape
    trajectory_matrix = np.zeros((u*v, (h-u+1)*(w-v+1)))  # shape (p, q)
    k = 0
    for i in range(h-u+1):
        for j in range(w-v+1):
            trajectory_matrix[:, k] = img[i:i+u, j:j+v].flatten()
            k += 1
    return trajectory_matrix


# no need for jit
def reconstruct_img(X_approx, params):
    h, w, u, v = params["h"], params["w"], params["u"], params["v"]
    img_approx = np.zeros((h, w))
    img_division = np.zeros((h, w))

    # image reconstruction from trajectory matrix
    k = 0
    for i in range(h-u+1):
        for j in range(w-v+1):
            img_approx[i:i+u, j:j+v] += np.reshape(X_approx[:, k], (u, v))
            img_division[i:i+u, j:j+v] += 1
            k += 1
    return img_approx/img_division



def svd(X):
    return np.linalg.svd(X)
