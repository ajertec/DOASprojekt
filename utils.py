import matplotlib.pyplot as plt
import numpy as np
import ssa
from skimage.measure import shannon_entropy
from skimage.util import random_noise


def plot_reconstruct_eigenvectors_img(U, Vh, X, params, verbose):

    """
    Plots image reconstruction for each eigenvector.
    If verbose == 3, plots w-correlation matrix.
    """

    h, w, u, v, l = params["h"], params["w"], params["u"], params["v"], params["l"]

    print("Computing img reconstruction for each eigenvector ...")

    fig = plt.figure()

    # for subplots grid
    n = np.round(np.sqrt(l))
    m = np.ceil(l/n)
    assert(n*m >= l)
    
    img_division = np.zeros((h,w))
    if verbose == 3: 
        img_approx_list = []

    first_pass = True
    for e in range(l):
        print("    Eigenvector: ", e+1, " ...")

        X_approx = np.dot(np.dot(U[:, e:e+1],Vh[e:e+1, :]), X)

        img_approx = np.zeros((h,w))

        # image reconstruction from trajectory matrix
        k = 0
        for i in range(h-u+1):
            for j in range(w-v+1):
                img_approx[i:i+u, j:j+v] += np.reshape(X_approx[:, k], (u, v))
                if first_pass: 
                    img_division[i:i+u, j:j+v] += 1 # no need in computing this matrix for each img_approx
                k += 1

        if verbose == 3: 
            img_approx_list.append(img_approx/img_division)

        first_pass = False

        # plotting reconstructed eigenvecs
        a = fig.add_subplot(n, m, e+1)
        plt.imshow(img_approx/img_division, cmap = "gray")
        a.set_title('ev {}'.format(e+1))
        a.axis("off")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    w_correlation_matrix = np.eye(l)
    if verbose == 3:
        for i in range(l):
            for j in range(i+1, l):
                w_correlation_matrix[i, j] = compute_weighted_correlation(img_approx_list[i], img_approx_list[j])
                w_correlation_matrix[j, i] = w_correlation_matrix[i, j]
        
        plt.matshow(w_correlation_matrix, cmap = "gray_r")
        plt.colorbar()
        plt.title("w-correlation matrix")


def compute_weighted_correlation(F1, F2):
    return np.sum(F1*F2) / (np.linalg.norm(F1, ord = "fro")*np.linalg.norm(F2, ord = "fro"))

def scale_img(img):
    """Scale image to interval 0 to 1."""
    return (img - np.min(img)) / (np.max(img)-np.min(img))


def add_gaussian_noise(img, noise_variance):
    return random_noise(img, mode = "gaussian", mean = 0, var = noise_variance, clip = True)


def add_salt_and_pepper_noise(img, amount):
    return random_noise(img, mode = "s&p", amount = amount, clip = True)


def add_speckle_noise(img, noise_variance):
    return random_noise(img, mode = "speckle", mean = 0, var = noise_variance, clip = True)

def add_periodic_noise(img):
    #TODO
    pass


def add_salt_and_pepper_noise_custom(img, perc):
    """
    Add S&P noise to img.
    
    # Parameters:
        img: numpy.ndarray
            2D grayscale image.
        perc: int 
            Percentage of image covered in S&P noise, 1 to 100.
    """

    if type(perc) != int:
        raise Exception("Parameter 'perc' must be integer.")
    if perc <= 0 or perc > 100:
        raise Exception("Valid values for parameter 'perc' are from 0 to 100.")

    noise_mask = np.random.randint(low = 256, size = img.shape)
    prob_den = 255*perc/100
    img[noise_mask < prob_den/2] = 0.
    img[(noise_mask > prob_den/2) * (noise_mask < prob_den)] = 1.

    return img
    
def plot_reconstr_eigenvectors(U, Vh, X, params):

    l = params["l"]

    print("Computing img reconstruction for each eigenvector ...")
    fig = plt.figure()

    # for subplots grid
    n = np.round(np.sqrt(l))
    m = np.ceil(l/n)
    assert(n*m >= l)

    for i in range(l):
        print("    Eigenvector: ", i+1, " ...")
        X_approx = np.dot(np.dot(U[:, i:i+1],Vh[i:i+1, :]), X)
        img_reconstr = ssa.reconstruct_img(X_approx, params)

        a = fig.add_subplot(n, m, i+1)
        plt.imshow(img_reconstr, cmap = "gray")
        a.set_title('l = {}'.format(i+1))
        #plt.colorbar()
        #plt.tight_layout()