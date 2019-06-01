from ssa import ssa_2d

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim

import utils


img = mpimg.imread('lena.bmp')
img = utils.scale_img(img)

img_noise = utils.add_gaussian_noise(img, noise_variance=0.2)
img_noise = utils.scale_img(img_noise)

window_height = 10
window_width = 15
number_of_eigenvectors = 25
#number_of_eigenvectors_rec = 17

img_reconstructed = ssa_2d(img=img_noise,
                           u=window_height, v=window_width,
                           l=number_of_eigenvectors,
                           #l_rec = number_of_eigenvectors_rec,
                           verbose=3)

img_reconstructed = utils.scale_img(img_reconstructed)

plt.figure()
plt.subplot(131)
plt.imshow(img, cmap = "gray")
plt.title("Original image.")

plt.subplot(132)
plt.imshow(img_noise, cmap = "gray")
plt.title("Noisy image.")

try:
    eigenvec_rec = number_of_eigenvectors_rec
except NameError:
    eigenvec_rec = number_of_eigenvectors

plt.subplot(133)
plt.imshow(img_reconstructed, cmap = "gray")
plt.title("Reconstructed image. First {} eigenvectors.".format(eigenvec_rec))

print("PSNR (origigi vs reconstr): ", psnr(img, img_reconstructed))
print("PSNR (origigi vs noise): ", psnr(img, img_noise))

print("SSIM (origigi vs reconstr): ", ssim(img, img_reconstructed))
print("SSIM (origigi vs noise): ", ssim(img, img_noise))

print("Frobenius norm (origigi vs reconstr): ",
      np.linalg.norm(img - img_reconstructed, ord="fro"))
print("Frobenius norm  (origigi vs noise): ",
      np.linalg.norm(img - img_noise, ord="fro"))

plt.show()
