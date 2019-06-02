from ssa import ssa_2d

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim

import utils

img = mpimg.imread('hay_bales.bmp')
#img = mpimg.imread('person.bmp')
img = utils.scale_img(img)

img_noise = utils.add_gaussian_noise(img, noise_variance=0.1)
img_noise = utils.scale_img(img_noise)

window_height = 5
window_width = 5
total_number_of_eigenvectors = window_height * window_width
#number_of_eigenvectors_rec = 17

psnr_line = psnr(img, img_noise)
ssim_line = ssim(img, img_noise)
frobenius_line = np.linalg.norm(img - img_noise, ord="fro")

psnr_list = []
ssim_list = []
frobenius_list = []


# calculate the metrics for different number of eigenvectors
for number_of_eigenvectors in range(2, total_number_of_eigenvectors):

    img_reconstructed = ssa_2d(img=img_noise,
                               u=window_height, v=window_width,
                               l=number_of_eigenvectors,
                               #l_rec = number_of_eigenvectors_rec,
                               verbose=1)

    img_reconstructed = utils.scale_img(img_reconstructed)
    
    psnr_list.append(psnr(img, img_reconstructed))
    ssim_list.append(ssim(img, img_reconstructed))
    frobenius_list.append(np.linalg.norm(img - img_reconstructed, ord="fro"))
    
    
# plot metrics
plt.figure()
plt.subplot(131)
plt.plot(psnr_list, 'ro')
plt.plot([psnr_line] * len(psnr_list))
plt.title("PSNR with {}x{} window.".format(window_width, window_height))

plt.subplot(132)
plt.plot(ssim_list, 'ro')
plt.plot([ssim_line] * len(ssim_list))
plt.title("SSIM with {}x{} window.".format(window_width, window_height))

plt.subplot(133)
plt.plot(frobenius_list, 'ro')
plt.plot([frobenius_line] * len(frobenius_list))
plt.title("Frobenius norm with {}x{} window.".format(window_width, window_height))

plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.05, wspace=0.1, hspace=0.15)


plt.figure()
plt.subplot(311)
plt.plot(psnr_list, 'ro')
plt.plot([psnr_line] * len(psnr_list))
plt.title("PSNR with {}x{} window.".format(window_width, window_height))

plt.subplot(312)
plt.plot(ssim_list, 'ro')
plt.plot([ssim_line] * len(ssim_list))
plt.title("SSIM with {}x{} window.".format(window_width, window_height))

plt.subplot(313)
plt.plot(frobenius_list, 'ro')
plt.plot([frobenius_line] * len(frobenius_list))
plt.title("Frobenius norm with {}x{} window.".format(window_width, window_height))

plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.05, wspace=0.1, hspace=0.15)


plt.show()

