from ssa import ssa_2d
import utils

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np

#img = mpimg.imread('hay_bales.bmp')
img = mpimg.imread('person.bmp')
img = utils.scale_img(img)


# first row

img_noise = utils.add_gaussian_noise(img, noise_variance=0.01)
img_noise = utils.scale_img(img_noise)

window_height = 10
window_width = 10


plt.figure()
plt.subplot(241)
plt.imshow(img_noise, cmap = "gray")
plt.title("Noisy image.")
plt.axis("off")


number_of_eigenvectors = 2
#number_of_eigenvectors_rec = 17

img_reconstructed = ssa_2d(img=img_noise,
                           u=window_height, v=window_width,
                           l=number_of_eigenvectors,
                           #l_rec = number_of_eigenvectors_rec,
                           verbose=0)

img_reconstructed = utils.scale_img(img_reconstructed)

plt.subplot(242)
plt.imshow(img_reconstructed, cmap = "gray")
plt.title("Reconstructed with {} eigenvectors.".format(number_of_eigenvectors))
plt.axis("off")


number_of_eigenvectors = 15
#number_of_eigenvectors_rec = 17

img_reconstructed = ssa_2d(img=img_noise,
                           u=window_height, v=window_width,
                           l=number_of_eigenvectors,
                           #l_rec = number_of_eigenvectors_rec,
                           verbose=0)

img_reconstructed = utils.scale_img(img_reconstructed)

plt.subplot(243)
plt.imshow(img_reconstructed, cmap = "gray")
plt.title("Reconstructed with {} eigenvectors.".format(number_of_eigenvectors))
plt.axis("off")


number_of_eigenvectors = 30
#number_of_eigenvectors_rec = 17

img_reconstructed = ssa_2d(img=img_noise,
                           u=window_height, v=window_width,
                           l=number_of_eigenvectors,
                           #l_rec = number_of_eigenvectors_rec,
                           verbose=0)

img_reconstructed = utils.scale_img(img_reconstructed)

plt.subplot(244)
plt.imshow(img_reconstructed, cmap = "gray")
plt.title("Reconstructed with {} eigenvectors.".format(number_of_eigenvectors))
plt.axis("off")




#  second row

img_noise = utils.add_gaussian_noise(img, noise_variance=0.1)
img_noise = utils.scale_img(img_noise)

window_height = 10
window_width = 10


plt.subplot(245)
plt.imshow(img_noise, cmap = "gray")
plt.title("Noisy image.")
plt.axis("off")


number_of_eigenvectors = 2
#number_of_eigenvectors_rec = 17

img_reconstructed = ssa_2d(img=img_noise,
                           u=window_height, v=window_width,
                           l=number_of_eigenvectors,
                           #l_rec = number_of_eigenvectors_rec,
                           verbose=0)

img_reconstructed = utils.scale_img(img_reconstructed)

plt.subplot(246)
plt.imshow(img_reconstructed, cmap = "gray")
plt.title("Reconstructed with {} eigenvectors.".format(number_of_eigenvectors))
plt.axis("off")


number_of_eigenvectors = 15
#number_of_eigenvectors_rec = 17

img_reconstructed = ssa_2d(img=img_noise,
                           u=window_height, v=window_width,
                           l=number_of_eigenvectors,
                           #l_rec = number_of_eigenvectors_rec,
                           verbose=0)

img_reconstructed = utils.scale_img(img_reconstructed)

plt.subplot(247)
plt.imshow(img_reconstructed, cmap = "gray")
plt.title("Reconstructed with {} eigenvectors.".format(number_of_eigenvectors))
plt.axis("off")


number_of_eigenvectors = 30
#number_of_eigenvectors_rec = 17

img_reconstructed = ssa_2d(img=img_noise,
                           u=window_height, v=window_width,
                           l=number_of_eigenvectors,
                           #l_rec = number_of_eigenvectors_rec,
                           verbose=0)

img_reconstructed = utils.scale_img(img_reconstructed)

plt.subplot(248)
plt.imshow(img_reconstructed, cmap = "gray")
plt.title("Reconstructed with {} eigenvectors.".format(number_of_eigenvectors))
plt.axis("off")







plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

plt.show()

