from ssa import ssa_2d

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('lena.bmp')

plt.figure()
plt.imshow(img)
plt.title("Original image.")

window_height = 20
window_width = 20
number_of_eigenvectors = 10

img_approx = ssa_2d(img = img, u = window_height , v = window_width, l = number_of_eigenvectors, verbose = 1)

plt.figure()
plt.imshow(img_approx)
plt.title("Reconstructed image. L = {}".format(number_of_eigenvectors))

plt.show()