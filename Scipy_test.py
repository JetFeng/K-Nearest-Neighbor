import numpy as np
from scipy.misc import imread,imsave,imresize
import matplotlib.pyplot as plt

#Read an JPEG image to a numpy array

img = imread("img/cat.jpg");
print img.dtype,img.shape

# show the original image
plt.subplot(1,2,1)
plt.imshow(img)

img_tinted = img * [1,0.95,0.9]

#show the tinted image
plt.subplot(1,2,2)

plt.imshow(np.uint8(img_tinted))
plt.show()

# img_tinted = imresize(img_tinted,(1200,1200))
#
# imsave("img/cat_tinted.jpg" , img_tinted)