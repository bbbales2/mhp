#%%
import numpy
import skimage.io

images = ['t0_coarsening.png', 't2_coarsening.png', 'te_rafting.png', 't0_rafting.png', 'te_columnar.png', 't9_coarsening.png']
x, y = 375, 118
dx = 732

for filename in images:
    im = skimage.io.imread('/home/bbales2/microstructure_hog_paper/images/3d_rotated/{0}'.format(filename), as_grey = True)

    im = im[y : y + dx, x : x + dx]

    skimage.io.imsave('/home/bbales2/microstructure_hog_paper/images/3d_rotated/{0}.cropped.png'.format(filename), im)
#%%