#%%
import matplotlib.pyplot as plt
import skimage.io
import skimage.filters
import os
import mahotas
import numpy

# Get path to current script
folder = '/home/bbales2/microstructure_hog_paper/'

imr4 = skimage.io.imread('{0}/images/renen4s.png'.format(folder), as_grey = True)
im88 = imr4#skimage.io.imread('{0}/../static/images/rene88.png'.format(folder), as_grey = True)

plt.figure(figsize = (6,4))
plt.imshow(imr4, cmap = plt.cm.gray, interpolation = 'None')
plt.tight_layout()
ax = plt.gca()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.savefig('{0}/images/renen4plot.png'.format(folder), format='png', dpi = 150, bbox_inches='tight', pad_inches=0)
#plt.show()

plt.figure(figsize = (6,4))
plt.imshow(im88, cmap = plt.cm.gray, interpolation = 'None')
plt.tight_layout()
ax = plt.gca()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.savefig('{0}/images/rene88plot.png'.format(folder), format='png', dpi = 150, bbox_inches='tight', pad_inches=0)


fimr4 = skimage.filters.gaussian_filter(imr4.astype('float'), 1.0)
fim88 = skimage.filters.gaussian_filter(im88.astype('float'), 2.0)

tr4 = skimage.filters.threshold_otsu(fimr4)
t88 = skimage.filters.threshold_otsu(fim88)

plt.figure(figsize = (6,4))
plt.imshow(fimr4, cmap = plt.cm.gray, interpolation = 'None')
plt.tight_layout()
ax = plt.gca()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.savefig('{0}/images/renen4blur.png'.format(folder), format='png', dpi = 150, bbox_inches='tight', pad_inches=0)
#plt.show()

plt.figure(figsize = (6,4))
plt.imshow(fim88, cmap = plt.cm.gray, interpolation = 'None')
plt.tight_layout()
ax = plt.gca()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.savefig('{0}/images/rene88blur.png'.format(folder), format='png', dpi = 150, bbox_inches='tight', pad_inches=0)

#Show
plt.figure(figsize = (6,4))
plt.imshow(fimr4 > tr4, cmap = plt.cm.gray, interpolation = 'None')
plt.tight_layout()
ax = plt.gca()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.savefig('{0}/images/renen4otsu.png'.format(folder), format='png', dpi = 150, bbox_inches='tight', pad_inches=0)
#plt.show()

plt.figure(figsize = (6,4))
plt.imshow(fim88 > t88, cmap = plt.cm.gray, interpolation = 'None')
plt.tight_layout()
ax = plt.gca()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.savefig('{0}/images/rene88otsu.png'.format(folder), format='png', dpi = 150, bbox_inches='tight', pad_inches=0)
#plt.show()

mark = mahotas.labeled.borders((fimr4 >= tr4))
plt.figure(figsize = (6,4))
plt.imshow(imr4, interpolation = 'None', cmap = plt.cm.gray)
plt.imshow(numpy.ma.masked_where(mark < 0.1, mark), interpolation = 'None', cmap = plt.cm.gray)
plt.tight_layout()
ax = plt.gca()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.savefig('{0}/images/renen4otsuBorders_noblur.png'.format(folder), format='png', dpi = 150, bbox_inches='tight', pad_inches=0)

#%%
print numpy.mean(imr4[:, :imr4.shape[1] / 2])
print numpy.mean(imr4[:, imr4.shape[1] / 2:])
print tr4

#tr4 = skimage.filters.threshold_otsu(imr4)
#t88 = skimage.filters.threshold_otsu(im88)
#Write thresholded versions
