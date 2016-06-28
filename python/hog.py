#%%
import matplotlib.pyplot as plt
import numpy
import os
import itertools
import math
import scipy.integrate
import mahotas
import collections
import skimage.measure, skimage.io, skimage.feature, skimage.util, skimage.filters
import matplotlib.patches
from util import rescale
import hog

folder = '/home/bbales2/microstructure_hog_paper/'

os.chdir(folder)

#%%

#             'images/r5_5pct.png',
#             'images/r5_rupture.png'
#
ims2 = []
for path in ['images/r5_0pct.png',
             'images/r5_2pct.png']:
#'images/molybdenum1.png',
#             'images/molybdenum3.png'
#]:
             #
             #'images/rene88_2.png',
             #'images/renen4_2.png'
    ims2.append(rescale(skimage.io.imread(path, as_grey = True), 0.0, 1.0))

for im in ims2:
    plt.imshow(im)
    plt.show()

#%%
reload(hog)

histograms = []
histograms2 = []

filtrd_ims = []

for i, im in enumerate(ims2):
    filtrd = scipy.ndimage.filters.gaussian_filter(im, 2.0)

    plt.imshow(filtrd, cmap = plt.cm.gray, interpolation = 'NONE')
    fig = plt.gcf()
    fig.set_size_inches(6., 4.)
    plt.tight_layout()
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.savefig('{0}/images/rene_n5.{1}.blurred.png'.format(folder, i), format='png', dpi = 150, bbox_inches='tight', pad_inches=0)
    plt.show()

    filtrd_ims.append(filtrd)

ffts = []
cs = []
styles = ['k-', 'k--']
for i, filtrd in enumerate(filtrd_ims[:]):
    dys, dxs = numpy.gradient(filtrd)
    dys = -dys

    angles, histogram = hog.build2dHist(dys, dxs)

    angles2 = numpy.concatenate((angles, [angles[0]]))
    histogram2 = numpy.concatenate((histogram, [histogram[0]]))

    #histogram2 /= numpy.linalg.norm(histogram2)

    #histogram2 -= histogram2.mean()

    fft = numpy.log(numpy.abs(numpy.fft.fft(histogram2)))#

    ffts.append(fft)
    print fft[0:8]

    #plt.plot(fft)
    #plt.show()

    c = numpy.abs(numpy.fft.fft(fft))

    cs.append(c)
    #plt.plot(c)
    #plt.show()

    print c[0:8]

    #fig = plt.gcf()
    #fig.set_size_inches(6., 4.)
    #plt.tight_layout()
    #ax = plt.gca()
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
    #plt.show()

    histograms.append([angles, histogram])
    histograms2.append([angles2, histogram2])
#%%
import matplotlib_scalebar.scalebar

fig = plt.figure()

ax1 = plt.subplot(222)
ax2 = plt.subplot(224)
ax3 = plt.subplot(121, projection = 'polar')

ax3.plot(histograms2[0][0], histograms2[0][1], styles[0])
ax3.plot(histograms2[1][0], histograms2[1][1], styles[1])

ax1.imshow(filtrd_ims[0], cmap = plt.cm.gray, interpolation = 'NONE')
ax2.imshow(filtrd_ims[1], cmap = plt.cm.gray, interpolation = 'NONE')

ax1.get_xaxis().set_visible(False)
ax1.get_yaxis().set_visible(False)

ax2.get_xaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)

ax3.set_yticklabels([])

ffts = numpy.array(ffts)
cs = numpy.array(cs)

scale1 = matplotlib_scalebar.scalebar.ScaleBar(0.029411765e-6)
scale2 = matplotlib_scalebar.scalebar.ScaleBar(0.029585799e-6)
ax1.add_artist(scale1)
ax2.add_artist(scale2)
ax3.legend(['Base (Top)', 'Rafted (Bottom)'], loc = [0.60, 0.95])
fig.set_size_inches((6.5, 4.5))
plt.savefig('{0}/images/renehog.png'.format(folder, i), format='png', dpi = 150, bbox_inches='tight', pad_inches = 0.1)
plt.show()
#%%
scale1 = matplotlib_scalebar.scalebar.ScaleBar(0.713383066e-9)
scale2 = matplotlib_scalebar.scalebar.ScaleBar(0.726487743e-9)
ax1.add_artist(scale1)
ax2.add_artist(scale2)
ax3.legend(['Square (Top)', 'Circular (Bottom)'], loc = [0.75, 0.90])
fig.set_size_inches((8, 4.5))
plt.savefig('{0}/images/mollyhog.png'.format(folder, i), format='png', dpi = 150, bbox_inches='tight', pad_inches = 0.1)
plt.show()
#%%
#%%

for im, (angles, histogram) in zip(ims2, histograms[:]):
    h = histogram# - numpy.min(histogram)
    h /= numpy.max(h)
    #h -= numpy.mean(h)

    fft = numpy.fft.fft(h)

    fft = fft[0:(len(fft) + 1) / 2]
    fft[-1] *= 0.5

    r = numpy.real(fft[4::4])
    c = numpy.imag(fft[4::4])

    tmp = numpy.array(zip(r, c))
    n = numpy.array((r[0], c[0]))
    n /= numpy.linalg.norm(n)
    mags = tmp#n.dot(tmp.transpose())
    posc4 = numpy.sum(mags**2)

    r = numpy.real(fft[2::2])
    c = numpy.imag(fft[2::2])

    tmp = numpy.array(zip(r, c))
    n = numpy.array((r[0], c[0]))
    n /= numpy.linalg.norm(n)
    mags = tmp#n.dot(tmp.transpose())
    posc2 = numpy.sum(mags**2)

    posct = numpy.sum(numpy.abs(fft[1:])**2)
    ptotal = numpy.sum(numpy.abs(fft[:])**2)

    #print (numpy.abs(fft)**2)#[1::4]
    print posc4 / posct, posc2 / posct, numpy.abs(fft[0]**2) / posct

    plt.plot(numpy.abs(fft), 'x', markeredgewidth = 1)
    plt.title('FFT')
    plt.xlabel('Frequency bin', fontsize = 20)
    plt.ylabel('|FFT|', fontsize = 20)
    plt.gca().tick_params(axis='both', which='major', labelsize=16)
    #fig = plt.gcf()
    #fig.set_size_inches(15, 15)
    #plt.plot(numpy.abs(fft))
    print numpy.abs(fft)[0:10]
#%%
    plt.show()
    plt.imshow(im, cmap = plt.cm.gray)
    ax = plt.gca()
    ax.get_yaxis().set_ticks([])
    ax.get_xaxis().set_ticks([])
    plt.show()
    degrees = 180 * angles / numpy.pi
    angles = list(angles)
    angles.append(angles[-1] + angles[1] - angles[0])
    histogram = list(histogram)
    histogram.append(histogram[0])
    degrees = list(degrees)
    degrees.append(degrees[-1] + degrees[1] - degrees[0])
    plt.plot(degrees, histogram, linewidth = 1)
    plt.ylim((0, plt.ylim()[1]))
    plt.xlabel('Theta (Degrees)', fontsize = 20)
    plt.ylabel('HoG intensity', fontsize = 20)
    plt.gca().tick_params(axis='both', which='major', labelsize=16)
    #ax = plt.gca()
    #ax.get_yaxis().set_ticks([])
    plt.show()
    plt.polar(angles, histogram, linewidth = 1)
    plt.xlabel('Theta (Degrees)', fontsize = 20)
    plt.ylabel('HoG intensity (r)', fontsize = 20)
    plt.gca().tick_params(axis='both', which='major', labelsize=16)
    plt.show()