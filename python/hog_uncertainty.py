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
for path in [#'images/r5_0pct.png',
             #'images/r5_2pct.png']:
'images/molybdenum1.png',
             'images/molybdenum3.png']:
             #
             #'images/rene88_2.png',
             #'images/renen4_2.png'
    ims2.append(rescale(skimage.io.imread(path, as_grey = True), 0.0, 1.0))

for im in ims2:
    plt.imshow(im)
    plt.show()

#%%
reload(hog)

translations = [50, 100]
for offset in translations:
    histograms = []
    histograms2 = []

    filtrd_ims = []

    for i, im_ in enumerate(ims2):
        im = numpy.pad(im_, ((0, 0), (0, offset)), 'constant')

        for r in range(im.shape[0]):
            im[r] = numpy.interp(numpy.linspace(0, im.shape[1], im.shape[1]) - offset * float(r) / im.shape[0], numpy.linspace(0, im.shape[1], im.shape[1]), im[r])

        im = im[:, offset:-offset]

        filtrd = scipy.ndimage.filters.gaussian_filter(im, 2.0)

        plt.imshow(filtrd, cmap = plt.cm.gray, interpolation = 'NONE')
        fig = plt.gcf()
        fig.set_size_inches(6., 4.)
        plt.tight_layout()
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        #plt.savefig('{0}/images/rene_n5.{1}.blurred.png'.format(folder, i), format='png', dpi = 150, bbox_inches='tight', pad_inches=0)
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

        c = numpy.abs(numpy.fft.fft(fft))

        cs.append(c)

        histograms.append([angles, histogram])

    #for im, (angles, histogram) in zip(ims2, histograms[:]):
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

        #plt.plot(numpy.abs(fft), 'x', markeredgewidth = 1)
        #plt.title('FFT')
        #plt.xlabel('Frequency bin', fontsize = 20)
        #plt.ylabel('|FFT|', fontsize = 20)
        #plt.gca().tick_params(axis='both', which='major', labelsize=16)
        #fig = plt.gcf()
        #fig.set_size_inches(15, 15)
        #plt.plot(numpy.abs(fft))
        #print numpy.abs(fft)[0:10]
