#%%
import util
reload(util)
import matplotlib.pyplot as plt
import numpy
import os
import itertools
import math
import scipy.integrate
import mahotas
import collections
import skimage.measure, skimage.io, skimage.util, skimage.transform, skimage.filters
import matplotlib.patches
from util import build_rot
import h5py
from util import saveStack
from util import rescale
from util import detect_local_minima
import spharm
import hog
import faehrmann
import pyshtools
reload(faehrmann)

os.chdir('/home/bbales2/microstructure_hog_paper/')

#%%
import faehrmann
from faehrmann.spectral import buildReferenceHistograms, toMatrix
reload(faehrmann.spectral)
from faehrmann.spectral import buildReferenceHistograms, toMatrix
TN = 10
PN = 20

lookup, lookupAngles = buildReferenceHistograms(TN, PN)
lookupSquare, lookupAnglesSquare = buildReferenceHistograms(TN, PN, [(90.0, 0.0), (90.0, 270.0), (90.0, 180.0), (90.0, 90.0)])
lookupLine, lookupAnglesLine = buildReferenceHistograms(TN, PN, [(0.0, 0.0), (180.0, 0.0)])
#print toMatrix(lookup[(5, 3)])[4, 177:184], toMatrix(lookup[(5, 1)])[4, 177:184]

#%%
reload(hog)

stacks = []
stacks_rotated = []
for stack in range(10):
    ims = skimage.io.imread_collection('/home/bbales2/rafting/rafting2ah5/images_{0}/*.png'.format(stack))#nrafting2a
    ims = ims.concatenate()

    #ims2 = []
    #for im in ims:
    #    ims2.append(skimage.transform.rescale(im, 3.0))

    #ims = numpy.array(ims2)

    ims = rescale(ims, 0.0, 1.0)

    filtrd = scipy.ndimage.filters.gaussian_filter(ims, 1.0)
    dzs, dys, dxs = numpy.gradient(filtrd)
    dys = -dys

    thetas, phis, histogram = hog.build3dHist(dxs, dys, dzs, 180, 360)

    for j in range(0, histogram.shape[0]):
        angle = j * (180.0 / float(histogram.shape[0])) * numpy.pi / 180.0
        histogram[j, :] /= -2 * numpy.pi * (numpy.cos(angle + (180.0 / float(histogram.shape[0])) * numpy.pi / 180.0) - numpy.cos(angle))#numpy.max(histogram[i, :])

    plt.imshow(histogram)
    plt.colorbar()
    plt.show()

    theta = 72 * numpy.pi / 180.0
    phi = 48 * numpy.pi / 180.0
    vectors = numpy.array([dxs, dys, dzs])

    vectors = numpy.rollaxis(vectors, 0, 3)
    #print phi
    vectors = build_rot(0.0, 0.0, -phi).dot(vectors)
    #print vectors.shape
    vectors = numpy.rollaxis(vectors, 0, 3)
    #print vectors.shape
    vectors = build_rot(0.0, -theta, 0).dot(vectors)
    #print vectors.shape
    dxs2, dys2, dzs2 = vectors
    #print dxs.shape
    thetas, phis, histogram2 = hog.build3dHist(dxs2, dys2, dzs2, 180, 360)

    for j in range(0, histogram2.shape[0]):
        angle = j * numpy.pi / float(histogram2.shape[0])
        factor = -2 * numpy.pi * (numpy.cos(angle + numpy.pi / float(histogram2.shape[0])) - numpy.cos(angle))
        histogram2[j, :] /= factor

    plt.imshow(histogram2)
    #plt.colorbar()
    ax = plt.gca()
    ax.get_yaxis().set_ticks([])
    ax.get_xaxis().set_ticks([])
    plt.show()

    stacks.append(histogram)
    stacks_rotated.append(histogram2)
#%%
for histogram2 in stacks_rotated:
    histogram = numpy.array(histogram2)

    #histogram[histogram < 0] = 0.0

    plt.imshow(histogram, cmap = plt.cm.gray)
    #plt.colorbar()
    ax = plt.gca()
    ax.get_yaxis().set_ticks([])
    ax.get_xaxis().set_ticks([])
    plt.show()

#%%'/home/bbales2/virt/border_removed/*.png'
#ims = skimage.io.imread_collection('/home/bbales2/rafting/rafting2ah5/rotated_images_0/*.png')
ims = skimage.io.imread_collection('/home/bbales2/virt/segmentation/UCSB_Rene88DT_FIBSS/BSEhigh/1/aligned/output3/*.png')
ims = ims.concatenate()

ims = rescale(ims, 0.0, 1.0)

dy = 13.333
dz = 20.0

ims2 = []
for i in range(ims.shape[0]):
    ims2.append(skimage.transform.rescale(ims[i], [0.5, 0.5]))

ims = numpy.array(ims2)
#%%
ims = skimage.io.imread_collection('/home/bbales2/virt/knot-gpu/tosegment/shrunk2/*.png')
ims = ims.concatenate()

ims = rescale(ims, 0.0, 1.0)

dy = 17.738342285
dz = 20.0

#print ims.shape
ims2 = []
for i in range(ims.shape[1]):
    ims2.append(skimage.transform.rescale(ims[:, i, :], [dz / dy, 1.0]))

ims2 = numpy.array(ims2)
ims = numpy.rollaxis(ims2, 0, 2)

print ims.shape
reload(hog)

filtrd = scipy.ndimage.filters.gaussian_filter(ims, 1.0)
dzs, dys, dxs = numpy.gradient(filtrd)
dys = -dys

thetas, phis, histogram = hog.build3dHist(dxs, dys, dzs, 180, 360)

for j in range(0, histogram.shape[0]):
    angle = j * (180.0 / float(histogram.shape[0])) * numpy.pi / 180.0
    histogram[j, :] /= -2 * numpy.pi * (numpy.cos(angle + (180.0 / float(histogram.shape[0])) * numpy.pi / 180.0) - numpy.cos(angle))#numpy.max(histogram[i, :])
#%%
plt.imshow(histogram / histogram.max(), cmap = plt.cm.gray, interpolation = 'NONE')
cbar = plt.colorbar(fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize = 14)
plt.xlabel('Longitude', fontsize = 20)
plt.ylabel('Latitude', fontsize = 20)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tick_params(axis='both', which='minor', labelsize=16)
fig = plt.gcf()
fig.set_size_inches(10., 5.5)
plt.savefig('images/3dhistogram.png', format='png', dpi = 150, bbox_inches='tight', pad_inches = 0.1)
plt.show()
#%%
weights = numpy.ones(stacks[0].shape)

for j in range(0, weights.shape[0]):
    angle = j * (180.0 / float(weights.shape[0])) * numpy.pi / 180.0
    weights[j, :] *= -2 * numpy.pi * (numpy.cos(angle + (180.0 / float(weights.shape[0])) * numpy.pi / 180.0) - numpy.cos(angle))#numpy.max(histogram[i, :])

#weights /= sum(weights.flatten())

hist, bins, patches = plt.hist(stacks[0].flatten(), weights = weights.flatten(), bins = 250)
plt.show()
plt.imshow(stacks[0])
plt.show()
#bincenters = 0.5 * (bin_edges[1:] + bin_edges[:-1])
#plt.bar(bincenters, hist, align='center')
#%%
dj = pyshtools.djpi2(90)
#%%
thetas, phis, Ntheta, Nphi = pyshtools.GLQGridCoord(histogram.shape[0] - 1)
phis = numpy.linspace(0, 360, histogram.shape[1], endpoint = False)

thetas, phis = numpy.meshgrid(90 - thetas, phis, indexing = 'ij')

xs = numpy.sin(numpy.pi * thetas / 180.0) * numpy.cos(numpy.pi * phis / 180.0)
ys = numpy.sin(numpy.pi * thetas / 180.0) * numpy.sin(numpy.pi * phis / 180.0)
zs = numpy.cos(numpy.pi * thetas / 180.0)

#ys = numpy.sin(numpy.pi * thetas / 180.0)
#zs = numpy.cos(numpy.pi * thetas / 180.0)

xx = pyshtools.SHExpandDH(ys * ys + zs * zs, sampling = 2)
yy = pyshtools.SHExpandDH(xs * xs + zs * zs, sampling = 2)
zz = pyshtools.SHExpandDH(xs * xs + ys * ys, sampling = 2)

xy = pyshtools.SHExpandDH(-xs * ys, sampling = 2)
yz = pyshtools.SHExpandDH(-ys * zs, sampling = 2)
xz = pyshtools.SHExpandDH(-xs * zs, sampling = 2)
#%%

histogram = stacks[0]
output = pyshtools.SHExpandDH(histogram, sampling = 2)

#%%
#realSpaces = []

def get75percentile(fm):
    T = sum(fm)
    fs = numpy.cumsum(fm)
    for i in range(len(fs)):
        if T * 0.75 < fs[i]:
            break

    alpha = (fs[i - 1] - T * 0.75) / (fs[i - 1] - fs[i])

    return (i - 1 + alpha) / len(fm)

def getEnergies(matrix):
    t = toMatrix(matrix)
    return numpy.sqrt(numpy.sum(t**2, axis = 1) / (2 * numpy.arange(t.shape[0]) + 1))

def sharpness(t):
    #h = cubicSymmetry(t)
    h = numpy.array(t)

    mass = getMass(h)

    h /= mass

    realSpace = pyshtools.MakeGridDH(h, sampling = 2)

    #print numpy.sum((realSpace * numpy.log2(realSpace)).flatten())
    #if numpy.any(realSpace < 0):
    #    print realSpace[numpy.where(realSpace < 0)]

    #realSpaces.append(realSpace)

    #plt.imshow(realSpace)
    #plt.colorbar()
    #plt.show()

#    return mean# / std#getMass(pyshtools.SHExpandDH(1.0 / (1 + numpy.exp(-(realSpace - thresh) / realSpace)), sampling = 2)) / (4 * numpy.pi)

    fm = getEnergies(h)
    fm /= max(fm)
    #plt.plot(fm)
    #plt.xlabel('Harmonic number L')
    #plt.ylabel('L2 norm across all M values for a given L (this is how you get rotation invariance)')
    #plt.title('Rotation invariant feature detectors for ideal microstructures')
    #plt.show()

    print '75', get75percentile(fm[1:]), sum(fm[2::2]) / sum(fm[1:]), fm[0] / sum(fm[1:])

    inf = -4 * numpy.pi * sum(pyshtools.SHCrossPowerSpectrum(h, pyshtools.SHExpandDH(numpy.log2(realSpace + 1e-10), sampling = 2)))

    return inf

def cubicSymmetry(o):
    angles = ((0.0, 0.0), (0.0, 270.0), (0.0, 180.0), (0.0, 90.0), (90.0, 0.0), (90.0, 270.0), (90.0, 180.0), (90.0, 90.0), (180.0, 0.0), (180.0, 270.0), (180.0, 180.0), (180.0, 90.0))

    totalSPH = numpy.zeros(o.shape)

    for thetaphi in angles:
        theta, phi = numpy.radians(thetaphi)

        totalSPH += pyshtools.SHRotateRealCoef(o, (phi, theta, 0.0), dj)

    return totalSPH

def getMass(output):
    return 4 * numpy.pi * sum(pyshtools.SHCrossPowerSpectrum(output, pyshtools.SHExpandDH(numpy.ones(histogram.shape), sampling = 2)))
#print thetas.shape
#%%
plt.imshow(pyshtools.MakeGridDH(cubicSymmetry(output), sampling = 2))
#%%
if True:
#for g in range(len(stacks[:])):
    #histogram = numpy.array(stacks[g])

    output = pyshtools.SHExpandDH(histogram, sampling = 2)

#if True:
   # output = numpy.array(lookup[0, 0])
    output /= getMass(output)

    histogram = pyshtools.MakeGridDH(output, sampling = 2)

    #plt.imshow(histogram)
    #plt.colorbar()
    #plt.show()

    #if numpy.any(histogram < 0):
    #    print 'yadadaddasadsdsaasdda', histogram[numpy.where(histogram < 0)]

    thetas, phis, Ntheta, Nphi = pyshtools.GLQGridCoord(histogram.shape[0] - 1)
    phis = numpy.linspace(0, 360, histogram.shape[1], endpoint = False)

    thetas, phis = numpy.meshgrid(90 - thetas, phis, indexing = 'ij')

    xs = numpy.sin(numpy.pi * thetas / 180.0) * numpy.cos(numpy.pi * phis / 180.0)
    ys = numpy.sin(numpy.pi * thetas / 180.0) * numpy.sin(numpy.pi * phis / 180.0)
    zs = numpy.cos(numpy.pi * thetas / 180.0)

    #ys = numpy.sin(numpy.pi * thetas / 180.0)
    #zs = numpy.cos(numpy.pi * thetas / 180.0)

    yzc = ys * ys + zs * zs
    xzc = xs * xs + zs * zs
    xyc = ys * ys + xs * xs

    xx = pyshtools.SHExpandDH((ys * ys + zs * zs) / numpy.sqrt(yzc), sampling = 2)
    yy = pyshtools.SHExpandDH((xs * xs + zs * zs) / numpy.sqrt(xzc), sampling = 2)
    zz = pyshtools.SHExpandDH((xs * xs + ys * ys) / numpy.sqrt(xyc), sampling = 2)

    xy = pyshtools.SHExpandDH(-xs * ys / numpy.sqrt(xyc), sampling = 2)
    yz = pyshtools.SHExpandDH(-ys * zs / numpy.sqrt(yzc), sampling = 2)
    xz = pyshtools.SHExpandDH(-xs * zs / numpy.sqrt(xzc), sampling = 2)

    Ixx = 4 * numpy.pi * sum(pyshtools.SHCrossPowerSpectrum(output, xx))
    Iyy = 4 * numpy.pi * sum(pyshtools.SHCrossPowerSpectrum(output, yy))
    Izz = 4 * numpy.pi * sum(pyshtools.SHCrossPowerSpectrum(output, zz))
    Ixy = 4 * numpy.pi * sum(pyshtools.SHCrossPowerSpectrum(output, xy))
    Ixz = 4 * numpy.pi * sum(pyshtools.SHCrossPowerSpectrum(output, xz))
    Iyz = 4 * numpy.pi * sum(pyshtools.SHCrossPowerSpectrum(output, yz))

#
#    Ixx = 4 * numpy.pi * sum(pyshtools.SHCrossPowerSpectrum(output, xx))
#    Iyy = 4 * numpy.pi * sum(pyshtools.SHCrossPowerSpectrum(output, yy))
#    Izz = 4 * numpy.pi * sum(pyshtools.SHCrossPowerSpectrum(output, zz))
#    Ixy = 4 * numpy.pi * sum(pyshtools.SHCrossPowerSpectrum(output, xy))
#    Ixz = 4 * numpy.pi * sum(pyshtools.SHCrossPowerSpectrum(output, xz))
#    Iyz = 4 * numpy.pi * sum(pyshtools.SHCrossPowerSpectrum(output, yz))

    M = numpy.array([[Ixx, Ixz, Ixy],
                 [Ixz, Iyy, Iyz],
                 [Ixy, Iyz, Izz]])

    w, v = numpy.linalg.eig(M)

    #for i, theta in enumerate(thetas):
    #    for k, phi in enumerate(phis):
            #print i, k
    #        result = pyshtools.SHRotateRealCoef(output, (theta, phi, 0.0), dj)#pyshtools.MakeGridDH(, sampling = 2)

            #sum(numpy.abs(result.flatten()))
            #result *= histogram

            #for j in range(0, result.shape[0]):
            #    angle = j * numpy.pi / float(result.shape[0])
            #    result[j, :] *= -2 * numpy.pi * (numpy.cos(angle + numpy.pi / float(result.shape[0])) - numpy.cos(angle))#numpy.max(histogram[i, :])

            #plt.imshow(result)
            #plt.show()
    #        distance[i, k] = sum(pyshtools.SHCrossPowerSpectrum(result, output))

    #print M
    #angles = []
    #for i in range(v.shape[1]):
    #    theta = numpy.degrees(numpy.arccos(v[2, i]))
    #    phi = numpy.degrees(numpy.arctan2(v[1, i], v[0, i]))
    #    angles.append((theta, phi))
    #print angles
    #print v
    print sorted(w / sum(w)), sum(w)
    #sharpness(output)
    #plt.imshow(histogram)
    #ax = plt.gca()
    #for theta, phi in angles:
    #    if phi < 0.0:
    #        phi += 360
    #    circle1 = plt.Circle((phi, theta), 5, color='g')
    #    ax.add_artist(circle1)
    #plt.show()

    #plt.imshow(distance, interpolation='none', extent=[0, 360, 180, 0])
    #plt.xlabel('Phi (East West)')
    #plt.ylabel('Theta (North South)')
    #plt.title('Non-Rotated')
    #plt.show()
#%%
for stack in range(10):
#if True:
#    stack = 0
    ims = skimage.io.imread_collection('/home/bbales2/rafting/rafting2ah5/images_{0}/*.png'.format(stack))
    ims = ims.concatenate()

    ims = rescale(ims, 0.0, 1.0)

    filtrd = scipy.ndimage.filters.gaussian_filter(ims, 1.0)

    imx = filtrd[filtrd.shape[0] / 2, :, :]
    imy = filtrd[:, filtrd.shape[1] / 2, :]
    imz = filtrd[:, :, filtrd.shape[2] / 2]
    #imx = xslices[stack]
    #imy = yslices[stack]
    #imz = zslices[stack]

    dxs, dys, dzs = numpy.gradient(filtrd)
    #dys, dzs = numpy.gradient(imx)
    #dys = -dys

    thetas, histogramx = hog.build2dHist(dys, dzs, 360)

    thetas = numpy.hstack((thetas, [2 * numpy.pi]))

    histogramx = numpy.hstack((histogramx, [histogramx[0]]))

    #plt.polar(thetas, histogramx)
    #plt.show()

    #dxs, dzs = numpy.gradient(imy)
    #dys = -dys

    thetas, histogramy = hog.build2dHist(dxs, dzs, 360)

    thetas = numpy.hstack((thetas, [2 * numpy.pi]))
    histogramy = numpy.hstack((histogramy, [histogramy[0]]))
    #plt.polar(thetas, histogramy)
    #plt.show()

    #dxs, dys = numpy.gradient(imz)
    #dys = -dys

    thetas, histogramz = hog.build2dHist(dxs, dys, 360)

    thetas = numpy.hstack((thetas, [2 * numpy.pi]))
    histogramz = numpy.hstack((histogramz, [histogramz[0]]))

    #plt.polar(thetas, histogramz)
    #plt.show()


    #print '----------'

    histogramx /= (imx.shape[0] * imx.shape[1])
    histogramy /= (imy.shape[0] * imy.shape[1])
    histogramz /= (imz.shape[0] * imz.shape[1])

    a = scipy.integrate.cumtrapz(histogramx, thetas)[-1] + scipy.integrate.cumtrapz(histogramy, thetas)[-1] + scipy.integrate.cumtrapz(histogramz, thetas)[-1]

    histogramx /= a
    histogramy /= a
    histogramz /= a

    #print scipy.integrate.cumtrapz(histogramx, thetas)[-1]
    #print scipy.integrate.cumtrapz(histogramy, thetas)[-1]
    #print scipy.integrate.cumtrapz(histogramz, thetas)[-1]

    #histogramx /= scipy.integrate.cumtrapz(histogramx, thetas)[-1]
    #histogramy /= scipy.integrate.cumtrapz(histogramy, thetas)[-1]
    #histogramz /= scipy.integrate.cumtrapz(histogramz, thetas)[-1]

    #plt.polar(thetas, histogramx)
    #plt.polar(thetas, histogramy)
    #plt.polar(thetas, histogramz)
    #plt.legend(['x', 'y', 'z'])
    #plt.show()

    xs = numpy.cos(thetas)
    ys = numpy.sin(thetas)

    #xs = histogramx * numpy.cos(thetas)
    #ys = histogramx * numpy.sin(thetas)
    Ixx = scipy.integrate.cumtrapz(histogramx * (xs ** 2 + ys ** 2), thetas)[-1] # y z
    #xs = histogramy * numpy.cos(thetas)
    #ys = histogramy * numpy.sin(thetas)
    Iyy = scipy.integrate.cumtrapz(histogramy * (xs ** 2 + ys ** 2), thetas)[-1] # x z
    #xs = histogramz * numpy.cos(thetas)
    #ys = histogramz * numpy.sin(thetas)
    Izz = scipy.integrate.cumtrapz(histogramz * (xs ** 2 + ys ** 2), thetas)[-1] # x y
    #xs = histogramx * numpy.cos(thetas)
    #ys = histogramx * numpy.sin(thetas)
    Iyz = scipy.integrate.cumtrapz(histogramx * -(xs * ys), thetas)[-1] # y z
    #xs = histogramy * numpy.cos(thetas)
    #ys = histogramy * numpy.sin(thetas)
    Ixz = scipy.integrate.cumtrapz(histogramy * -(xs * ys), thetas)[-1] # x z
    #xs = histogramz * numpy.cos(thetas)
    #ys = histogramz * numpy.sin(thetas)
    Ixy = scipy.integrate.cumtrapz(histogramz * -(xs * ys), thetas)[-1] # x y

    M = numpy.array([[Ixx, Ixy, Ixz],
                     [Ixy, Iyy, Iyz],
                     [Ixz, Iyz, Izz]])

    w, v = numpy.linalg.eig(M)

    #print M

    print sorted(w)#, v


