import numpy as np
from Corrfunc import utils as cfutils
import os


def parse_coords(coords):

	"""
	coord expected to be a 3-tuple. If only doing an angular measurement, append None for the distance column
	also convert arrays to native endianness
	"""
	lons, lats = coords[0], coords[1]
	lons, lats = cfutils.convert_to_native_endian(lons), cfutils.convert_to_native_endian(lats)
	if coords is not None:
		if len(coords) == 2:
			coords = lons, lats, None
		else:
			coords = lons, lats, coords[2]
		if coords[2] is not None:
			coords = lons, lats, cfutils.convert_to_native_endian(coords[2])
	return coords


def parse_weights(numcoords, weights):
	"""
	Normalizing a correlation function requires number of data and random points
	If data/randoms are weighted, want a sum of weights rather than a count of coordinates
	:return: int, number of points
	"""
	if weights is not None:
		n = np.sum(weights)
		weights = cfutils.convert_to_native_endian(weights)
	else:
		n = numcoords
	return n, weights


def index_tuple(mytuple, idxs):
	mylist = list(mytuple)
	for j in range(len(mylist)):
		if mylist[j] is not None:
			mylist[j] = np.array(mylist[j])
			mylist[j] = mylist[j][idxs]
	return tuple(mylist)


def get_nthreads():
	"""
	count number of available threads for Corrfunc to take advantage of
	:return:
	"""
	return os.cpu_count()


# calculate either logarithmic or linear centers of scale bins
def bin_centers(binedges):
	# check if linear
	if (binedges[2] - binedges[1]) == (binedges[1] - binedges[0]):
		return (binedges[1:] + binedges[:-1]) / 2
	# otherwise do geometric mean
	else:
		return np.sqrt(binedges[1:] * binedges[:-1])


def match_random_zdist(samplecat, randomcat, zbins, n_rand2n_dat=10):
	minz, maxz = np.min(samplecat['Z']) - 0.01, np.max(samplecat['Z']) + 0.01
	zhist, edges = np.histogram(samplecat['Z'], bins=zbins, range=(minz, maxz), density=True)
	randhist, edges = np.histogram(randomcat['Z'], bins=zbins, range=(minz, maxz), density=True)
	ratio = zhist/randhist
	sampleweights = ratio[np.digitize(randomcat['Z'], bins=edges) - 1]
	subrandcat = randomcat[np.random.choice(len(randomcat),
											size=(n_rand2n_dat*len(samplecat)),
											replace=False,
											p=(sampleweights / np.sum(sampleweights)))]
	return subrandcat


def process_catalog(cat):
	ra, dec = cat['RA'], cat['DEC']
	try:
		chi = cat['CHI']
	except:
		print('No distances given, doing angular clustering')
		chi = None

	try:
		weight = cat['weight']
	except:
		print('No weights given')
		weight = None

	coord = (ra, dec, chi)
	return coord, weight

