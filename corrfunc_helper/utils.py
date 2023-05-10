import numpy as np
from Corrfunc import utils as cfutils
import os


def parse_coords(coords):

	"""
	coord expected to be a 3-tuple. If only doing an angular measurement, append None for the distance column
	also convert arrays to native endianness
	"""
	if coords is not None:
		lons, lats = coords[0], coords[1]
		lons, lats = cfutils.convert_to_native_endian(lons), cfutils.convert_to_native_endian(lats)
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

def bin_in_pi(cf, pimax=40, dpi=2.):
	"""
	corrfunc by default only allows pi bin sizes of 1 Mpc/h
	this function sums up pair counts with a new dpi value (should be int > 1)

	:param cf:
	:param pimax:
	:param dpi:
	:return:
	"""
	pimax = int(pimax)
	npi = int(pimax / dpi)
	newarr = []
	nrbins = int(len(cf) / pimax)

	# for each rp bin
	for i in range(nrbins):
		# counts within rp bin
		cf_r = cf[int(pimax) * i:(int(pimax) * (i + 1))]

		rmin = cf_r[0][0]
		rmax = cf_r[0][1]

		# collect pair counts and weights
		pairs, weights = [], []
		for j in range(pimax):
			pairs.append(cf_r[j][4]), weights.append(cf_r[j][5])

		# sum up number of pairs, take average of weights within new pi bin
		newpairs, newweights = [], []
		for j in range(npi):
			pairs_in_newbin = np.array(pairs[int(dpi) * j:int(dpi) * (j + 1)])
			weights_in_newbin = np.array(weights[int(dpi) * j:int(dpi) * (j + 1)])
			# only want average of weights where pairs detected
			nonzero_idxs = (pairs_in_newbin > 0)

			newpairs.append(np.sum(pairs_in_newbin))
			if np.any(nonzero_idxs):
				newweight = np.mean(weights_in_newbin[nonzero_idxs], )
			else:
				newweight = 0.

			newweights.append(newweight)

		# package back up into format corrfunc expects
		for j in range(npi):
			newtuple = (rmin, rmax, 0., ((j + 1) * dpi), newpairs[j], newweights[j])
			newarr.append(newtuple)
	return np.array(newarr, dtype=cf.dtype)