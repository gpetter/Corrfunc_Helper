import numpy as np
from Corrfunc import utils as cfutils
import os
from scipy.special import legendre

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

def check_crosscorr(coords1, coords2, randcoords1, randcoords2):
	"""
	Ensures that a cross correlation won't crash if one sample has distances (redshifts/chis) and other does not
	(want an angular cross correlation between a spectroscopic and photometric sample)
	"""
	if coords1[2] is None:
		print("One sample has distance, but other doesn't, falling back to angular")
		coords2 = coords2[0], coords2[1], None
		if randcoords2 is not None:
			randcoords2 = randcoords2[0], randcoords2[1], None
	if coords2[2] is None:
		print("One sample has distance, but other doesn't, falling back to angular")
		coords1 = coords1[0], coords1[1], None
		if randcoords1 is not None:
			randcoords1 = randcoords1[0], randcoords1[1], None
	return coords1, coords2, randcoords1, randcoords2

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
	"""
	indexing not supported for tuples of numpy arrays. Convert and then convert back to indexed tuple
	:param mytuple: tuple of (ras, decs, chis)
	:param idxs: np array of indices to keep
	:return:
	"""
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


def bin_centers(binedges):
	"""
	calculate either logarithmic or linear centers of scale bins
	:param binedges:
	:return:
	"""
	# check if linear
	if (binedges[2] - binedges[1]) == (binedges[1] - binedges[0]):
		return (binedges[1:] + binedges[:-1]) / 2
	# otherwise do geometric mean
	else:
		return np.sqrt(binedges[1:] * binedges[:-1])


def match_random_zdist(samplecat, randomcat, zbins, n_rand2n_dat=10):
	"""
	Subsample the random catalog to match the dn/dz of the data sample
	:param samplecat: astropy data table
	:param randomcat: astropy random table
	:param zbins: int, number of z bins
	:param n_rand2n_dat: int, multiple more randoms than data
	:return:
	"""
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
	"""
	Parse input astropy table and return tuples of coordinates and weights
	:param cat: astropy table
	:return:
	"""
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


def tpcf_multipole(s_mu_tcpf_result, mu_bins, order=0):
    """
    Taken from halotools
	Original author Duncan Campbell
    Calculate the multipoles of the two point correlation function
    after first computing `~halotools.mock_observables.s_mu_tpcf`.

    Parameters
    ----------
    s_mu_tcpf_result : np.ndarray
        2-D array with the two point correlation function calculated in bins
        of :math:`s` and :math:`\mu`.  See `~halotools.mock_observables.s_mu_tpcf`.

    mu_bins : array_like
        array of :math:`\mu = \cos(\theta_{\rm LOS})`
        bins for which ``s_mu_tcpf_result`` has been calculated.
        Must be between [0,1].

    order : int, optional
        order of the multpole returned.

    Returns
    -------
    xi_l : np.array
        multipole of ``s_mu_tcpf_result`` of the indicated order.

    Examples
    --------
    For demonstration purposes we create a randomly distributed set of points within a
    periodic cube of length 250 Mpc/h.

    """

    # process inputs
    s_mu_tcpf_result = np.atleast_1d(s_mu_tcpf_result)
    mu_bins = np.atleast_1d(mu_bins)
    order = int(order)

    # calculate the center of each mu bin
    mu_bin_centers = (mu_bins[:-1]+mu_bins[1:])/(2.0)

    # get the Legendre polynomial of the desired order.
    Ln = legendre(order)

    # numerically integrate over mu
    result = (2.0*order + 1.0)/2.0 * np.sum(s_mu_tcpf_result * np.diff(mu_bins) *\
        (Ln(mu_bin_centers) + Ln(-1.0*mu_bin_centers)), axis=1)

    return result