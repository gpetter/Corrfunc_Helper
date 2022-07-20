import numpy as np
import healpy as hp
from sklearn import cluster


# split sample into N jackknife regions using K-means clustering
def bin_on_sky(ras, decs, npatches, nsides=64):
	# healpix pixels with sources in them
	occupiedpix = hp.ang2pix(nside=nsides, theta=ras, phi=decs, lonlat=True)
	# get rid of repeats
	uniquepix = np.unique(occupiedpix)
	# coordinates of occupied healpix pixels
	hp_ra, hp_dec = hp.pix2ang(nside=nsides, ipix=uniquepix, lonlat=True)
	# use Kmeans clustering to divide into N jackknife regions
	coords = np.concatenate([[hp_ra], [hp_dec]])
	coords = coords.T
	kmeans = cluster.KMeans(n_clusters=npatches, n_init=10)
	kmeans.fit(coords)

	# integers to identify each patch
	labels = kmeans.labels_
	# setup healpix map fully masked
	blankmap = hp.UNSEEN * np.ones(hp.nside2npix(nsides))
	# set occupied pixels to their patch value
	blankmap[uniquepix] = labels
	return blankmap


# estimate covariance from many bootstrap/jackknife resamplings of data
def covariance_matrix(resampled_profiles, avg_profile):
	n_bins, n_realizations = len(resampled_profiles[0]), len(resampled_profiles)
	# initialize matrix
	c_ij = np.zeros((n_bins, n_bins))
	# loop over resamplings, for each i and j combination, calculate product of difference from mean result
	for i in range(n_bins):
		for j in range(n_bins):
			product = (resampled_profiles[:, i] - avg_profile[i]) * (resampled_profiles[:, j] - avg_profile[j])
			sum = np.sum(product)
			c_ij[i, j] = 1 / (n_realizations - 1) * sum

	return c_ij
