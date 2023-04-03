from Corrfunc.mocks.DDtheta_mocks import DDtheta_mocks
from Corrfunc.mocks.DDrppi_mocks import DDrppi_mocks
from Corrfunc.mocks.DDsmu_mocks import DDsmu_mocks
import os
from . import estimators
import numpy as np
import healpy as hp
from . import jackknife
from . import plots
from functools import partial

coord_dict = {'RA': 0, 'DEC': 1, 'CHI': 2}


def parse_coords(coords):
	if coords is not None:
		if len(coords) == 2:
			coords = coords[0], coords[1], None
	return coords


def parse_weights(numcoords, weights):
	if weights is not None:
		n = np.sum(weights)
	else:
		n = numcoords
	return n

def index_tuple(mytuple, idxs):
	mylist = list(mytuple)
	for j in range(len(mylist)):
		if mylist[j] is not None:
			mylist[j] = np.array(mylist[j])
			mylist[j] = mylist[j][idxs]
	return tuple(mylist)



def get_nthreads():
	return os.cpu_count()


# calculate either logarithmic or linear centers of scale bins
def bin_centers(binedges):
	# check if linear
	if (binedges[2] - binedges[1]) == (binedges[1] - binedges[0]):
		return (binedges[1:] + binedges[:-1]) / 2
	# otherwise do geometric mean
	else:
		return np.sqrt(binedges[1:] * binedges[:-1])


# count autocorrelation pairs
def auto_counts(scales, coords, weights, nthreads=1, fulldict=True, pimax=40., mubins=None):
	# unpack coordinate tuple
	# third entry should be None if you only want angular counts
	ras, decs, chis = coords
	# if weights not given, don't use weights
	if weights is None:
		weights, weight_type = None, None
	else:
		weight_type = 'pair_product'

	# if distances given, measure counts in rp and pi bins
	if chis is not None:
		if mubins is None:
			dd = DDrppi_mocks(autocorr=1, cosmology=2, nthreads=nthreads, pimax=pimax, binfile=scales,
							RA1=ras, DEC1=decs, CZ1=chis, weights1=weights,
							is_comoving_dist=True, weight_type=weight_type)
		else:
			dd = DDsmu_mocks(autocorr=1, cosmology=2, nthreads=nthreads, mu_max=1., nmu_bins=mubins,
							binfile=scales, RA1=ras, DEC1=decs, CZ1=chis, weights1=weights, is_comoving_dist=True,
							weight_type=weight_type)
	# if no distances given, only get angular counts
	else:
		dd = DDtheta_mocks(1, nthreads, scales, ras, decs, weights1=weights, weight_type=weight_type)
	# return the full dictionary from Corrfunc or simply the weighted number of counts
	if fulldict:
		return dd
	else:
		if np.max(dd['weightavg']) > 0:
			return dd['npairs'] * dd['weightavg']
		else:
			return dd['npairs']


# count cross-pairs between samples
def cross_counts(scales, coords1, coords2, weights1, weights2, nthreads=1, fulldict=True, pimax=40., mubins=None):
	# unpack coordinate tuple
	# third entry should be None if you only want angular counts
	ras1, decs1, chis1 = coords1
	ras2, decs2, chis2 = coords2

	# if weights not given don't use weights
	if (weights1 is None) or (weights2 is None):
		weights1, weights2, weight_type = None, None, None
	else:
		weight_type = 'pair_product'

	# if distances given, measure counts in rp and pi bins
	if chis1 is not None:
		if mubins is None:
			dr = DDrppi_mocks(0, 2, nthreads, pimax, scales, ras1, decs1, chis1, weights1=weights1,
								RA2=ras2, DEC2=decs2, CZ2=chis2, weights2=weights2, is_comoving_dist=True,
								weight_type=weight_type)
		else:
			dr = DDsmu_mocks(autocorr=0, cosmology=2, nthreads=nthreads, mu_max=1., nmu_bins=mubins,
							binfile=scales, RA1=ras1, DEC1=decs1, CZ1=chis1, weights1=weights1,
							RA2=ras2, DEC2=decs2, CZ2=chis2, weights2=weights2, is_comoving_dist=True,
							weight_type=weight_type)
	# if no distances given, only get angular counts
	else:
		dr = DDtheta_mocks(0, nthreads, scales, ras1, decs1, RA2=ras2, DEC2=decs2, weights1=weights1,
							weights2=weights2, weight_type=weight_type)
	# return the full dictionary from Corrfunc or simply the weighted number of counts
	if fulldict:
		return dr
	else:
		if np.max(dr['weightavg']) > 0:
			return dr['npairs'] * dr['weightavg']
		else:
			return dr['npairs']


# count pairs inside a jackknife region
def counts_in_patch(patchval, patchmap, scales, nthreads,
					coords1, randcoords1, weights1, randweights1,
					coords2, randcoords2, weights2, randweights2,
					mubins, estimator):
	nside = hp.npix2nside(len(patchmap))

	# unpack coordinates
	ras1, decs1, chis1 = coords1
	randras1, randdecs1, randchis1 = randcoords1

	# get only sources/randoms inside patch
	idxs1_in_patch = np.where(patchmap[hp.ang2pix(nside, ras1, decs1, lonlat=True)] == patchval)
	randidxs1_in_patch = np.where(patchmap[hp.ang2pix(nside, randras1, randdecs1, lonlat=True)] == patchval)
	coords1 = index_tuple(coords1, idxs1_in_patch)
	randcoords1 = index_tuple(randcoords1, randidxs1_in_patch)

	# if no weights given, use number of points
	n_data1, n_rands1 = len(coords1[0]), len(randcoords1[0])

	# otherwise sum up weights
	if weights1 is not None:
		weights1 = weights1[idxs1_in_patch]
		n_data1 = parse_weights(len(weights1), weights1)
	if randweights1 is not None:
		randweights1 = randweights1[randidxs1_in_patch]
		n_rands1 = parse_weights(len(randweights1), randweights1)

	# if doing autocorrelation
	if coords2 is None:
		n_data2, n_rands2 = n_data1, n_rands1
		d1d2counts = auto_counts(scales, coords1, weights1, nthreads=nthreads, fulldict=False, mubins=mubins)
		d1r2counts = cross_counts(scales, coords1, randcoords1, weights1, randweights1, nthreads=nthreads,
								fulldict=False, mubins=mubins)
		# data-random cross counts are symmetric
		d2r1counts = d1r2counts

		# only count random-random pairs if using Landy-Szalay estimator
		if estimator == 'LS':
			r1r2counts = auto_counts(scales, randcoords1, randweights1, nthreads=nthreads, fulldict=False, mubins=mubins)
		else:
			r1r2counts = np.zeros_like(d1d2counts)
	# otherwise a cross correlation
	else:
		ras2, decs2, chis2 = coords2
		# get only sources/randoms inside patch
		idxs2_in_patch = np.where(patchmap[hp.ang2pix(nside, ras2, decs2, lonlat=True)] == patchval)
		coords2 = index_tuple(coords2, idxs2_in_patch)
		n_data2, n_rands2 = len(coords2[0]), 0
		if weights2 is not None:
			weights2 = weights2[idxs2_in_patch]
			n_data2 = parse_weights(len(weights2), weights2)

		# do cross correlations
		d1d2counts = cross_counts(scales, coords1, coords2, weights1, weights2, nthreads=nthreads,
								fulldict=False, mubins=mubins)
		# only required to have one random catalog
		d2r1counts = cross_counts(scales, coords2, randcoords1, weights2, randweights1, nthreads=nthreads,
								fulldict=False, mubins=mubins)
		d1r2counts, r1r2counts = np.zeros_like(d1d2counts), np.zeros_like(d1d2counts)
		# if using LS estimator
		if estimator == 'LS':
			# then second random catalog is required
			randras2, randdecs2, randchis2 = randcoords2
			randidxs2_in_patch = np.where(patchmap[hp.ang2pix(nside, randras2, randdecs2, lonlat=True)] == patchval)
			randcoords2 = index_tuple(randcoords2, randidxs2_in_patch)
			n_rands2 = len(randcoords2[0])
			if randweights2 is not None:
				randweights2 = randweights2[randidxs2_in_patch]
				n_rands2 = parse_weights(len(randweights2), randweights2)

			d1r2counts = cross_counts(scales, coords1, randcoords2, weights1, randweights2, nthreads=nthreads,
								fulldict=False, mubins=mubins)
			r1r2counts = cross_counts(scales, randcoords1, randcoords2, randweights1, randweights2, nthreads=nthreads,
								fulldict=False, mubins=mubins)

	return [d1d2counts, d1r2counts, d2r1counts, r1r2counts, n_data1, n_rands1, n_data2, n_rands2]


#
def bootstrap_realizations(scales, nbootstrap, nthreads, coords1, randcoords1, weights1, randweights1,
						coords2=None, randcoords2=None, weights2=None, randweights2=None,
						oversample=1, pimax=40., mubins=None, npatches=30, estimator='LS', wedges=None):

	# split footprint into N patches
	patchmap = jackknife.bin_on_sky(randcoords1, npatches=npatches)
	# get IDs for each patch
	unique_patchvals = np.unique(patchmap)
	# remove masked patches
	unique_patchvals = unique_patchvals[np.where(unique_patchvals > -1e30)]

	part_func = partial(counts_in_patch, patchmap=patchmap, scales=scales, nthreads=nthreads,
						coords1=coords1, randcoords1=randcoords1,
						weights1=weights1, randweights1=randweights1,
						coords2=coords2, randcoords2=randcoords2,
						weights2=weights2, randweights2=randweights2,
						mubins=mubins, estimator=estimator)

	# map cores to different patches, count pairs within
	counts = list(map(part_func, unique_patchvals))
	counts = np.array(counts)

	# separate out DD, DR, RR counts from returned array
	d1d2_counts, d1r2_counts, d2r1_counts, r1r2_counts = counts[:, 0], counts[:, 1], counts[:, 2], counts[:, 3]
	ndata1, nrands1 = counts[:, 4], counts[:, 5]
	ndata2, nrands2 = counts[:, 6], counts[:, 7]

	# keep track of both the 1D and 2D CFs obtained by each resampling
	w_realizations, xi_realizations = [], []

	# for each bootstrap, randomly select patches (oversampled by factor of N) and sum counts in those patches
	for k in range(nbootstrap):
		# randomly select patches with replacement
		boot_patches = np.random.choice(np.arange(len(unique_patchvals)), oversample * len(unique_patchvals))

		# sum pairs in those patches
		bootndata1, bootnrands1 = ndata1[boot_patches], nrands1[boot_patches]
		bootndata2, bootnrands2 = ndata2[boot_patches], nrands2[boot_patches]
		totdata1, totrands1 = np.sum(bootndata1) / np.float(oversample), np.sum(bootnrands1) / np.float(oversample)
		totdata2, totrands2 = np.sum(bootndata2) / np.float(oversample), np.sum(bootnrands2) / np.float(oversample)
		boot_d1d2counts, boot_d1r2counts, boot_d2r1counts, bootr1r2counts = d1d2_counts[boot_patches], \
																			d1r2_counts[boot_patches], \
																			d2r1_counts[boot_patches], \
																			r1r2_counts[boot_patches]
		totd1d2counts, totd1r2counts, totd2r1counts, totr1r2counts = np.sum(boot_d1d2counts, axis=0), \
																	 np.sum(boot_d1r2counts, axis=0), \
																	 np.sum(boot_d2r1counts, axis=0), \
																	 np.sum(bootr1r2counts, axis=0)

		cf = estimators.convert_counts_to_cf(totdata1, totdata2, totrands1,
											totrands2, totd1d2counts, totd1r2counts, totd2r1counts,
											totr1r2counts, estimator=estimator)
		if coords1[2] is None:
			w_realizations.append(cf)
			xi_realizations = None
		else:
			xi_realizations.append(cf)
			if mubins is None:
				w_realizations.append(estimators.convert_cf_to_wp(cf, nrpbins=len(scales)-1, pimax=pimax))
			else:
				xi_s = estimators.convert_cf_to_xi_s(cf, nsbins=len(scales) - 1, nmubins=mubins, wedges=wedges)
				#mono, quad = xi_s[0], xi_s[1]
				w_realizations.append(xi_s)
	return w_realizations, xi_realizations


# Measure an autocorrelation function. Coords should be either tuple of (ra, dec) or (ra, dec, chi)
# the former measures an angular CF, latter a spatial CF
def autocorr_from_coords(scales, coords, randcoords, weights=None, randweights=None,
						nthreads=None, estimator='LS', pimax=40., dpi=1., mubins=None,
						nbootstrap=0, oversample=1, plot_2dcf=False, lims_2dcf=(None, None), wedges=None):
	if nthreads is None:
		nthreads = get_nthreads()
	coords, randcoords = parse_coords(coords), parse_coords(randcoords)

	# number of objects = sum of weights
	if (weights is not None) and (randweights is not None):
		n_data, n_rands = np.sum(weights), np.sum(randweights)
	# if no weights given, weights are 1
	else:
		n_data, n_rands = len(coords[0]), len(randcoords[0])

	# if plotting xi(rp, pi), want number of rp and pi bins to match
	if plot_2dcf & (mubins is None):
		scales = np.linspace(0.1, pimax, int(pimax)+1)

	# autocorrelation of catalog
	DD_counts = auto_counts(scales, coords, weights=weights, nthreads=nthreads, pimax=pimax, mubins=mubins)

	# cross correlation between data and random catalog
	DR_counts = cross_counts(scales, coords, randcoords, weights1=weights, weights2=randweights,
								nthreads=nthreads, pimax=pimax, mubins=mubins)

	# autocorrelation of random points
	if estimator == 'LS':
		RR_counts = auto_counts(scales, randcoords, weights=randweights, nthreads=nthreads, pimax=pimax, mubins=mubins)
	# If using DD/DR-1, don't need to calculate expensive RR pairs
	else:
		RR_counts = None

	# prepare dictionary to return as output
	outdict = {}
	# calculate correlation function
	cf = estimators.convert_counts_to_cf(n_data, n_data, n_rands, n_rands, DD_counts, DR_counts,
									DR_counts, RR_counts, estimator=estimator)

	# find centers of bins in log or linear space
	effective_scales = bin_centers(scales)

	# angular correlation function if no redshifts given
	if coords[2] is None:
		outdict['theta'] = effective_scales
		outdict['w_theta'] = cf
		# Poisson error e.g. Dipompeo et al. 2017
		outdict['w_err_poisson'] = np.sqrt(2 * np.square(1 + cf) / DD_counts['npairs'])

	# spatial correlation function if redshifts given
	else:
		# plot xi(rp, pi) or xi(s, mu) if desired
		if plot_2dcf:
			if mubins is None:
				return plots.plot_2d_corr_func(cf, inputrange=lims_2dcf)
			else:
				return plots.xi_mu_s_plot(cf, nsbins=len(scales)-1, nmubins=mubins, inputrange=lims_2dcf)

		# if getting projected correlation function wp(rp), and full 2D xi(rp, pi)
		if mubins is None:
			# find centers of rp bins
			outdict['rp'] = effective_scales
			# integrate xi(rp, pi) over pi to get wp(rp)
			outdict['wp'] = estimators.convert_cf_to_wp(cf, nrpbins=len(scales)-1,
											pimax=pimax, dpi=dpi)
			# Poisson errors on 2D CF
			# if N_pairs > N_data points (at large scales), use minimum of the two to get Poisson errors
			# e.g. Shen et al 2007
			mincounts = np.min([np.array(DD_counts['npairs']), len(coords[0]) * np.ones_like(DD_counts['npairs'])], axis=0)
			# when number of counts in a bin == 0, poisson uncertainty is infinite.
			# 0 +/- 1 is approximate, but use caution
			# where N_pair==0, set to 1, as sqrt(1) = 1
			mincounts[np.where(mincounts == 0)] = 1
			xi_rp_poisson_errs = np.sqrt(2 * np.square(1 + cf) / mincounts)
			# propogate Poisson error on xi(rp, pi) to projected clustering
			outdict['wp_poisson_err'] = np.sqrt(2 * dpi * estimators.convert_cf_to_wp(np.square(xi_rp_poisson_errs),
											nrpbins=len(scales)-1, pimax=pimax, dpi=dpi))
			# add 2D CF and Poisson error to output
			outdict['xi_rp_pi'] = cf
			outdict['xi_rp_pi_poisson_err'] = xi_rp_poisson_errs

		# otherwise get redshift space clustering
		else:
			outdict['s'] = effective_scales
			w = estimators.convert_cf_to_xi_s(cf, nsbins=len(scales)-1, nmubins=mubins, wedges=wedges)
			outdict['mono'], outdict['quad'] = w[0], w[1]


	# perform bootstrap resampling of patches for uncertainty
	if nbootstrap > 0:
		# collect 2D and 3D CF realizations from N different resamplings
		w_realizations, xi_realizations = bootstrap_realizations(scales, nbootstrap, nthreads,
											coords1=coords, randcoords1=randcoords, weights1=weights,
											randweights1=randweights, coords2=None, randcoords2=None,
											weights2=None, randweights2=None,
											oversample=oversample, pimax=pimax,
											estimator=estimator, mubins=mubins, wedges=wedges)
		# error estimate is variance of realizations
		w_realization_variance = np.std(w_realizations, axis=0)

		if coords[2] is None:
			outdict['w_err'] = w_realization_variance
			outdict['covar'] = jackknife.covariance_matrix(np.array(w_realizations), np.array(outdict['w_theta']))
		else:
			xi_realization_variance = np.std(xi_realizations, axis=0)
			if mubins is None:
				outdict['wp_err'] = w_realization_variance
				outdict['xi_rp_pi_err'] = xi_realization_variance
				outdict['covar'] = jackknife.covariance_matrix(np.array(w_realizations), np.array(outdict['wp']))
			else:
				outdict['mono_err'], outdict['quad_err'] = w_realization_variance[0], w_realization_variance[1]

	return outdict




def crosscorr_from_coords(scales, coords1, coords2, randcoords1, randcoords2=None, weights1=None,
						weights2=None, randweights1=None, randweights2=None,
						nthreads=None, estimator='LS', pimax=40., dpi=1., mubins=None,
						nbootstrap=0, oversample=1, plot_2dcf=False, lims_2dcf=(None, None), wedges=None):
	if nthreads is None:
		nthreads = get_nthreads()

	coords1, coords2 = parse_coords(coords1), parse_coords(coords2)
	randcoords1, randcoords2 = parse_coords(randcoords1), parse_coords(randcoords2)

	n_data1, n_rands1 = parse_weights(len(coords1[0]), weights1), parse_weights(len(randcoords1[0]), randweights1)
	n_data2, n_rands2 = parse_weights(len(coords2[0]), weights2), parse_weights(len(randcoords2[0]), randweights2)

	# if plotting xi(rp, pi), want number of rp and pi bins to match
	if plot_2dcf & (mubins is None):
		scales = np.linspace(0.1, pimax, int(pimax)+1)

	# cross correlation of data points
	D1D2_counts = cross_counts(scales, coords1, coords2,
							weights1=weights1, weights2=weights2,
							nthreads=nthreads, pimax=pimax, mubins=mubins)

	# cross correlation between first data and second random catalogs
	D2R1_counts = cross_counts(scales, coords2, randcoords1, weights1=weights2, weights2=randweights1,
							nthreads=nthreads, pimax=pimax, mubins=mubins)

	if estimator == 'LS':
		# cross correlation between second data and first random catalogs
		D1R2_counts = cross_counts(scales, coords1, randcoords2, weights1=weights1, weights2=randweights2,
								nthreads=nthreads, pimax=pimax, mubins=mubins)

		R1R2_counts = cross_counts(scales, randcoords1, randcoords2, weights1=randweights1, weights2=randweights2,
								nthreads=nthreads, pimax=pimax, mubins=mubins)
	else:
		D1R2_counts, R1R2_counts = None, None


	# prepare dictionary to return as output
	outdict = {}
	# calculate correlation function
	cf = estimators.convert_counts_to_cf(n_data1, n_data2, n_rands1, n_rands2, D1D2_counts, D1R2_counts,
									D2R1_counts, R1R2_counts, estimator=estimator)

	# find centers of bins in log or linear space
	effective_scales = bin_centers(scales)

	# angular correlation function if no redshifts given
	if coords1[2] is None:
		outdict['theta'] = effective_scales
		outdict['w_theta'] = cf
		# Poisson error e.g. Dipompeo et al. 2017
		outdict['w_err_poisson'] = np.sqrt(1 * np.square(1 + cf) / D1D2_counts['npairs'])

	# spatial correlation function if redshifts given
	else:
		# plot xi(rp, pi) or xi(s, mu) if desired
		if plot_2dcf:
			if mubins is None:
				return plots.plot_2d_corr_func(cf, inputrange=lims_2dcf)
			else:
				return plots.xi_mu_s_plot(cf, nsbins=len(scales)-1, nmubins=mubins, inputrange=lims_2dcf)

		# if getting projected correlation function wp(rp), and full 2D xi(rp, pi)
		if mubins is None:
			# find centers of rp bins
			outdict['rp'] = effective_scales
			# integrate xi(rp, pi) over pi to get wp(rp)
			outdict['wp'] = estimators.convert_cf_to_wp(cf, nrpbins=len(scales)-1,
											pimax=pimax, dpi=dpi)
			# Poisson errors on 2D CF
			# if N_pairs > N_data points (at large scales), use minimum of the two to get Poisson errors
			# e.g. Shen et al 2007
			mincounts = np.min([np.array(D1D2_counts['npairs']),
								len(coords1[0]) * np.ones_like(D1D2_counts['npairs']),
								len(coords2[0]) * np.ones_like(D1D2_counts['npairs'])], axis=0)
			# when number of counts in a bin == 0, poisson uncertainty is infinite.
			# 0 +/- 1 is approximate, but use caution
			# where N_pair==0, set to 1, as sqrt(1) = 1
			mincounts[np.where(mincounts == 0)] = 1
			xi_rp_poisson_errs = np.sqrt(1 * np.square(1 + cf) / mincounts)
			# propogate Poisson error on xi(rp, pi) to projected clustering
			outdict['wp_poisson_err'] = np.sqrt(2 * dpi * estimators.convert_cf_to_wp(np.square(xi_rp_poisson_errs),
											nrpbins=len(scales)-1, pimax=pimax, dpi=dpi))
			# add 2D CF and Poisson error to output
			outdict['xi_rp_pi'] = cf
			outdict['xi_rp_pi_poisson_err'] = xi_rp_poisson_errs

		# otherwise get redshift space clustering
		else:
			outdict['s'] = effective_scales
			w = estimators.convert_cf_to_xi_s(cf, nsbins=len(scales)-1, nmubins=mubins, wedges=wedges)
			outdict['mono'], outdict['quad'] = w[0], w[1]


	# perform bootstrap resampling of patches for uncertainty
	if nbootstrap > 0:
		# collect 2D and 3D CF realizations from N different resamplings
		w_realizations, xi_realizations = bootstrap_realizations(scales, nbootstrap, nthreads,
												coords1=coords1, randcoords1=randcoords1, weights1=weights1,
												randweights1=randweights1, coords2=coords2, randcoords2=randcoords2,
												weights2=weights2, randweights2=randweights2,
												oversample=oversample, pimax=pimax,
												estimator=estimator, mubins=mubins, wedges=wedges)
		# error estimate is variance of realizations
		w_realization_variance = np.std(w_realizations, axis=0)

		if coords1[2] is None:
			outdict['w_err'] = w_realization_variance
			outdict['covar'] = jackknife.covariance_matrix(np.array(w_realizations), np.array(outdict['w_theta']))
		else:
			xi_realization_variance = np.std(xi_realizations, axis=0)
			if mubins is None:
				outdict['wp_err'] = w_realization_variance
				outdict['xi_rp_pi_err'] = xi_realization_variance
				outdict['covar'] = jackknife.covariance_matrix(np.array(w_realizations), np.array(outdict['wp']))
			else:
				outdict['mono_err'], outdict['quad_err'] = w_realization_variance[0], w_realization_variance[1]

	return outdict

