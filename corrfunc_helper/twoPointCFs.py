
from Corrfunc.mocks.DDtheta_mocks import DDtheta_mocks
from Corrfunc.mocks.DDrppi_mocks import DDrppi_mocks
from . import estimators
import numpy as np
import healpy as hp
from . import jackknife
from functools import partial


# count autocorrelation pairs
def auto_counts(scales, coords, weights, nthreads=1, fulldict=True, pimax=40.):
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
		dd = DDrppi_mocks(1, 2, nthreads, pimax, scales, ras, decs, chis, weights1=weights,
							is_comoving_dist=True, weight_type=weight_type)
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
def cross_counts(scales, coords1, coords2, weights1, weights2, nthreads=1, fulldict=True, pimax=40.):
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
		dr = DDrppi_mocks(0, 2, nthreads, pimax, scales, ras1, decs1, chis1, weights1=weights1,
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
def counts_in_patch(patchval, patchmap, coords, randcoords, weights, randweights, scales, nthreads):
	nside = hp.npix2nside(len(patchmap))
	ras, decs, chis = coords
	randras, randdecs, randchis = randcoords
	# get only sources/randoms inside patch
	idxs_in_patch = np.where(patchmap[hp.ang2pix(nside, ras, decs, lonlat=True)] == patchval)
	randidxs_in_patch = np.where(patchmap[hp.ang2pix(nside, randras, randdecs, lonlat=True)] == patchval)

	# if weights given, sum up weights in patch
	if (weights is not None) and (randweights is not None):
		weights, randweights = weights[idxs_in_patch], randweights[randidxs_in_patch]
		n_data, n_rands = np.sum(weights), np.sum(randweights)
	else:
		n_data, n_rands = len(coords[0]), len(randcoords[0])

	# select only coordinates in patch
	if chis is None:
		coords = (ras[idxs_in_patch], decs[idxs_in_patch], None)
		randcoords = (randras[randidxs_in_patch], randdecs[randidxs_in_patch], None)
	else:
		coords = (ras[idxs_in_patch], decs[idxs_in_patch], chis[idxs_in_patch])
		randcoords = (randras[randidxs_in_patch], randdecs[randidxs_in_patch], randchis[randidxs_in_patch])

	# get raw pair counts
	ddcounts = auto_counts(scales, coords, weights, nthreads=nthreads, fulldict=False)
	drcounts = cross_counts(scales, coords, randcoords, weights, randweights, nthreads=nthreads, fulldict=False)
	rrcounts = auto_counts(scales, randcoords, randweights, nthreads=nthreads, fulldict=False)

	return [ddcounts, drcounts, rrcounts, n_data, n_rands]


#
def bootstrap_realizations(coords, randcoords, weights, randweights, scales, nbootstrap, nthreads,
							oversample=1, pimax=40., npatches=30):
	patchmap = jackknife.bin_on_sky(ras=coords[0], decs=coords[1], npatches=npatches)
	# get IDs for each patch
	unique_patchvals = np.unique(patchmap)
	# remove masked patches
	unique_patchvals = unique_patchvals[np.where(unique_patchvals > -1e30)]
	part_func = partial(counts_in_patch, patchmap=patchmap,
						coords=coords, randcoords=randcoords,
						weights=weights, randweights=randweights, scales=scales, nthreads=nthreads)
	# map cores to different patches, count pairs within
	counts = list(map(part_func, unique_patchvals))
	counts = np.array(counts)
	# separate out DD, DR, RR counts from returned array
	dd_counts, dr_counts, rr_counts = counts[:, 0], counts[:, 1], counts[:, 2]
	ndata, nrands = counts[:, 3], counts[:, 4]

	w_realizations = []
	# for each bootstrap, randomly select patches (oversampled by factor of N) and sum counts in those patches
	for k in range(nbootstrap):
		# randomly select patches with replacement
		boot_patches = np.random.choice(np.arange(len(unique_patchvals)), oversample * len(unique_patchvals))
		# sum pairs in those patches
		bootndata, bootnrands = ndata[boot_patches], nrands[boot_patches]
		totdata, totrands = np.sum(bootndata) / np.float(oversample), np.sum(bootnrands) / np.float(oversample)
		boot_ddcounts, boot_drcounts, bootrrcounts = dd_counts[boot_patches], dr_counts[boot_patches], \
													 rr_counts[boot_patches]
		totddcounts, totdrcounts, totrrcounts = np.sum(boot_ddcounts, axis=0), np.sum(boot_drcounts, axis=0), \
												np.sum(bootrrcounts, axis=0)
		if coords[2] is None:
			w_realizations.append(estimators.convert_raw_counts_to_cf(totdata, totdata, totrands,
																	totrands, totddcounts, totdrcounts, totdrcounts,
																	totrrcounts))
		else:
			w_realizations.append(estimators.convert_raw_counts_to_wp(totdata, totdata, totrands,
																	totrands, totddcounts, totdrcounts, totdrcounts,
																	totrrcounts, nrpbins=len(scales)-1, pimax=pimax))
	return w_realizations


# Measure an autocorrelation function. Coords should be either tuple of (ra, dec) or (ra, dec, chi)
def autocorr_from_coords(coords, randcoords, scales, weights=None, randweights=None,
						nthreads=1, randcounts=None, estimator='LS', pimax=40., dpi=1.,
						nbootstrap=0, oversample=1):

	if (weights is not None) and (randweights is not None):
		n_data, n_rands = np.sum(weights), np.sum(randweights)
	else:
		n_data, n_rands = len(coords[0]), len(randcoords[0])

	# autocorrelation of catalog
	DD_counts = auto_counts(scales, coords, weights=weights, nthreads=nthreads, pimax=pimax)

	# cross correlation between data and random catalog
	DR_counts = cross_counts(scales, coords, randcoords, weights1=weights, weights2=randweights,
								nthreads=nthreads, pimax=pimax)

	# can optionally send random counts from previous run as it
	if randcounts is None:
		# autocorrelation of random points
		RR_counts = auto_counts(scales, randcoords, weights=randweights, nthreads=nthreads, pimax=pimax)
	else:
		RR_counts = randcounts

	# angular correlation function
	if coords[2] is None:
		w = estimators.convert_counts_to_cf(n_data, n_data, n_rands, n_rands, DD_counts, DR_counts,
									DR_counts, RR_counts, estimator=estimator)
		poisson_err = np.sqrt(2 * np.square(1 + w) / DD_counts['npairs'])
	# spatial projected correlation function
	else:
		w = estimators.convert_counts_to_wp(n_data, n_data, n_rands, n_rands, DD_counts, DR_counts, DR_counts,
											RR_counts, nrpbins=len(scales)-1, pimax=pimax, dpi=dpi, estimator=estimator)
		poisson_err = None

	if nbootstrap > 0:
		w_realizations = bootstrap_realizations(coords, randcoords, weights, randweights, scales, nbootstrap, nthreads,
												oversample=oversample, pimax=pimax)
		bootstrap_err = np.std(w_realizations, axis=0)
		covar_matrix = jackknife.covariance_matrix(np.array(w_realizations), np.array(w))
	else:
		w_realizations, bootstrap_err, covar_matrix = None, None, None

	return w, poisson_err, bootstrap_err, covar_matrix


# angular cross correlation
def cross_corr_from_coords(coords, refcoords, randcoords, scales, refrandcoords=None, weights=None,
							refweights=None, randweights=None, refrandweights=None, nthreads=1, pimax=40.):

	if (refweights is not None) | (weights is not None) | (randweights is not None) | (refrandweights is not None):
		n_data, n_ref, n_rands, n_refrands = np.sum(weights), np.sum(refweights), np.sum(randweights), \
											np.sum(refrandweights)
	else:
		n_data, n_ref, n_rands, n_refrands = len(coords[0]), len(refcoords[0]), len(randcoords[0]), \
											len(refrandcoords[0])

	# count pairs between sample and control sample
	D1D2_counts = cross_counts(scales, coords, refcoords, weights1=weights,
								weights2=refweights, nthreads=nthreads)
	D2R1_counts = cross_counts(scales, refcoords, randcoords, weights1=refweights,
								weights2=randweights, nthreads=nthreads)

	# if there is a random catalog available for the reference population, we want the LS estimator
	if refrandcoords is not None:
		D1R2_counts = cross_counts(scales, coords, refrandcoords, weights1=weights,
						weights2=refrandweights, nthreads=nthreads)
		R1R2_counts = cross_counts(scales, randcoords, refrandcoords, weights1=randweights,
						weights2=refrandweights, nthreads=nthreads)
		# angular
		if coords[2] is None:
			w = estimators.convert_counts_to_cf(ND1=n_data, ND2=n_ref, NR1=n_rands, NR2=n_refrands,
												D1D2=D1D2_counts, D1R2=D1R2_counts, D2R1=D2R1_counts,
												R1R2=R1R2_counts, estimator='LS')
		# spatial
		else:
			w = estimators.convert_counts_to_wp(ND1=n_data, ND2=n_ref, NR1=n_rands, NR2=n_refrands, D1D2=D1D2_counts,
												D1R2=D1R2_counts, D2R1=D2R1_counts,
												R1R2=R1R2_counts, nrpbins=len(scales)-1, pimax=pimax, dpi=1.,
												estimator='LS')
	# no reference random catalog, use Peebles estimator
	else:
		# angular
		if coords[2] is None:
			w = estimators.convert_counts_to_cf(ND1=n_data, ND2=None, NR1=n_rands, NR2=None,
												D1D2=D1D2_counts, D1R2=D2R1_counts, D2R1=D2R1_counts,
												R1R2=D2R1_counts, estimator='Peebles')
		# spatial
		else:
			w = estimators.convert_counts_to_wp(ND1=n_data, ND2=None, NR1=n_rands, NR2=None, D1D2=D1D2_counts,
												D1R2=D2R1_counts, D2R1=D2R1_counts,
												R1R2=D2R1_counts, nrpbins=len(scales)-1, pimax=pimax, dpi=1.,
												estimator='Peebles')

	w_poisson_err = (1 + w) / np.sqrt(D1D2_counts['npairs'])

	return w, w_poisson_err


