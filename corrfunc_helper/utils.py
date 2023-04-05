import numpy as np


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

