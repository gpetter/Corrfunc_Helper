import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import signal, interpolate


def gaussKern(size):
	"""
	Calculate a normalised Gaussian kernel to apply as a smoothing function

	Parameters
	size (int) : kernel size (how many points will be used in the smoothing operation)

	Returns
	g (array(size,size)) : normalised 2D kernel array for use in convolutions
	"""
	size = int(size)
	x, y = np.mgrid[-size:size + 1, -size:size + 1]
	g = np.exp(-(x ** 2 / float(size) + y ** 2 / float(size)))
	return g / g.sum()

def smooth(im, n=15):
	"""
	Smooth a 2D array im by convolving with a Gaussian kernel of size n

	Parameters
	im, n : 2D array, kernel size

	Returns
	improc : smoothed array (same dimensions as the input array)
	"""
	g = gaussKern(n)
	improc = signal.convolve2d(im, g, mode='same', boundary='symm')
	return (improc)


def plot_2d_corr_func(cf, setnegszero=True, inputrange=(None, None)):

	nbins = int(np.sqrt(len(cf)))
	cf2d = np.reshape(np.array(cf), (-1, nbins))

	logxi = np.log10(cf2d)
	qTR = logxi.copy()
	if setnegszero:
		qTR[~np.isfinite(qTR)] = 0
	qTR = qTR.T  # top right quadrant
	qTL = np.fliplr(qTR)  # top left quadrant
	qBL = np.flipud(qTL)  # bottom left quadrant
	qBR = np.fliplr(qBL)  # bottom right quadrant
	qT = np.hstack((qTL, qTR))  # top half
	qB = np.hstack((qBL, qBR))  # bottom half
	qq = np.vstack((qB, qT))  # full array
	if setnegszero:
		qqs = smooth(qq, n=8)  # smoothed full array
	else:
		qqs = qq

	fig, ax = plt.subplots(figsize=(8, 8))
	outputrange = (np.amin(qqs), np.amax(qqs))
	halfwidth = float(nbins)

	if setnegszero:
		# Get bin coordinates
		x,y = np.meshgrid(np.arange(-nbins, nbins+1), np.arange(-nbins, nbins+1))

		# Plot array
		pc = ax.pcolor(x,y,qqs,cmap='jet', vmin=inputrange[0], vmax=inputrange[1])
		# Plot contours
		lev = np.linspace(np.amin(qqs),np.amax(qqs),15)
		ax.contour(np.linspace(-halfwidth, halfwidth, 2*nbins),
				np.linspace(-halfwidth, halfwidth, 2*nbins),
				qqs, levels=lev, colors='k',
				linestyles='solid', linewidths=1)
	else:
		plotxs = np.arange(-nbins, nbins)
		pc = ax.contourf(plotxs, plotxs, qqs)

	"""nbins = int(np.sqrt(len(cf)))
	ara = np.arange(nbins)
	xx, yy = np.meshgrid(ara, ara, sparse=True)
	rs = np.sqrt(xx**2 + yy**2)
	cf2d = np.reshape(np.array(cf), (-1, nbins))
	bottomrightquad = np.flip(cf2d, axis=0)
	righthalf = np.vstack((bottomrightquad, cf2d))
	lefthalf = np.flip(righthalf, axis=1)
	full = np.hstack((lefthalf, righthalf))

	flipped = np.swapaxes(full, 0, 1)"""
	divider = make_axes_locatable(ax)
	cax = divider.append_axes('right', size='5%', pad=0.05)
	fig.colorbar(pc, cax=cax, orientation='vertical', label=r'log $\xi(r_{p}, \pi)$')

	ax.set_xlabel('$r_{p}$', fontsize=30)
	ax.set_ylabel('$\pi$', fontsize=30)

	return ax, outputrange

def xi_mu_s_plot(cf, nsbins, nmubins, inputrange=(None, None)):

	twodcf = np.reshape(cf, (nsbins, nmubins))
	logxi = np.log10(twodcf)
	qTR = logxi.copy()
	qTR[~np.isfinite(qTR)] = 0
	f = interpolate.interp2d(np.linspace(0, 1, nmubins), np.arange(nsbins), qTR)
	twodcf = np.fliplr(f(np.linspace(0, 1, nsbins), np.arange(nsbins)))
	nbins = len(twodcf)
	qqs = smooth(twodcf, n=8)
	outputrange = (np.amin(qqs), np.amax(qqs))
	fig, ax = plt.subplots(figsize=(8, 8))
	x, y = np.meshgrid(np.linspace(1, 0, nbins+1), np.arange(0, nbins+1))
	halfwidth = float(nbins)

	# Plot array
	pc = ax.pcolor(x, y, qqs, cmap='jet', vmin=inputrange[0], vmax=inputrange[1])
	# Plot contours
	lev = np.linspace(np.amin(qqs), np.amax(qqs), 15)
	ax.contour(np.linspace(-halfwidth, halfwidth, 2*nbins), np.linspace(-halfwidth, halfwidth, 2*nbins),
			qqs, levels=lev, colors='k', linestyles='solid', linewidths=1)
	ax.invert_xaxis()
	divider = make_axes_locatable(ax)
	cax = divider.append_axes('right', size='5%', pad=0.05)
	fig.colorbar(pc, cax=cax, orientation='vertical', label=r'log $\xi(\mu, s)$')

	ax.set_xlabel(r'$\mu$', fontsize=30)
	ax.set_ylabel('$s / (h^{-1}$ Mpc)', fontsize=30)

	return ax, outputrange


def w_theta_plot(cf):
	fig, ax = plt.subplots(figsize=(8, 6))
	ax.scatter(cf['theta'], cf['w_theta'], c='k')
	ax.errorbar(cf['theta'], cf['w_theta'], yerr=cf['w_err'], fmt='none', ecolor='k')
	baderrs = np.where((np.logical_not(np.isfinite(cf['w_err']))) | (cf['w_err'] == 0))[0]
	if len(baderrs) > 0:
		ax.errorbar(cf['theta'][baderrs], cf['w_theta'][baderrs],
					yerr=cf['w_poisson_err'][baderrs], fmt='none', ecolor='r')

	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_xlabel('$r_p$', fontsize=20)
	ax.set_ylabel('$w_p$', fontsize=20)
	plt.close()
	return fig

def wp_rp_plot(cf):
	fig, ax = plt.subplots(figsize=(8, 6))
	ax.scatter(cf['rp'], cf['wp'], c='k')
	ax.errorbar(cf['rp'], cf['wp'], yerr=cf['wp_err'], fmt='none', ecolor='k')
	baderrs = np.where((np.logical_not(np.isfinite(cf['wp_err']))) | (cf['wp_err'] == 0))[0]
	if len(baderrs) > 0:
		ax.errorbar(cf['rp'][baderrs], cf['wp'][baderrs],
					yerr=cf['wp_poisson_err'][baderrs], fmt='none', ecolor='r')

	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_xlabel('$r_p$', fontsize=20)
	ax.set_ylabel('$w_p$', fontsize=20)
	plt.close()
	return fig