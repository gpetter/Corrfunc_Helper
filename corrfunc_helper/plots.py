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

def gausssmooth(im, n=15):
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


def plot_2d_corr_func(cf, setnegszero=True, inputrange=(None, None), smooth=8, cmap='jet',
					  ncontour=10, rps=None, pis=None):
	#implement calculating s^2 *xi(rp, pi), s = sqrt(rp^2+pi^2), so no need for log
	if rps is not None:
		s = np.sqrt(np.add.outer(np.square(rps), np.square(pis)))

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
	if setnegszero and (smooth > 0):
		qqs = gausssmooth(qq, n=smooth)  # smoothed full array
	else:
		qqs = qq

	fig, ax = plt.subplots(figsize=(8, 8))
	outputrange = (np.amin(qqs), np.amax(qqs))
	halfwidth = float(nbins)

	if setnegszero:
		# Get bin coordinates
		x,y = np.meshgrid(np.arange(-nbins, nbins+1), np.arange(-nbins, nbins+1))

		# Plot array
		pc = ax.pcolor(x,y,qqs,cmap=cmap, vmin=inputrange[0], vmax=inputrange[1])
		# Plot contours
		lev = np.linspace(np.amin(qqs),np.amax(qqs),ncontour)
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
	plt.close()

	return fig

def plot_s2_xi(cf, inputrange=(None, None), smooth=8, cmap='jet',
					  ncontour=10, rps=None, pis=None):
	#implement calculating s^2 *xi(rp, pi), s = sqrt(rp^2+pi^2), so no need for log
	rpgrid, pigrid = np.meshgrid(rps, pis)
	sgrid = np.sqrt(rpgrid ** 2 + pigrid ** 2)

	nbins = int(np.sqrt(len(cf)))
	cf2d = np.reshape(np.array(cf), (-1, nbins))

	qTR =  cf2d.copy() * sgrid ** 2
	qTR = qTR.T  # top right quadrant
	qTL = np.fliplr(qTR)  # top left quadrant
	qBL = np.flipud(qTL)  # bottom left quadrant
	qBR = np.fliplr(qBL)  # bottom right quadrant
	qT = np.hstack((qTL, qTR))  # top half
	qB = np.hstack((qBL, qBR))  # bottom half
	qq = np.vstack((qB, qT))  # full array
	if (smooth > 0):
		qqs = gausssmooth(qq, n=smooth)  # smoothed full array
	else:
		qqs = qq

	fig, ax = plt.subplots(figsize=(8, 8))
	outputrange = (np.amin(qqs), np.amax(qqs))
	halfwidth = float(nbins)

	# Get bin coordinates
	x,y = np.meshgrid(np.arange(-nbins, nbins+1), np.arange(-nbins, nbins+1))

	# Plot array
	pc = ax.pcolor(x,y,qqs,cmap=cmap, vmin=inputrange[0], vmax=inputrange[1])
	# Plot contours
	lev = np.linspace(np.amin(qqs),np.amax(qqs),ncontour)
	ax.contour(np.linspace(-halfwidth, halfwidth, 2*nbins),
			np.linspace(-halfwidth, halfwidth, 2*nbins),
			qqs, levels=lev, colors='k',
			linestyles='solid', linewidths=1)

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
	fig.colorbar(pc, cax=cax, orientation='vertical', label=r'$s^2  \xi(r_{p}, \pi)$')

	ax.set_xlabel('$r_{p}$', fontsize=30)
	ax.set_ylabel('$\pi$', fontsize=30)
	plt.close()

	return fig

def plotmultiple_2d_corr_func(cfs, setnegszero=True, inputrange=(None, None), smooth=8, cmap='jet', ncontour=10):
	if len(cfs) > 5:
		cfs = [cfs]
	fig, ax = plt.subplots(figsize=(len(cfs) * 8, 8), ncols=len(cfs))

	smoothcf = []
	for j in range(len(cfs)):
		thiscf = cfs[j]
		nbins = int(np.sqrt(len(thiscf)))
		cf2d = np.reshape(np.array(thiscf), (-1, nbins))

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
		if setnegszero and (smooth > 0):
			qqs = gausssmooth(qq, n=smooth)  # smoothed full array
		else:
			qqs = qq
		smoothcf.append(qqs)
	maxval, minval = np.max(smoothcf), np.min(smoothcf)

	outputrange = (minval, maxval)
	halfwidth = float(nbins)

	for j in range(len(cfs)):
		if len(cfs) == 1:
			thisax = ax
		else:
			thisax = ax[j]

		if setnegszero:
			# Get bin coordinates
			x, y = np.meshgrid(np.arange(-nbins, nbins + 1), np.arange(-nbins, nbins + 1))

			# Plot array
			pc = thisax.pcolor(x, y, smoothcf[j], cmap=cmap, vmin=inputrange[0], vmax=inputrange[1])
			# Plot contours
			lev = np.linspace(np.amin(smoothcf[j]), np.amax(qqs), ncontour)
			thisax.contour(np.linspace(-halfwidth, halfwidth, 2 * nbins),
						   np.linspace(-halfwidth, halfwidth, 2 * nbins),
						   smoothcf[j], levels=lev, colors='k',
						   linestyles='solid', linewidths=1)
		else:
			plotxs = np.arange(-nbins, nbins)
			pc = ax.contourf(plotxs, plotxs, smoothcf[j])
		thisax.set_xlabel('$r_{p}$', fontsize=30)
		thisax.set_ylabel('$\pi$', fontsize=30)

	divider = make_axes_locatable(thisax)
	cax = divider.append_axes('right', size='5%', pad=0.05)
	fig.colorbar(pc, cax=cax, orientation='vertical', label=r'log $\xi(r_{p}, \pi)$')

	plt.close()

	return fig

def xi_mu_s_plot(cf, nsbins, nmubins, inputrange=(None, None)):

	twodcf = np.reshape(cf, (nsbins, nmubins))
	logxi = np.log10(twodcf)
	qTR = logxi.copy()
	qTR[~np.isfinite(qTR)] = 0
	f = interpolate.interp2d(np.linspace(0, 1, nmubins), np.arange(nsbins), qTR)
	twodcf = np.fliplr(f(np.linspace(0, 1, nsbins), np.arange(nsbins)))
	nbins = len(twodcf)
	qqs = gausssmooth(twodcf, n=8)
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
	try:
		ax.errorbar(cf['theta'], cf['w_theta'], yerr=cf['w_err'], fmt='none', ecolor='k')
	except:
		ax.errorbar(cf['theta'], cf['w_theta'], yerr=cf['w_err_poisson'], fmt='none', ecolor='k')
	#baderrs = np.where((np.logical_not(np.isfinite(cf['w_err']))) | (cf['w_err'] == 0))[0]
	#if len(baderrs) > 0:
	#	ax.errorbar(cf['theta'][baderrs], cf['w_theta'][baderrs],
	#				yerr=cf['w_poisson_err'][baderrs], fmt='none', ecolor='r')

	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_xlabel(r'$\theta$ [deg]', fontsize=20)
	ax.set_ylabel(r'$w(\theta)$', fontsize=20)
	plt.close()
	return fig

def wp_rp_plot(cf):
	fig, ax = plt.subplots(figsize=(8, 6))
	ax.scatter(cf['rp'], cf['wp'], c='k')
	try:
		ax.errorbar(cf['rp'], cf['wp'], yerr=cf['wp_err'], fmt='none', ecolor='k')
	except:
		ax.errorbar(cf['rp'], cf['wp'], yerr=cf['wp_poisson_err'], fmt='none', ecolor='k')
	#baderrs = np.where((np.logical_not(np.isfinite(cf['wp_err']))) | (cf['wp_err'] == 0))[0]
	#if len(baderrs) > 0:
	#	ax.errorbar(cf['rp'][baderrs], cf['wp'][baderrs],
	#				yerr=cf['wp_poisson_err'][baderrs], fmt='none', ecolor='r')

	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_xlabel('$r_p \ [\mathrm{Mpc}/h]$', fontsize=20)
	ax.set_ylabel('$w_p(r_p)$', fontsize=20)
	plt.close()
	return fig

def xi_s_plot(cf):
	fig, ax = plt.subplots(figsize=(8, 6))
	if len(np.shape(cf['mono'])) > 1:
		for j in range(len(cf['mono'])):
			ax.scatter(cf['s'], cf['mono'][j])
			ax.errorbar(cf['s'], cf['mono'][j], yerr=cf['mono_err'][j], fmt='none')
	else:
		ax.scatter(cf['s'], cf['mono'], c='k')
		ax.errorbar(cf['s'], cf['mono'], yerr=cf['mono_err'], fmt='none', ecolor='k')

	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_xlabel('$s \ [\mathrm{Mpc}/h]$', fontsize=20)
	ax.set_ylabel(r'$\xi_{0}(s)$', fontsize=20)
	plt.close()
	return fig


def cf_plot(cf):
	if 'theta' in cf:
		return w_theta_plot(cf)
	elif 'mono' in cf:
		return xi_s_plot(cf)
	else:
		return wp_rp_plot(cf)