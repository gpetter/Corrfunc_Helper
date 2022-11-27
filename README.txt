Wrapper for Corrfunc (https://github.com/manodeep/Corrfunc) codes to calculate statistics in one line

Calculate angular clustering in 10 logarithmic bins from 10^-2 to 10^0 degrees
twoPointCFs.autocorr_from_coords((ras, decs), (randras, randdecs), scales=np.logspace(-2, 0, 10), weights=weights, randweights=randweights, nbootstrap=500)

Calculate projected 3D clustering wp(rp) in 10 bins of rp from 1 to ~30 Mpc/h with pi_max = 40 Mpc/h
twoPointCFs.autocorr_from_coords((ras, decs, dists), (randras, randdecs, randdists), scales=np.logspace(0, 1.5, 10), weights=weights, randweights=randweights, pimax=40, nbootstrap=500)

Calculate redshift-space clustering xi(s), monopole and quadrupole
twoPointCFs.autocorr_from_coords((ras, decs, dists), (randras, randdecs, randdists), scales=np.logspace(0, 1.5, 10), weights=weights, randweights=randweights, mubins=10, nbootstrap=500)
