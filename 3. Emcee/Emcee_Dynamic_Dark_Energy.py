import healpy as hp
import pyccl as ccl
import numpy as np
import matplotlib.pyplot as plt
import itertools
import time
import emcee
from pathos.multiprocessing import ProcessingPool as Pool
import os
from scipy import stats

def default_nz(z):
    """
    Basic Galaxy distribution used for populating the Universe.
    We use a simple Gaussian centered at 0.5 with standard deviation 0.1
    """
    return np.exp(-0.5 * ((z - 0.5) / 0.1) ** 2)

def default_b1(z):
    """
    Multipkication factor to describe the relationship between matter and galaxies. 
    Here, we assume the galaxy distribution exactly describes the matter density field.
    """
    return np.ones_like(z)

class CosmologicalModel:
    """
    This class populates the cosmology that will be used for further analysis.
    The populated cosmology can be based on lambda CDM or dynamic dark energy depending on the 
    parameter choices.

    We use galaxy clustering and weak lensing and their angular power spectra (individual & cross correlation) 
    for the analysis.
    """
    def __init__(self, z_min=0, z_max=1, z_step=256, max_ell=1000, omega_c=0.25, omega_b=0.05,
                 h=0.67, a_s=2.1e-9, n_s=0.96, m_nu=0.06, w0=-1, wa=0):
        self.z_min = z_min
        self.z_max = z_max
        self.z_step = z_step
        self.max_ell = max_ell
        self.z = np.linspace(self.z_min, self.z_max, self.z_step)
        self.omega_c = omega_c
        self.omega_b = omega_b
        self.h = h
        self.a_s = a_s
        self.n_s = n_s
        self.m_nu = m_nu
        self.w0 = w0
        self.wa = wa
        self.ells = np.arange(0, self.max_ell, 1)
        self.model = ccl.Cosmology(Omega_c=omega_c, Omega_b=omega_b, h=h, A_s=a_s, 
                                   n_s=n_s, m_nu=m_nu, w0=w0, wa=wa, mass_split="single", 
                                   transfer_function="boltzmann_camb", 
                                   extra_parameters={"camb": {"dark_energy_model": "ppf"}})

    def generate_galaxy_clustering(self, nz_fn=default_nz, b1_fn=default_b1):
        nz = nz_fn(self.z)
        b1 = b1_fn(self.z)
        return ccl.NumberCountsTracer(self.model, has_rsd=False, dndz=(self.z, nz), bias=(self.z, b1))

    def generate_lensing(self, z_source=1100):
        return ccl.CMBLensingTracer(self.model, z_source=z_source)

    def generate_angular_ps(self, nz_fn=default_nz, b1_fn=default_b1, z_source=1100, output_type="all", is_plot=True):
        galaxy_dist = self.generate_galaxy_clustering(nz_fn, b1_fn)
        lensing_dist = self.generate_lensing(z_source=z_source)
        cgg = ccl.angular_cl(self.model, galaxy_dist, galaxy_dist, self.ells)
        ckk = ccl.angular_cl(self.model, lensing_dist, lensing_dist, self.ells)
        ckg = ccl.angular_cl(self.model, lensing_dist, galaxy_dist, self.ells)
        if output_type == "cgg":
            if is_plot:
                plt.loglog(self.ells, cgg, label="Galaxy-Galaxy")
                plt.xlabel("Angular Scale")
                plt.ylabel("Spectrum Density")
                plt.show()
            return cgg
        elif output_type == "ckg":
            if is_plot:
                plt.loglog(self.ells, ckg, label="Lensing-Galaxy")
                plt.xlabel("Angular Scale")
                plt.ylabel("Spectrum Density")
                plt.show()
            return ckg
        elif output_type == "ckk":
            if is_plot:
                plt.loglog(self.ells, ckk, label="Lensing-Lensing")
                plt.xlabel("Angular Scale")
                plt.ylabel("Spectrum Density")
                plt.show()
            return ckk
        else:
            if is_plot:
                plt.loglog(self.ells, ckg, label="Lensing-Galaxy")
                plt.loglog(self.ells, cgg, label="Galaxy-Galaxy")
                plt.loglog(self.ells, ckk, label="Lensing-Lensing")
                plt.xlabel("Angular Scale")
                plt.ylabel("Spectrum Density")
                plt.legend(loc="best")
                plt.show()
            return cgg, ckg, ckk

    def generate_survey_map(self, nz_fn=default_nz, b1_fn=default_b1, z_source=1100, nside=1024, pol=False, is_plot=True):
        np.random.seed(998905029)
        cl = self.generate_angular_ps(nz_fn=nz_fn, b1_fn=b1_fn, z_source=z_source, is_plot=False)
        delta_g, delta_k = hp.synfast(list(cl), nside=nside, pol=pol)
        if is_plot:
            hp.mollview(delta_g, cmap='seismic', min=-.1, max=.1, title="Galaxy Overdensity Field")
            hp.mollview(delta_k, cmap='seismic', min=-.1, max=.1, title="Lensing Overdensity Field")
            plt.show()
        return delta_g, delta_k
    

def get_unique_run_number(base_name):
    """
    To give unique names to the file saved
    """
    run_number = 1
    while os.path.exists(f"{base_name}_run_{run_number}.npy"):
        run_number += 1
    return run_number


def main():
    # Measure the time took for the execution of the code
    start_time = time.time()

    # Due to the limitation of the pathos multiprocessing, we host the functions in one function

    def log_posterior(params):
        """
        Define log posterior. This involves calculation of log prior, and likelihood.
        """
        stacked_ps = np.load("stacked_ps_6_param_dark.npy", allow_pickle=True) # import the power spectra data
        cov = np.load("cov_6_param_dark.npy", allow_pickle=True) # import the covalence matrix
        min_ell = 75
        max_ell = 1000
        n = 10

        def log_prior(params):
            """
            Define log prior. We refer the data from DESI and Planck 2018 results to set up the reasonable priors.
            """
            omega_c, n_s, lnas, m_nu, w0, wa = params
            # prior belief
            if 0.1 < omega_c < 0.9 and 0.8 < n_s < 1.2 and 1.61 < lnas < 3.91 and 0.06 < m_nu < 0.8 and -2 < w0 < 0.5 and -2 < wa < 2 and w0 + wa < 0:
                return 0.0
            return -np.inf

        def log_likelihood(params):
            """
            Log likelihood calculation. 

            Notice we use fixed values for h and omega_b.
            """
            h = 0.6766
            omega_b = 0.02218 / (h**2)
            omega_c, n_s, lnas, m_nu, w0, wa = params
            a_s = np.exp(lnas) / (10**10)
            model = CosmologicalModel(h=h, omega_b=omega_b, omega_c=omega_c, n_s=n_s, a_s=a_s, m_nu=m_nu, w0=w0, wa=wa, max_ell=max_ell)
            cgg, ckg, ckk = model.generate_angular_ps(is_plot=False)
            _, cgg_bins, ckg_bins, ckk_bins = create_bin(min_ell, max_ell, n + 1, cgg[min_ell:], ckg[min_ell:], ckk[min_ell:])
            mu = np.hstack((cgg_bins, ckg_bins, ckk_bins)).flatten()
            diff = stacked_ps - mu
            diff = diff.reshape((1, 30))
            return -0.5 * np.dot(diff, np.dot(np.linalg.inv(cov), diff.T)).item()

        def create_bin(min_ell, max_ell, n, cgg, ckg, ckk):
            """
            Groups the range of multipoles to bins from the angular power spectra.
            """
            x = np.arange(min_ell, max_ell, 1)
            bin_edges = np.linspace(np.min(x), np.max(x), n)
            bin_centers, galaxy_galaxy_bin = bin_mat(x, np.array(cgg), bin_edges)
            _, lensing_galaxy_bin = bin_mat(x, np.array(ckg), bin_edges)
            _, lensing_lensing_bin = bin_mat(x, np.array(ckk), bin_edges)
            return bin_centers, galaxy_galaxy_bin, lensing_galaxy_bin,lensing_lensing_bin

        def bin_mat(r=[],mat=[],r_bins=[]):
            """
            Divides the range of l using the edges provided from r_bins.
            Returns the center of each bins and matrix of binned data
            """
            bin_center=0.5*(r_bins[1:]+r_bins[:-1])
            n_bins=len(bin_center)
            ndim=len(mat.shape)
            mat_int=np.zeros([n_bins]*ndim,dtype='float64')
            norm_int=np.zeros([n_bins]*ndim,dtype='float64')
            bin_idx=np.digitize(r,r_bins)-1
            r2=np.sort(np.unique(np.append(r,r_bins))) #this takes care of problems around bin edges
            dr=np.gradient(r2)
            r2_idx=[i for i in np.arange(len(r2)) if r2[i] in r]
            dr=dr[r2_idx]
            r_dr=r*dr

            ls=['i','j','k','l']
            s1=ls[0]
            s2=ls[0]
            r_dr_m=r_dr
            for i in np.arange(ndim-1):
                s1=s2+','+ls[i+1]
                s2+=ls[i+1]
                r_dr_m=np.einsum(s1+'->'+s2,r_dr_m,r_dr)#works ok for 2-d case

            mat_r_dr=mat*r_dr_m
            for indxs in itertools.product(np.arange(min(bin_idx),n_bins),repeat=ndim):
                # x={} #np.zeros_like(mat_r_dr,dtype='bool')
                norm_ijk=1
                mat_t=[]
                for nd in np.arange(ndim):
                    slc = [slice(None)] * (ndim)
                    #x[nd]=bin_idx==indxs[nd]
                    slc[nd]=bin_idx==indxs[nd]
                    if nd==0:
                        mat_t=mat_r_dr[slc[0]]
                    else:
                        mat_t=mat_t[slc[0]]
                    norm_ijk*=np.sum(r_dr[slc[nd]])
                if norm_ijk==0:
                    continue
                mat_int[indxs]=np.sum(mat_t)/norm_ijk
                norm_int[indxs]=norm_ijk
            return bin_center,mat_int

        lp = log_prior(params)
        if not np.isfinite(lp):
            return -np.inf, None, None
        ll = log_likelihood(params)
        if not np.isfinite(ll):
            return -np.inf, None, None
        h = 0.6766
        omega_b = 0.02218 / (h**2)
        omega_c, n_s, lnas, m_nu, w0, wa = params
        a_s = np.exp(lnas) / (10**10)
        cosmo_model = CosmologicalModel(h=h, omega_b=omega_b, omega_c=omega_c, n_s=n_s, a_s=a_s, m_nu=m_nu, w0=w0, wa=wa, max_ell=max_ell)
        return lp + ll, lp, cosmo_model.model.sigma8()

    # set up for the parallelization
    ndim = 6  # number of parameters
    nwalkers = 64  # adjusted number of walkers
    nsteps = 4500  # adjusted number of steps
    burn_in_iteration = 2 # burn-in iteration
    each_burn_in_step = 500 # each burn in step

    # intial position set up
    pos = np.random.uniform(low=[0.1, 0.8, 1.61, 0.06, -2, -2], high=[0.9, 1.2, 3.91, 0.8, 0.5, 2], size=(nwalkers, ndim))
    dtype = [("log_prior", float), ("sigma_8", float)]

    # utilize Emcee to conduct Bayesian sampling
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, blobs_dtype=dtype, pool=pool)
        for l in range(burn_in_iteration):
            sampler.run_mcmc(pos, each_burn_in_step, progress=True)
            print(f"{l}th burn-in finished for pool {pool}")
            log_probs = sampler.get_log_prob()[-1, :] # (steps, nwalker) thus pick the last step
            samples = sampler.get_chain() # (steps, nwalker, ndim)
            pos = samples[-1, :, :] # last step (nwalker, ndim)
            sorted_indices = np.argsort(log_probs)[::-1] # descending sort according to nwalkers
            top_half_indices = sorted_indices[:nwalkers // 2] # top half
            bottom_half_indices = sorted_indices[nwalkers // 2:] # bottom half

            # reset bottom half positions to be near the top half positions
            for i in bottom_half_indices:
                # choose a random walker from the top half
                j = np.random.choice(top_half_indices)

                for k in range(len(samples[0, 0, :])):
                    temp_samples = samples[250 * (l + 1) + (250 * l):, j, k]
                    new_mean = np.mean(temp_samples)
                    new_sd = np.std(temp_samples)
                    # choose from normal distribution
                    pos[i, k] = stats.norm.rvs(loc=new_mean, scale=new_sd, size=1)  
        sampler.reset()
        sampler.run_mcmc(pos, nsteps, progress=True)

    end_time = time.time()
    print(f"It took {round((end_time - start_time)/3600, 2)} hours to run")

    # store results
    base_name = "chain_results_6_64_1000_4500"
    blob_name = "blob_results_6_64_1000_4500"
    base_name_prob = "prob_results_6_64_1000_4500"
    run_number = get_unique_run_number(base_name)
    chain_filename = f'{base_name}_run_{run_number}.npy'
    blob_filename = f"{blob_name}_run_{run_number}.npy"
    prob_filename = f"{base_name_prob}_run_{run_number}.npy"
    autocorr_filename = f'autocorrelation_time_run_{run_number}.npy'

    samples = sampler.get_chain(discard=500)
    blobs = sampler.get_blobs(discard=500)
    log_probs = sampler.get_log_prob(discard=500)

    np.save(chain_filename, samples)
    np.save(blob_filename, blobs)
    np.save(prob_filename, log_probs)

    try:
        tau = sampler.get_autocorr_time()
        print("Autocorrelation time:", tau)
        np.save(autocorr_filename, tau)
    except emcee.autocorr.AutocorrError as e:
        print("Error estimating autocorrelation time:", e)

if __name__ == "__main__":
    main()
