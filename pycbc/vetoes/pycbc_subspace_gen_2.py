import math
#from definitions import q_min, q_max, f0_min, f0_max, c2, GMbyc3
import cmath
import numpy as np
import matplotlib.pyplot as plt
import par_space_2 as ps
import pycbc
from pycbc.waveform import td_approximants, fd_approximants
from pycbc.waveform import get_td_waveform, get_fd_waveform, get_fd_waveform_from_td
from pycbc.waveform import sinegauss
from pycbc.filter.matchedfilter import match, overlap, matched_filter, sigmasq, overlap_cplx, sigma
from pycbc.vetoes.chisq import power_chisq, power_chisq_bins
import pycbc.psd
#from progressbar import ProgressBar
from tqdm import tqdm
import copy

GMbyc3=4.91657e-6

def generate_basis(Q_list, f0_list, hpt, psd, f_range, singval):
    ''' This function has been written to generate basis for unified chi-square using provided list of 
    points in sine-gaussian parameter space. It returns an orthonormal basis with the correct number of 
    vectors corresponding to given percentage of singular values.
    Q_list, f0_list         -> List of sine-gaussian parameters to define the subspace
    hpt                     -> plus polarisation template of chirp waveform
    psd                     -> PSD of noise
    f_range                 -> Frequency band
    singval                 -> Percentage of singular values (0.0 < singval < 1.0)
    '''

    # Define a few relvant quantities
    N_ps = len(Q_list)
    len2 = len(hpt)
    delta_f = hpt.get_delta_f()
    flow, fhigh = f_range
    klow = int(flow/delta_f)

    # Define cross polarisation template 
    hct = hpt*np.complex(0.0,1.0)

    sinegauss_matrix = np.empty(shape = (N_ps, 2*(len2 - klow),), dtype = np.complex64)
    for k in tqdm(range(N_ps), ncols = 30):
        # Initialize sine-gaussian at t = 0 and normalise it.
        sine_gauss = sinegauss.fd_sine_gaussian(1.0, Q_list[k], f0_list[k], flow, fhigh + delta_f, delta_f)
        sine_gauss.resize(len2)
        # Calculate time-lag and shift the sine-gaussian back by the correct amount.
        corr = matched_filter(hpt, sine_gauss, psd, low_frequency_cutoff = flow, high_frequency_cutoff = fhigh)
        index = np.argmax(abs(corr))
        #td = -1.0*corr.sample_times[index]
        sine_gauss = pycbc.waveform.utils.apply_fd_time_shift(sine_gauss, -1.0*corr.sample_times[index], copy = True)

        # Remove component of Sine-gaussian that is parallel to the templates and normalise it.
        ## FIXME: Can we get the match1 and match2 from the output of matched filter? In that case, we dont have to calculate it again.
        match1 = overlap(hpt, sine_gauss, psd, low_frequency_cutoff = flow, high_frequency_cutoff = fhigh, normalized = False)
        match2 = overlap(hct, sine_gauss, psd, low_frequency_cutoff = flow, high_frequency_cutoff = fhigh, normalized = False)
        sine_gauss = sine_gauss  - match1*hpt - match2*hct
        sine_gauss = sine_gauss/sigma(sine_gauss, psd, low_frequency_cutoff = flow, high_frequency_cutoff = fhigh)
        
        #leng=len(sine_gauss.data)
        # Whiten the clipped Sine-gaussian and making a two-sided array in FD.
        ## TODO: Why are you subtracting 200 from the leng variable?
        #sine_gaussian = [(x*math.sqrt(2.0*delta_f/y)) for x, y in zip(sine_gauss.data[klow:(leng-200)], psd.data[klow:(leng-200)])]
        scaled_sine_gaussian = np.sqrt(2.0*delta_f) * sine_gauss.numpy()[klow:len2]/ np.sqrt(psd.numpy()[klow:len2])
        sinegauss_matrix[k, :(len2 - klow)] = scaled_sine_gaussian.copy()
        sinegauss_matrix[k, (len2 - klow):] = np.conj(np.flip(scaled_sine_gaussian)).copy()
        
        #cart_sine_gauss = copy.deepcopy(sine_gaussian)
        #sine_gaussian.reverse()
        #cart_sine_gauss = cart_sine_gauss + [x for x in np.conj(sine_gaussian)]
    
        ## Store the whitened, clipped Sine-gaussian and the time-lag in arrays.
        #clipped_white_sgs.append(cart_sine_gauss)
    # Compute SVD of the sine-gaussian vector matrix in order to get orthonormal basis in the subspace.
    # The columns in v are the orthonormal basis vectors of the subspace
    u, s, v = np.linalg.svd(clipped_white_sgs, full_matrices = False)

    # Now 'colour' the basis vectors and choose only the first N basis vectors that contain $singval percentage of 
    # singular values.
    frac_s = s**2/np.sum(s**2)
    contri = 0.0
    Nbasis = 0
    while (contri < singval):
        contri = contri + frac_s[Nbasis]
        Nbasis = Nbasis + 1
    
    # Colour the basis vector.
    basis = np.zeroes(shape = (Nbasis, len2), dtype = np.complex64)
    basis[:, klow:] = v[:Nbasis,:len2-klow] * np.sqrt(psd[klow:]) / np.sqrt(2.0*delta_f) 
    #bas_vec = zero_vec + [x/math.sqrt(2.0*delta_f/y) for x, y in zip(v[flag][:len2-klow], psd[klow:])]
    #bas_vec = pycbc.types.frequencyseries.FrequencySeries(bas_vec, delta_f = delta_f, copy = True)
    #basis.append(bas_vec)
    return basis

def unified_chisq(hpt, data_vec, psd, mass1, mass2, min_proj, f_range, singval, idx_opt):
    ''' This function computes the unified chi-square for the data vector given the template that clicked at a given time.
    This function returns the value of unified chi-square and number of degrees of freedom of chi-square.
    
    hpt             ->      The template that gave max SNR.
    data_vec        ->      The segment of data that produced max SNR with template.
    psd             ->      Power spectral density of noise.
    mass1, mass2    ->      Component masses of binary that gave max SNR.
    f_range         ->      Frequency band
    singval         ->      Percentage of singular values (0.0 < singval < 1.0)
    idx_opt         ->      Index of the point with maximum SNR
    '''

    len2 = len(hpt)
    delta_f = hpt.get_delta_f()
    flow, fhigh = f_range
    GMbyc3= 4.91657e-6
    chirp_m = GMbyc3*(mass1*mass2)**(3.0/5.0)/(mass1 + mass2)**(1.0/5.0)
    q_min = 2.0
    q_max = 8.0
    f0_min= 100.0
    f0_max= 500.0
    ## FIXME: Check is template masses can be taken out of the template itself. 

    # Generate the points on parameter space of sine-gaussians (using the uniform method) and then generate the basis.
    z_list, y_list, Q_list, f0_list = ps.sample_parameter_space(min_proj, (q_min, q_max), (f0_min, f0_max), chirp_m)
    basis = generate_basis(Q_list, f0_list, hpt, psd, f_range, singval)
    Nb = len(basis)
    components=np.empty((Nb,))
    # Calculate unified chi-square of data vector.
    for k in range(Nb):
        basis_vector = pycbc.types.frequencyseries.FrequencySeries(basis[k], delta_f = delta_f, copy = True)
        basis_vector = basis_vector/sigma(basis_vector, psd, low_frequency_cutoff=flow, high_frequency_cutoff=fhigh)
        snr= matched_filter(basis_vector, data_vec, psd=psd, low_frequency_cutoff=flow,high_frequency_cutoff=fhigh, sigmasq=None)
        components[k] = abs(snr[idx_opt])**2
    components.sort(reverse = True)
    vec_num = min(len(components), 3)

    #print components
    #print ("Nb and vec_num=", Nb, vec_num)    
    return np.sum(components[:vec_num]), vec_num
