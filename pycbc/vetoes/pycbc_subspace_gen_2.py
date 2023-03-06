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
from pycbc.filter.matchedfilter import match, overlap, matched_filter, sigmasq, overlap_cplx
from pycbc.vetoes.chisq import power_chisq, power_chisq_bins
import pycbc.psd
#from progressbar import ProgressBar
from tqdm import tqdm
import copy

GMbyc3=4.91657e-6

def generate_basis(Q_list, f0_list, hpt, psd, len_time, f_s, flow, singval):
    ''' This function has been written to generate basis for unified chi-square using provided list of 
    points in sine-gaussian parameter space. It returns an orthonormal basis with the correct number of 
    vectors corresponding to given percentage of singular values.
    Q_list, f0_list         -> List of sine-gaussian parameters to define the subspace
    hpt                     -> plus polarisation template of chirp waveform
    psd                     -> PSD of noise
    len_time                -> length of data in time-series considered
    f_s                     -> sampling frequency of timeseries
    flow                    -> Lower Frequency cut-off
    singval                 -> Percentage of singular values (0.0 < singval < 1.0)
    '''

    # Define a few relvant quantities
    N_ps = len(Q_list)
    fhigh = f_s/2.0
    len2 = 1 + int(len_time*fhigh)
    delta_f = 1.0/len_time
    klow = int(flow/delta_f)

    # Define cross polarisation template 
    hct = hpt*np.complex(0.0,1.0)

    clipped_white_sgs = []
    clipped_sgs=[]
    for k in tqdm(range(N_ps), ncols = 30):
        # Initialize sine-gaussian at t = 0 and normalise it.
        sine_gauss = sinegauss.fd_sine_gaussian(1.0, Q_list[k], f0_list[k], flow, fhigh + delta_f, delta_f)
        '''
        if k==1:
             
             plt.plot(sine_gauss.sample_frequencies, (sine_gauss.data), label = 'Sine-Gaussian')
             plt.legend()
             plt.show()
        '''
        sine_gauss.resize(len(hpt))
        # Calculate time-lag and shift the sine-gaussian back by the correct amount.
        corr = matched_filter(hpt, sine_gauss, psd, low_frequency_cutoff = flow, high_frequency_cutoff = fhigh)
        corr = abs(corr)
        index = np.argmax(corr)
        td = -1.0*corr.sample_times[index]
        #print("td is=",td)
        sine_gauss = pycbc.waveform.utils.apply_fd_time_shift(sine_gauss, td, copy = True)
        norm_sg = sigmasq(sine_gauss, psd, low_frequency_cutoff = flow, high_frequency_cutoff = fhigh)
        norm_hp = sigmasq(hpt, psd, low_frequency_cutoff = flow, high_frequency_cutoff = fhigh)
        norm_hc=sigmasq(hct, psd, low_frequency_cutoff = flow, high_frequency_cutoff = fhigh)
        sine_gauss=sine_gauss/np.sqrt(norm_sg)
        hpt=hpt/np.sqrt(norm_hp)
        hct=hct/np.sqrt(norm_hc)
        
        # Remove component of Sine-gaussian that is parallel to the templates and normalise it.
        match1 = overlap(hpt, sine_gauss, psd, low_frequency_cutoff = flow, high_frequency_cutoff = fhigh, normalized = False)
        match2 = overlap(hct, sine_gauss, psd, low_frequency_cutoff = flow, high_frequency_cutoff = fhigh, normalized = False)
        #print "matches=", match1, match2
        '''
        if k==1:
             plt.plot(hpt.sample_frequencies, hpt)
             plt.plot(sine_gauss.sample_frequencies,sine_gauss)
             
             plt.show()
        '''
        sine_gauss = sine_gauss  - match1*hpt - match2*hct
        
        norm = sigmasq(sine_gauss, psd, low_frequency_cutoff = flow, high_frequency_cutoff = fhigh)
        sine_gauss = sine_gauss/np.sqrt(norm)
        
        #plt.plot(sine_gauss.sample_frequencies,np.real(sine_gauss.data))
        
        #plt.plot(freq, np.real(sine_gauss.data), label = 'Clipped Sine-Gaussian')
        #plt.legend()
        #plt.show()
        clipped_sgs.append(sine_gauss)
        ## Compute inner product with template to see if the clipped sg is actually orthogonal.
        #match1 = overlap(hpt, sine_gauss, psd, low_frequency_cutoff = flow, high_frequency_cutoff = fhigh, normalized = False)
        #match2 = overlap(hct, sine_gauss, psd, low_frequency_cutoff = flow, high_frequency_cutoff = fhigh, normalized = False)
        #prod = math.sqrt(match1**2 + match2**2)
        #print(prod)
        leng=len(sine_gauss.data)
        # Whiten the clipped Sine-gaussian and making a two-sided array in FD.
        sine_gaussian = [(x*math.sqrt(2.0*delta_f/y)) for x, y in zip(sine_gauss.data[klow:(leng-200)], psd.data[klow:(leng-200)])]
        
        cart_sine_gauss = copy.deepcopy(sine_gaussian)
        sine_gaussian.reverse()
        cart_sine_gauss = cart_sine_gauss + [x for x in np.conj(sine_gaussian)]
    
        # Store the whitened, clipped Sine-gaussian and the time-lag in arrays.
        clipped_white_sgs.append(cart_sine_gauss)
        #timelags.append(lag)
    #plt.xlabel("f(Hz)")
    #plt.show() 
    # Compute SVD of the sine-gaussian vector matrix in order to get orthonormal basis in the subspace.
    # The columns in v are the orthonormal basis vectors of the subspace
    u, s, v = np.linalg.svd(clipped_white_sgs, full_matrices = False)

    #plt.plot(np.arange(1, N_ps+1, 1), s)
    #plt.yscale('log')
    #plt.show()

    # Now 'colour' the basis vectors and choose only the first N basis vectors that contain $singval percentage of 
    # singular values.
    zero_vec = [0.0 + 0.0j for x in range(klow)]
    basis = []
    total = sum([x**2 for x in s])
    contri = 0.0
    flag = 0
    #print('v is =', v)
    while (contri < singval):
         
        #print(s[flag])
        #print(total)
        #print(s[flag]**2/total) 
        contri = contri + s[flag]**2/total
    
        # Colour the basis vector.
        bas_vec = zero_vec + [x/math.sqrt(2.0*delta_f/y) for x, y in zip(v[flag][:len2-klow], psd[klow:])]
        bas_vec = pycbc.types.frequencyseries.FrequencySeries(bas_vec, delta_f = delta_f, copy = True)
        basis.append(bas_vec)
    
        flag = flag + 1
    
    return basis

def unified_chisq(data_vec,Q_list,f0_list,min_proj, hpt, psd, mass1, mass2, len_time, f_s, flow, singval,idx_opt):
    ''' This function computes the unified chi-square for the data vector given the template that clicked at a given time.
    This function returns the value of unified chi-square and number of degrees of freedom of chi-square.
    
    data_vec        ->      The segment of data that produced max SNR with template.
    hpt             ->      The template that gave max SNR.
    psd             ->      Power spectral density of noise.
    mass1, mass2    ->      Component masses of binary that gave max SNR.
    len_time                -> length of data in time-series considered
    f_s                     -> sampling frequency
    flow                    -> Lower Frequency cut-off
    singval                 -> Percentage of singular values (0.0 < singval < 1.0)
    '''
    fhigh = f_s/2.0
    len2 = 1 + int(len_time*fhigh)
    delta_f = 1.0/len_time


    # Generate the points on parameter space of sine-gaussians (using the uniform method) and then generate the subspace.
    #chirp_m = GMbyc3*(mass1*mass2)**(3.0/5.0)/(mass1 + mass2)**(1.0/5.0)    # Chirp mass of the binary in units of time
    #Q_val, N_ps, Q_list, f0_list = ps.parspace_sg_wrap('totalM_based', 2, min_proj, q_min, q_max, f0_min, f0_max, mass1, mass2, c2, GMbyc3, 'qf')
    #Q_val, N_ps, Q_list, f0_list = ps.parspace_sg_wrap('uniform', 2, 0.9, q_min, q_max, f0_min, f0_max, mass1, mass2, c2, GMbyc3, 'qf')
    basis = generate_basis(Q_list, f0_list, hpt, psd, len_time, f_s, flow, singval)
    Nb = len(basis)
    components=[0]*Nb
    # Calculate unified chi-square of data vector.
    snr= matched_filter(basis[0], data_vec, psd=psd, low_frequency_cutoff=flow,high_frequency_cutoff=fhigh, sigmasq=None)
    uni_chi2 =0.0 #np.zeros(len(snr))
    for k in range(Nb):
        norm = sigmasq(basis[k], psd, low_frequency_cutoff=flow, high_frequency_cutoff=fhigh)
        basis[k]=basis[k]/np.sqrt(norm)
        snr= matched_filter(basis[k], data_vec, psd=psd, low_frequency_cutoff=flow,high_frequency_cutoff=fhigh, sigmasq=None)
        #proj0 = overlap_cplx(basis[k], data_vec, psd, low_frequency_cutoff = flow, high_frequency_cutoff = fhigh, normalized = 	False)
        #projp = overlap_cplx(1.0j*basis[k], data_vec, psd, low_frequency_cutoff = flow, high_frequency_cutoff = fhigh, normalized = False)
        #print proj0,projp
        #uni_chi2 = uni_chi2 +np.abs(snr)**2#+ np.real(proj0)**2 +np.real(projp)**2
        components[k]=np.abs(snr[idx_opt])**2
    
    
    components.sort()
    if len(components)>=3:
	vec_num=3
    else:
	vec_num=len(components)
    print (Nb)
    print (len(components))
    print components
    for i in range (vec_num):
        inx=(Nb-vec_num)+i  	
 	uni_chi2=uni_chi2+components[inx]
           
    
    #print ("Nb and vec_num=", Nb, vec_num)    
    
    return uni_chi2, vec_num
