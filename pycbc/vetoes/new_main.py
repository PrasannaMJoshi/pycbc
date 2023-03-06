#!/usr/bin/env python
# coding: utf-8

#'Author: Sunil'
import matplotlib
matplotlib.use("Agg")
import sys
import glob
import numpy as np
import pycbc
import par_space_2 as ps
import pycbc_subspace_gen_2 as psg
import AN_test as ant
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pycbc.psd as pp
from pycbc.waveform import get_fd_waveform, get_td_waveform
from pycbc.filter.matchedfilter import match, overlap, matched_filter, sigmasq, sigma
from pycbc.waveform import get_td_waveform, get_fd_waveform, get_fd_waveform_from_td
from pycbc.types import frequencyseries, timeseries
from pycbc.waveform import sinegauss
from pycbc.noise.gaussian import frequency_noise_from_psd
import pycbc.filter as p_filter
import pycbc.psd as pp
import pycbc.vetoes as vt
from pycbc.filter import highpass_fir, lowpass_fir
from math import pi
import sys
from scipy import interpolate
from scipy import signal
import copy
import random
from tqdm import tqdm
q_min = 2.0
q_max = 8.0
f0_min= 100.0
f0_max= 500.0
mass1 = 70.0
mass2 = 70.0
c2    = 1.0/((q_min+q_max)/2.0)**2
GMbyc3= 4.91657e-6
min_proj=0.8

#z_list1, y_list1, q_list1, f0_list1 = ps.parspace_sg_wrap('totalM_based', 1, min_proj, q_min, q_max, f0_min, f0_max, mass1, mass2, c2, GMbyc3, 'qf')
#sys.exit()

'''
q_min = 2.0
q_max = 8.0
f0_min= 300.0
f0_max= 500.0
'''
#z_list2, y_list2, q_list2, f0_list2 = ps.parspace_sg_wrap('totalM_based', 1, min_proj, q_min, q_max, f0_min, f0_max, mass1, mass2, c2, GMbyc3, 'qf')

#print('Total points=',len(y_list))
'''
plt.scatter(f0_list, q_list, marker='o', color='green')
plt.xlabel(r'$f_0$')
plt.ylabel('Q')
#plt.xlim(280.0,520.0)
#plt.ylim(2.0,11.0)
#plt.grid(which='major', color='#CCCCCC', linestyle='--')
plt.grid(which='major', color='grey', linestyle=':')
plt.show()
'''
len_time=16
f_s=4096
flow=20.0
fhigh=f_s/2.0
singval=0.9


hpt, _ =get_fd_waveform(approximant="IMRPhenomD", mass1 = mass1, mass2 = mass2, delta_f = 1.0/len_time, f_lower=20.0, f_final=2048.0)

#plt.plot(hpt.sample_frequencies,hpt)
#plt.show()


#psd = pp.aLIGOZeroDetHighPowerGWINC(len(hpt.data), 1.0/len_time , 10.0)
#f_noise = frequency_noise_from_psd(psd) 

bank=np.loadtxt("/home/sunil.choudhary/Bank_Mtotal_100_180.txt", delimiter=',')

catalog=np.loadtxt("/home/sunil.choudhary/SG_ChiSq_Proj/Blip_20_40_cat_3.txt",delimiter=',')

mass_bank_1=bank[:,0]
mass_bank_2=bank[:,1]
mass_cat_1=catalog[:,4]
mass_cat_2=catalog[:,5]

def scatter_hist(x1,x2,y1, y2, ax, ax_histx, ax_histy):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    
    # the scatter plot:
    ax.scatter(x1, x2, color='green', label='Template bank', marker='o',s=80.)
    ax.scatter(y1, y2, color='red',   label='Triggered templates', marker='.', s=70.)
    ax.set_xlabel(r'$M_1$',fontsize=20)
    ax.set_ylabel(r'$M_2$',fontsize=20)
    ax.tick_params(labelsize=15)
    ax.tick_params(labelsize=15)
    ax.legend(framealpha=0.5, fontsize=14)
    # now determine nice limits by hand:
    binwidth = 1.5
    #xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    #lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(10.0, 90.0 + binwidth, binwidth)
    bins2 = np.arange(5.0,50.0 + binwidth, binwidth)
    ax_histx.hist(y1, bins=bins, color='red')
    ax_histy.hist(y2, bins=bins2, color='red', orientation='horizontal')
'''
# definitions for the axes
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
spacing = 0.005


rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom + height + spacing, width, 0.2]
rect_histy = [left + width + spacing, bottom, 0.2, height]

# start with a square Figure
fig = plt.figure(figsize=(10, 10))

ax = fig.add_axes(rect_scatter)
ax_histx = fig.add_axes(rect_histx, sharex=ax)
ax_histy = fig.add_axes(rect_histy, sharey=ax)

# use the previously defined function
scatter_hist(mass_bank_1, mass_bank_2,mass_cat_1, mass_cat_2, ax, ax_histx, ax_histy)
plt.savefig('Bank_triggers.png')
sys.exit()

plt.figure(figsize=(15, 15))
plt.title('Template bank points',fontsize=20)
plt.scatter(mass_bank_1, mass_bank_2, color='green', label='Template bank', marker='.',s=80.)
#plt.scatter(mass_cat_1, mass_cat_2, color='red', label='triggered templates', marker='*', s=70.)
plt.xlabel(r'$M_1$',fontsize=20)
plt.ylabel(r'$M_2$',fontsize=20)
plt.loglog()
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.legend(framealpha=0.5, fontsize=20)
plt.savefig('Bank_SG_log.png')
#sys.exit()
'''

def BlipSNR(blip_f,df,psd=None):

    blip_snr=np.zeros(len(bank[:,0]))
    idx_list=np.zeros(len(bank[:,0]))
    for p in range (len(bank[:,0])):

        temp, _ = get_fd_waveform(approximant="IMRPhenomD", mass1 = bank[p,0], mass2 = bank[p,1], delta_f = df, f_lower=20.0, f_final=2048.0)

        #snr0,snrp=matched_filter(temp,blip_f,psd)
        #snr1=phase_maxed_snr(snr0,snrp)
        #idx=np.argmax(snr1)
        snr0=matched_filter(temp,blip_f,psd, low_frequency_cutoff=flow, high_frequency_cutoff=fhigh)
        snr= abs(snr0.data)
        idx=snr.argmax()
        blip_snr[p]=(snr[idx])
        idx_list[p]=idx


    max_idx = np.argmax(blip_snr)
    m1=bank[max_idx,0]
    m2=bank[max_idx,1]
    return blip_snr[max_idx],m1,m2

file_name2 ='/home/sunil.choudhary/ML_projects/sg_basis/test_gw_psd/H-H1_CLEANED_HOFT_C02_4KHZ_1169820555_64.txt'

chunk_for_psd= np.loadtxt(file_name2)
noise_PSD =chunk_for_psd[:,1]

#sim_psd = pp.aLIGOZeroDetHighPowerGWINC(len(hpt.data), 1.0/len_time , 10.0)
#sim_f_noise = frequency_noise_from_psd(sim_psd)

#Pxx, freqs = mlab.psd(noise_PSD, NFFT=2*f_s, Fs = f_s, noverlap = 1*f_s)

#f = interpolate.interp1d(freqs, Pxx)
#new_freq=np.linspace(np.min(freqs),np.max(freqs),(len(hpt)))
#print 'max_freq=', np.max(freqs)
#psd1=f(new_freq)
#psd=frequencyseries.FrequencySeries(psd1, delta_f=1.0/len_time)
file = glob.glob("/home/sunil.choudhary/GSPY_TestingBlips_O2/confidence_p60/Blip_*_16sec.txt")
file=np.sort(file)

hello=0
index=int(sys.argv[1])
noise_info=[]
for i in range (38):
    
     k=(index*38)+i 
     
     #sim_psd = pp.aLIGOZeroDetHighPowerGWINC(len(hpt.data), 1.0/len_time , 10.0)
     #sim_f_noise = frequency_noise_from_psd(sim_psd)

     print(file[k])
     blip_16K=np.loadtxt(file[k])
     #adj=np.loadtxt(files2[k])[:,1] 
     ##noise_chunk_for_SNR=chunk_for_psd[:,1]
     ##a=k*10*1024
     ##b=(k*10*1024)+(16*4096)
     ##test_noise=noise_PSD[a:b]
     skeleton_1 = timeseries.TimeSeries(blip_16K[:,1], delta_t=1.0/(16384))
     skeleton = highpass_fir(skeleton_1, 20, 8)
     skeleton = lowpass_fir(skeleton, 2000, 8)
     skeleton_resample=p_filter.resample.resample_to_delta_t(skeleton, 1.0/(f_s),method='butterworth' )
     #noise_PSD =skeleton_resample.data
     test_noise=skeleton_resample.data
     #skeleton_resample.data=test_noise 
     final_tdata=np.zeros(len(test_noise))
     window =signal.tukey(16*f_s, alpha=0.2)
     final_tdata[:16*f_s]=window*test_noise[int(0*4096):int(16*4096)]
     
     #inter_seg=np.zeros(16*f_s)
     #inter_seg[7*f_s:9*f_s]=test_noise
     tdata=timeseries.TimeSeries(final_tdata,delta_t=1.0/f_s)
     fdata=tdata.to_frequencyseries(delta_f=1.0/len_time)
     Pxx, freqs = mlab.psd(test_noise, NFFT=2*f_s, Fs = f_s, noverlap = 1*f_s)

     
     f = interpolate.interp1d(freqs, Pxx)
     new_freq=np.linspace(np.min(freqs),np.max(freqs),(len(hpt)))
     print ('max_freq=', np.max(freqs))
     psd1=f(new_freq)
     psd=frequencyseries.FrequencySeries(psd1, delta_f=1.0/len_time)
     
     fdata.data.resize(len(hpt))
     #waveform=copy.deepcopy(sine_gauss)
     
 
     #psd2 = pp.aLIGOZeroDetHighPowerGWINC(len(waveform), 1.0/len_time, 10.)
     snr, m1, m2 = BlipSNR(fdata,1.0/len_time,psd=psd)
     
     
     #snr_series=matched_filter(waveform, fdata, psd=psd, low_frequency_cutoff=flow,high_frequency_cutoff=fhigh, sigmasq=None)
     #opt_snr=np.abs(snr_series).max()
     #idx_opt=np.abs(snr_series).data.argmax()
     #print('index',idx_opt)
     #sg_snr, m1, m2 = BlipSNR(f_data_2,1.0/len_time,psd=psd)
     template, _ =get_fd_waveform(approximant="IMRPhenomD", mass1 = m1, mass2 = m2, delta_f = 1.0/len_time, f_lower=20.0, f_final=2048.0)
     snr_series=matched_filter(template, fdata, psd=psd, low_frequency_cutoff=flow,high_frequency_cutoff=fhigh, sigmasq=None)
     ''' 
     plt.figure(figsize=(10, 8))
     plt.plot(snr_series.sample_times,snr_series, color='blue', label='Matched filter time-series')
     plt.xlabel('Time (s)', fontsize=15)
     plt.ylabel('SNR', fontsize=15)
     plt.legend(fontsize=15)
     plt.savefig("/home/sunil.choudhary/Thesis_SNR_series.png")
     sys.exit()
     '''
     #bank_snr=np.abs(snr_series).max()
     idx_opt=np.abs(snr_series).data.argmax()
     #print('index',idx_opt)
     bins=16
     pchi_series=vt.chisq.power_chisq(template, fdata, num_bins=bins, psd=psd, low_frequency_cutoff=flow, high_frequency_cutoff=fhigh, return_bins=False)

     power_chi=pchi_series[idx_opt]
     dof_p=(bins*2-2)
     phase =0 #random.uniform(0.0,np.pi/2.0)
     print (phase)
     #data_vec=np.exp(1.0j*phase)*fdata #changing phase
     shift=0.0#idx_opt*(1.0/16)
     shifted_template=pycbc.waveform.utils.apply_fd_time_shift(template, shift, copy = True)
     
     z_list1, y_list1, q_list1, f0_list1 = ps.parspace_sg_wrap('totalM_based', 1, min_proj, q_min, q_max, f0_min, f0_max, m1, m2, c2, GMbyc3, 'qf')
     
     opt_chi,N_basis= psg.unified_chisq(fdata,q_list1, f0_list1, min_proj, template, psd, m1, m2, len_time, f_s, flow, singval,idx_opt)
     
     offsets=[15,30,45,60,75,90,105,120]
     print (len(offsets)) 
     sg_chisq = ant.sgchisq(template,fdata, psd, idx_opt, q=20.0, f0_offsets=offsets, f_low=20.0, f_high=2048.0)
     
     
     #opt_chi1=opt_chi_series1[idx_opt]

     #opt_chi_series2,N_basis2= psg.unified_chisq(fdata,q_list2, f0_list2, min_proj, shifted_template, psd, m1, m2, len_time, f_s, flow, singval,idx_opt)
     #opt_chi2=opt_chi_series1[idx_opt]

     #if (opt_chi2>opt_chi1):
     #   opt_chi=opt_chi2
     #   N_basis=N_basis2
     #else:
     #   opt_chi=opt_chi1
     #   N_basis=N_basis1
     
     dof_opt=(N_basis*2)
     dof_ant=(len(offsets)*2)
     #opt_chi=opt_chi_series[idx_opt]
     print("dof_opt=",dof_opt)
     print ("AN Chi sq is=", sg_chisq/dof_ant)

     print ("Opt chi per DOF is=", opt_chi/dof_opt)
     #print "max possible SNR=", opt_snr
     print ("Power chi per DOF is=", power_chi/dof_p)
     print ("bank snr=", snr)
     print (m1, m2)
     #print ("DOF=", dof_sg)
     #if (hello==0):
      
     file_real=open('/home/sunil.choudhary/SG_ChiSq_Proj/Review_Info_files/new_60_80M_Blip_%05d.txt'%(k),'w')

     file_real.write('Optimal_chisq, ')
     file_real.write('SG_chisq, ')
     file_real.write('Bank_SNR, ')
     file_real.write('Power_chisq ')
          
          

     file_real.write('\n')
          


     #file_real.write('%s, '%(file[k]))
     file_real.write('%s, '%(opt_chi/dof_opt))
     file_real.write('%s, '%(sg_chisq/dof_ant))
     file_real.write('%s, '%snr)
     file_real.write('%s, '%(power_chi/dof_p))
     #file_real.write('%s, %s '%(m1,m2))
     

     file_real.write('\n')
     
     #hello=hello+1 
     
