# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 18:17:29 2019

@author: JIAN

This file implement the visualization technique proposed in the paper.
The extracted dataset is available from:
    https://figshare.com/articles/dataset/EEG_driver_drowsiness_dataset/14273687
if you have any problem on using the codes, pls contact Dr. Cui Jian at cuij0006@ntu.edu.sg
"""
import torch
import scipy.io as sio
import numpy as np
import torch.optim as optim
from scipy.integrate import simps
from mne.time_frequency import psd_array_multitaper,psd_array_welch,tfr_array_morlet,stft
import mne
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.gridspec as gridspec
#from InterpretableCNN import InterpretableCNN


import os
import pickle
import json
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import utils
from data.data_utils import *
from data.dataloader_detection import load_dataset_detection
from data.data_loader_drowsiness import load_dataset_classification
from constants import *
from args import get_args
from collections import OrderedDict
from json import dumps
#from model.model_ori import DCRNNModel_classification
from model.model import DCRNNModel_classification
from CNNLSTM import CNNLSTM
from tensorboardX import SummaryWriter
from tqdm import tqdm
from dotted_dict import DottedDict
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy
from multiprocessing import Pool
from sklearn.metrics import precision_recall_curve, accuracy_score, roc_auc_score
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.cm as cmx
import matplotlib.colors as colors

def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv') 
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color

def plot_latent_space(x_batch, y_batch, iteration=None, dim=2):
    
    model = TSNE(n_components=dim, random_state=0, perplexity=50, learning_rate=500, n_iter=200)
    z_mu = model.fit_transform(mu.eval(feed_dict={X: x_batch}))
    n_classes = len(list(set(np.argmax(y_batch, 1))))
    cmap = get_cmap(n_classes)
    fig = plt.figure(2, figsize=(8,8))
    
    if dim == 3:
        for i in list(set(np.argmax(y_batch, 1))):
            bx = fig.add_subplot(111, projection='3d')

            index = np.where(np.argmax(y_batch, 1) == i)
            xs = z_mu[index, 0]
            ys = z_mu[index, 1:]
            zs = z_mu[index, 2]
            bx.scatter(xs, ys, zs,c=cmap(i), label=str(i))
    else:
        for i in list(set(np.argmax(y_batch, 1))):
            bx = fig.add_subplot(111)
            index = np.where(np.argmax(y_batch, 1) == i)
            xs = z_mu[index, 0]
            ys = z_mu[index, 1]
            bx.scatter(xs, ys, c=cmap(i), label=str(i))

    bx.set_xlabel('X Label')
    bx.set_ylabel('Y Label')
    bx.legend()
    bx.set_title('Truth')
    if iteration is None:
        plt.savefig('latent_space.png')
    else:
        plt.savefig('latent_space' + str(iteration) + '.png')
    plt.show()
    
torch.cuda.empty_cache()
torch.manual_seed(0)

plt.rcParams.update({'font.size': 14})

class VisTech():
    def __init__(self, model):
        self.model = model
        self.model.eval()              
        
              
    def generate_heatmap(self, batchInput,occlude_map,sampleidx,subid,samplelabel,likelihood,adj):
        """
        This function generates figures shown in the figure        
        input:
           batchInput:          all the samples in a batch for classification
           sampleidx:           the index of the sample
           subid:               the ID of the subject
           samplelabel:         the ground truth label of the sample
           likelihood:          the likelihood of the sample to be classified into alert and drowsy state 
        """        
        
        #if likelihood[0]>likelihood[1]:
        #    state=0
        #else:
        #    state=1

        if samplelabel==1:
            labelstr='alert'
        else:
            labelstr='drowsy'        
        
        
        #sampleInput=batchInput[sampleidx].cpu().detach().numpy().squeeze()
        #occlude_map=occlude_map[sampleidx].cpu().detach().numpy().squeeze()
        sampleInput=batchInput[sampleidx]
        occlude_map=occlude_map[sampleidx]
        sampleChannel=sampleInput.shape[0] 
        sampleLength=sampleInput.shape[1]  
        #import pdb; pdb.set_trace()
        #mean_occlude_map = np.mean(occlude_map)
        #std_occlude_map = np.std(occlude_map)  
        #occlude_map = np.clip(occlude_map, mean_occlude_map-3*std_occlude_map, mean_occlude_map+3*std_occlude_map)
        #print((occlude_map))
        #occlude_map = (occlude_map)/(np.std(occlude_map)+0.00001)
        #occlude_map = (occlude_map-np.mean(occlude_map))/(np.std(occlude_map)+0.00001)
        #import pdb; pdb.set_trace()
        #for ii in range(6):
        #    if np.mean(occlude_map[:,ii])<0.5:
        #        occlude_map[:,ii]=-1
        #occlude_map[occlude_map<0.5] = -1
        
        heatmap = np.zeros((sampleInput.shape),dtype=np.float32)
        #import pdb; pdb.set_trace()
        for ii in range(10):
            for channel in range(30):
                heatmap[channel,64*ii:64*(ii+1)] = (occlude_map[channel,ii]-occlude_map.min())/(occlude_map.max()-occlude_map.min())

        #heatmap=self.heatmap_calculation(batchInput=batchInput,sampleidx=sampleidx,state=state)
        
        fig = plt.figure(figsize=(24,7))
        
        gridlayout = gridspec.GridSpec(ncols=6, nrows=2, figure=fig,wspace=0.05, hspace=0.005)

        axs0 = fig.add_subplot(gridlayout[0:2,1:4])
        axs1 = fig.add_subplot(gridlayout[0:2,4:6])  
        #axs21= fig.add_subplot(gridlayout[0,4]) 
        #axs22= fig.add_subplot(gridlayout[0,5]) 
        #axs23= fig.add_subplot(gridlayout[1,4]) 
        #axs24= fig.add_subplot(gridlayout[1,5])        

        #fig.title('Drowsiness explainable result')),y=1.02)  
        fig.suptitle('Subject:'+str(int(subid))+'   '+'SampleIndex:'+str(int(sampleidx))+'   '+'Label:'+labelstr+'   '+'$P_{alert}=$'+str(round(likelihood[0],2))+'   $P_{drowsy}=$'+str(round(1-likelihood[0],2)),y=1.001)  
        thespan=np.percentile(sampleInput,98)        
        
        xx=np.arange(1,sampleLength+1)             
        for i in range(0,sampleChannel):            
            y=sampleInput[i,:]+thespan*(sampleChannel-1-i)
            dydx=heatmap[i,:]           
            points = np.array([xx, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            #norm = plt.Normalize(occlude_map.min(), occlude_map.max())
            norm = plt.Normalize(0,1)
            lc = LineCollection(segments, cmap='viridis', norm=norm)
            lc.set_array(dydx)
            lc.set_linewidth(2)
            axs0.add_collection(lc)
        
        fig.colorbar(lc,ax=axs0)
        yttics=np.zeros(sampleChannel)
        for gi in range(sampleChannel):
            yttics[gi]=gi*thespan

        axs0.set_ylim([-thespan,thespan*sampleChannel])          
        axs0.set_xlim([0,sampleLength+1]) 
        axs0.set_xticks([0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640])
        axs0.set_xticklabels([0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5])     
        axs0.set_xlabel('Time (s)')   
        
        channelnames=['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7', 'FC3', 'FCz', 'FC4', 'FT8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8','T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'Oz','O2']
 
        inversechannelnames=[]
        for i in range(sampleChannel):
            inversechannelnames.append(channelnames[sampleChannel-1-i])
                   
        plt.sca(axs0)
        plt.yticks(yttics, inversechannelnames)        
        
        deltapower=np.zeros(sampleChannel)
        thetapower=np.zeros(sampleChannel)
        alphapower=np.zeros(sampleChannel)
        betapower=np.zeros(sampleChannel)
        
        #freqs = np.logspace(*np.log10([6, 35]), num=8)
        #n_cycles = freqs / 2.  # different number of cycle per frequency
        #power, itc = tfr_morlet(sampleInput, freqs=freqs, n_cycles=n_cycles, use_fft=True,
        #                        return_itc=True, decim=3, n_jobs=None)
        #power.plot_joint(baseline=(-0.5, 0), mode='mean', tmin=-.5, tmax=2,
        #         timefreqs=[(.5, 10), (1.3, 8)])
        for kk in range(sampleChannel):
            #psd, freqs = psd_array_multitaper(sampleInput[kk,:], 200, adaptive=True,normalization='full', verbose=0)
            #psd, freqs = psd_array_welch(sampleInput[kk,:], 200, n_fft=600)
            wsize = 200
            psd = abs(stft(sampleInput[kk,:], wsize))
            #import pdb; pdb.set_trace()
            freqs = mne.time_frequency.stftfreq(wsize, sfreq=200)
            freq_res = freqs[1] - freqs[0]
            psd = np.sum(psd,axis=2)
            psd = np.squeeze(psd)
            idx_band = np.logical_and(freqs >= 1, freqs <= 30)
            totalpower=simps(psd, dx=freq_res)/30
            if totalpower<0.00000001:
               deltapower[kk]=0
               thetapower[kk]=0
               alphapower[kk]=0
               betapower[kk]=0
            else:
                idx_band = np.logical_and(freqs >= 1, freqs <= 4)
                deltapower[kk] = simps(psd[idx_band], dx=freq_res)/totalpower/4
                idx_band = np.logical_and(freqs >= 4, freqs <= 8)
                thetapower[kk]  = simps(psd[idx_band], dx=freq_res)/totalpower/5
                idx_band = np.logical_and(freqs >= 8, freqs <= 12)
                alphapower[kk]  = simps(psd[idx_band], dx=freq_res)/totalpower/5       
                idx_band = np.logical_and(freqs >= 12, freqs <= 30)
                betapower[kk]  = simps(psd[idx_band], dx=freq_res)/totalpower/19


        #axs21.set_title('Delta',y=-0.2)
        #axs22.set_title('Theta',y=-0.2)
        #axs23.set_title('Alpha',y=-0.2)
        #axs24.set_title('Beta',y=-0.2)

        montage ='standard_1020'
        sfreq = 200
        
        ch_names=channelnames
        
        info = mne.create_info(
            channelnames,
            ch_types=['eeg', 'eeg', 'eeg', 'eeg', 'eeg',\
                      'eeg', 'eeg', 'eeg', 'eeg', 'eeg',\
                      'eeg', 'eeg', 'eeg', 'eeg', 'eeg',\
                      'eeg', 'eeg', 'eeg', 'eeg', 'eeg',\
                      'eeg', 'eeg', 'eeg', 'eeg', 'eeg',\
                      'eeg', 'eeg', 'eeg', 'eeg', 'eeg'],
            sfreq=sfreq      
            #montage=montage
        )
        
        info.set_montage(montage=montage)


        topoHeatmap = np.mean(heatmap, axis=1)
        #im,cn=mne.viz.plot_topomap(data=topoHeatmap,pos=info, vmin=occlude_map.min(), vmax=occlude_map.max(), axes=axs1, names=ch_names,show_names=True,outlines='head',cmap='viridis',show=False)
        #fig.colorbar(im,ax=axs0)

        mixpower=np.zeros((4,sampleChannel))
        mixpower[0,:]=deltapower
        mixpower[1,:]=thetapower
        mixpower[2,:]=alphapower
        mixpower[3,:]=betapower

        #vmax=np.percentile(mixpower,95)  
        basepower = np.mean(psd)
        deltapower = np.log((np.mean(psd[1:4]))/basepower)
        thetapower  = np.log((np.mean(psd[4:9]))/basepower)
        alphapower  = np.log((np.mean(psd[8:15]))/basepower)     
        betapower  = np.log((np.mean(psd[14:30]))/basepower)
        
        vmax=2  
        vmin=-2     
        
        labels = ['Delta', 'Theta', 'Alpha', 'Beta']
        counts = [deltapower, thetapower, alphapower, betapower]
        #axs1.bar(labels, counts, color='tab:orange')
        #axs1.set_ylabel('Relative power (dB)')
        #axs1.set_title('Relative power in different frequency bands')
        tick_label = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7', 'FC3', 'FCZ', 'FC4', 'FT8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'Oz', 'O2']
        #import pdb; pdb.set_trace()
        pos = axs1.imshow(adj.cpu().numpy(),vmin=0.3,vmax=0.8)
        #pos = axs1.imshow(np.mean(adj.cpu().numpy(),axis=0))
        axs1.set_xticks(np.arange(30))
        axs1.set_xticklabels(tick_label, rotation=90)
        axs1.set_yticks(np.arange(30))
        axs1.set_yticklabels(tick_label)
        fig.colorbar(pos, ax=axs1)
        #import pdb; pdb.set_trace()
        #im,cn=mne.viz.plot_topomap(data=deltapower,pos=info, vmin=0, vmax=vmax, axes=axs21, names=ch_names,show_names=True,outlines='head',cmap='plasma',show=False)
        #im,cn=mne.viz.plot_topomap(data=thetapower,pos=info, vmin=0, vmax=vmax, axes=axs22, names=ch_names,show_names=True,outlines='head',cmap='plasma',show=False)
        #im,cn=mne.viz.plot_topomap(data=alphapower,pos=info, vmin=0, vmax=vmax, axes=axs23, names=ch_names,show_names=True,outlines='head',cmap='plasma',show=False)
        #im,cn=mne.viz.plot_topomap(data=betapower,pos=info, vmin=0, vmax=vmax, axes=axs24, names=ch_names,show_names=True,outlines='head',cmap='plasma',show=False)
        #fig.colorbar(im,ax=[axs21,axs22,axs23,axs24])
        plt.savefig('./figures/{}_{}.png'.format(subid,sampleidx))
        #plt.show()
 
def computeSliceMatrix(
        xdata,
        time_step_size=1,
        clip_len=10,
        is_fft=True):
    """
    Comvert entire EEG sequence into clips of length clip_len
    Args:
        h5_fn: file name of resampled signal h5 file (full path)
        edf_fn: full path to edf file
        seizure_idx: current seizure index in edf file, int
        time_step_size: length of each time step, in seconds, int
        clip_len: sliding window size or EEG clip length, in seconds, int
        is_fft: whether to perform FFT on raw EEG data
    Returns:
        eeg_clip: eeg clip (clip_len, num_channels, time_step_size*freq)
    """
    # get corresponding eeg clip
    signal_array = xdata[:, :]
    #for i in range(signal_array.shape[0]):
    #import pdb; pdb.set_trace()
    #signal_array = butter_bandpass_filter(signal_array, 4, 12, fs, order=3)
    FREQUENCY = 128
    time_step_size = 0.5
    physical_time_step_size = int(FREQUENCY * time_step_size)
    start_time_step = 0
    time_steps = []
    while start_time_step <= signal_array.shape[1] - physical_time_step_size:
        end_time_step = start_time_step + physical_time_step_size
        #print(end_time_step)
        # (num_channels, physical_time_step_size)
        curr_time_step = signal_array[:, start_time_step:end_time_step]
        if is_fft:
            curr_time_step, _ = computeFFT(
                curr_time_step, n=physical_time_step_size)
            #curr_time_step, freqs = psd_array_multitaper(curr_time_step, FREQUENCY, adaptive=True,normalization='full', verbose=0)
            #print(freqs[1]-freqs[0])
            #print(curr_time_step.shape)
            #curr_time_step[curr_time_step>10]=0
            #curr_time_step[curr_time_step<0]=0
            #plt.plot(curr_time_step.T)
            #plt.show()
            curr_time_step = curr_time_step[:,2:5]
            #for ii in range(curr_time_step.shape[0]):
            #    if sum(curr_time_step[ii,:])==0:
            #        curr_time_step[ii,:] = (curr_time_step[ii-1,:]+curr_time_step[ii+1,:])/2
            #print(curr_time_step[:,1]-curr_time_step[:,-1])
            #curr_time_step = gen_features_psd(curr_time_step)

        time_steps.append(curr_time_step)
        start_time_step = end_time_step

    eeg_clip = np.stack(time_steps, axis=0)
    #print(eeg_clip.shape)
    #eeg_clip = (eeg_clip - np.mean(eeg_clip))/np.std(eeg_clip)
    #print(eeg_clip)
    #eeg_clip = np.expand_dims(signal_array,axis=0)
    return eeg_clip     

def run(args):

    device = "cuda"
    lr = 1e-3
    #filename = r'/data/zhuzhuan/drowsy_200hz_11subs_clean3.mat'
    #filename = r'/data/zhuzhuan/drowsy_200hz_11subs2.mat'
    filename = r'/data/zhuzhuan/drowsy_11subs_balanced_1880_5s_128hz.mat'
    tmp = sio.loadmat(filename)
    # xdata = np.array(tmp['EEGsample'])
    # label = np.array(tmp['substate'])
    # subIdx = np.array(tmp['subindex'])
    xdata = np.array(tmp['EEG_sample']).astype(np.float64)
    label = np.array(tmp['labels']) - 1
    subIdx = np.array(tmp['sub_index'])
 
    channelnum=30
    classes=2
    subjnum=11
    samplelength=3

    label.astype(int)    
    subIdx.astype(int)

    samplenum=label.shape[0]
    sf=128
    ydata=np.zeros(samplenum,dtype=np.longlong)
    
    for i in range(samplenum):
        ydata[i]=label[i]

    batch_size = 50   
    n_epoch =11    

    for subid in range(1,12):
        dataloaders, _ = load_dataset_classification(
                        input_dir=args.input_dir,
                        raw_data_dir=args.raw_data_dir,
                        train_batch_size=args.train_batch_size,
                        test_batch_size=args.test_batch_size,
                        time_step_size=args.time_step_size,
                        max_seq_len=args.max_seq_len,
                        standardize=True,
                        num_workers=args.num_workers,
                        padding_val=0.,
                        augmentation=args.data_augment,
                        adj_mat_dir='./data/electrode_graph/adj_mx_3d.pkl',
                        graph_type=args.graph_type,
                        top_k=args.top_k,
                        filter_type=args.filter_type,
                        use_fft=args.use_fft,
                        preproc_dir=args.preproc_dir,
                        sub_num = subid,
                        input_dim = args.input_dim)
                    
        testindx=np.where(subIdx == subid)[0]    
        xtest=xdata[testindx]
        y_test=ydata[testindx]
        
        #import pdb; pdb.set_trace()
        if args.task == 'detection':
            loss_fn = nn.BCEWithLogitsLoss().to(device)
        else:
            loss_fn = nn.CrossEntropyLoss().to(device)
        loss_fn = nn.BCEWithLogitsLoss().to(device)
        # Data loaders
        train_loader = dataloaders['train']
        dev_loader = dataloaders['dev']
        test_loader = dataloaders['test']

        model = DCRNNModel_classification(
                args=args, num_classes=args.num_classes, device=device)
        
        model = model.to(device)
        # To train mode
        model.train()
    
        # Get optimizer and scheduler
        optimizer = optim.Adam(params=model.parameters(),
                               lr=args.lr_init, weight_decay=args.l2_wd)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
        # average meter for validation loss
        nll_meter = utils.AverageMeter()
    
        # Train
        #log.info('Training...')
        epoch = 0
        step = 0
        prev_val_loss = 1e10
        patience_count = 0
        early_stop = False
        
        while (epoch != args.num_epochs) and (not early_stop):
            epoch += 1
            #log.info('Starting epoch {}...'.format(epoch))
            total_samples = len(train_loader.dataset)
            with torch.enable_grad(), \
                    tqdm(total=total_samples) as progress_bar:
                for xdata, x, y, seq_lengths, supports, adj_mat in train_loader:
                    batch_size = x.shape[0]
    
                    # input seqs
                    x = x.to(device)
                    y = 1-y.view(-1).to(device)  # (batch_size,)
                    seq_lengths = seq_lengths.view(-1).to(device)  # (batch_size,)
                    adj_mat = adj_mat.to(device)
                    for i in range(len(supports)):
                        supports[i] = supports[i].to(device)
    
                    # Zero out optimizer first
                    optimizer.zero_grad()
    
                    # Forward
                    # (batch_size, num_classes)
                    #print(adj_mat)
                    logits,adj_ori_batch = model(xdata, x, seq_lengths, supports,adj_mat)
                    if logits.shape[-1] == 1:
                        logits = logits.view(-1)  # (batch_size,)
                    #loss_class = torch.nn.NLLLoss().cuda()

                    loss = loss_fn(logits, y.float())
                    
                    #loss = loss_class(logits, y)
                    loss_val = loss.item()
    
                    # Backward
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    step += batch_size
    
                    # Log info
                    progress_bar.update(batch_size)
                    progress_bar.set_postfix(epoch=epoch,
                                             loss=loss_val,
                                             lr=optimizer.param_groups[0]['lr'])
        
        
        #pretrained_model = DCRNNModel_classification(args=args, num_classes=args.num_classes, device=device)
        #model = utils.load_model_checkpoint('/data/zhuzhuan/eeg-gnn-ssl-drowsiness/save/train/train-68_0.7159/last.pth.tar', model)
        model.cuda()
        model.eval()
        with torch.no_grad(), tqdm(total=len(test_loader.dataset)) as progress_bar:
            
            #x_test = xtest.reshape(xtest.shape[0], 1,channelnum, samplelength*sf)
            #x_test =  torch.DoubleTensor(x_test).cuda()
            x_test = xtest
            for xdata, x, y, seq_lengths, supports, adj_mat in test_loader:
                batch_size = x.shape[0]
    
                # Input seqs
                x = x.to(device)
                y = 1-y.view(-1).to(device)  # (batch_size,)
                adj_mat = adj_mat.to(device)
                seq_lengths = seq_lengths.view(-1).to(device)  # (batch_size,)
    
                for i in range(len(supports)):
                    supports[i] = supports[i].to(device)
                # Forward
                # (batch_size, num_classes)
                
                logits,adj_test = model(xdata, x, seq_lengths, supports,adj_mat)
                #logits = F.softmax(logits, dim=1)
                #logits = (logits > 0.5).cpu().numpy().astype(int)
                occlude_map = torch.zeros(x.shape[0],30,10)
                np.sum(logits.T == y.cpu().numpy())/len(logits)

                print(logits)
                #import pdb; pdb.set_trace()
                #with Pool(processes=4) as pool:
                '''
                for channel in range(30):
                    #x_new = np.zeros((x.shape[0],6,30,15), dtype=np.float32)
                    #for ii in range(x.shape[0]):
                    #import pdb; pdb.set_trace()
                    x_new = x.clone().detach().cpu().numpy()
                    #for ii in range(x_new.shape[0]):
                    #import pdb; pdb.set_trace()
                    #mask_input = np.delete(x_new,channel,axis=2)
                    #mean_mask_input = np.mean(mask_input,axis=2)
                    #x_new[:,seq,channel,:]=mean_mask_input[:,seq]
                    #x_new[:,0:3,channel,:]=0
                    x_new =  torch.FloatTensor(x_new).cuda()                       
                    #x_occlude = x.clone()
                    #x_occlude[:,seq,channel,:]=0
                    logits_occlude,_ = model(x_new, seq_lengths, supports,adj_mat,block_chan=None)
                    logits_occlude = F.softmax(logits_occlude, dim=1)
                    #import pdb; pdb.set_trace()
                    #y_map = y.clone()
                    #y_map[y_map==0]=-1
                    #occlude_map[:,channel,seq] = -y_map*(logits_occlude[:,0]-logits[:,0])
                    #occlude_map[occlude_map<0]=0
                    #occlude_map[:,channel,seq] = torch.abs(logits_occlude[:,0]-logits[:,0])
                    probs = logits_occlude.cpu().numpy()
                    y_pred = np.argmax(probs, axis=1).reshape(-1)
                    acc = accuracy_score(y_true=y.cpu().numpy(), y_pred=y_pred)
                    print(acc)
                    
                import pdb; pdb.set_trace()
                
                '''
                for channel in range(30):
                    for seq in range(10):
                        #x_new = np.zeros((x.shape[0],6,30,15), dtype=np.float32)
                        #for ii in range(x.shape[0]):
                        #import pdb; pdb.set_trace()
                        x_new = x.clone().detach().cpu().numpy()
                        #for ii in range(x_new.shape[0]):
                        #import pdb; pdb.set_trace()
                        #mask_input = np.delete(x_new,channel,axis=2)
                        #mean_mask_input = np.mean(mask_input,axis=2)
                        #x_new[:,seq,channel,:]=mean_mask_input[:,seq]
                        #x_new[:,seq,channel,:]=20
                        print(channel)
                        
                        for ii in range(0,xtest.shape[0]):
                            
                            xdata_all = xtest.copy()
                            xdata_occlude = xdata_all[ii]
                            xdata_occlude[channel,seq*64:(seq+1)*64] = 0
                            #import pdb; pdb.set_trace()
                            #mask_input = np.delete(xdata_occlude,channel,axis=0)
                            #mean_mask_input = np.mean(mask_input,axis=0)
                            #xdata_occlude[channel,seq*100:(seq+1)*100]=mean_mask_input[seq*100:(seq+1)*100]
                            x_new[ii] = computeSliceMatrix(xdata_occlude,time_step_size=1,clip_len=10,is_fft=True)
                            #import pdb; pdb.set_trace()
                        
                        x_new =  torch.FloatTensor(x_new).cuda()                       
                        #x_occlude = x.clone()
                        #x_occlude[:,seq,channel,:]=0
                        logits_occlude,_ = model(xdata, x_new, seq_lengths, supports,adj_mat)
                        #logits_occlude = F.softmax(logits_occlude, dim=1)
                        #import pdb; pdb.set_trace()
                        y_map = y.clone()
                        y_map[y_map==0]=-1
                        # 0 for fatigue and 1 for alert
                        # when fatigue occlude-no_occlude is +, need multiply 1
                        # when alert, occlude-no_occlude is -, need multiply -1
                        #occlude_map[:,channel,seq] = y_map*(logits_occlude[:,0]-logits[:,0])
                        #occlude_map[occlude_map<0]=0
                        #import pdb; pdb.set_trace()
                        occlude_map[:,channel,seq] = torch.abs(logits_occlude[:,0]-logits[:,0])
                        #import pdb; pdb.set_trace()
                        #occlude_map[:,channel,seq] = torch.sum(torch.square(logits_occlude-logits),1)
                  
                #import pdb; pdb.set_trace()
                #probs = F.softmax(logits, dim=1).cpu().numpy()
                probs = torch.sigmoid(logits).cpu().numpy() 
                #probs = logits.cpu().numpy()
                y_pred = (probs > 0.5).astype(int)
                print(y_pred)
                print(probs)
                #plt.plot(x[90])
                #plt.show()
                #import pdb; pdb.set_trace()
                #probs=np.exp(logits.cpu().numpy())
                sampleVis =VisTech(model)
    
                # you can change the sample you want to visualize here
                #sampleidx=61
                #sampleidx=179
                #import pdb; pdb.set_trace()
        for sampleidx in range(0,xtest.shape[0]):
            #sampleidx=61
            sampleVis.generate_heatmap(batchInput=x_test,occlude_map=occlude_map,sampleidx=sampleidx,subid=subid,samplelabel=y[sampleidx],likelihood=probs[sampleidx],adj=adj_test[sampleidx])

if __name__ == '__main__':
    run(get_args())
    
