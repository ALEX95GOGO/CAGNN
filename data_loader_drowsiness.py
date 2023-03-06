import scipy.io as sio
import sys
from pathlib import Path
import scipy.signal
import scipy
import pickle
import os
import numpy as np
import h5py
import math
import torch
from torch.utils.data import Dataset, DataLoader
from utils import StandardScaler
from data_utils import *
import utils
from sklearn import preprocessing
from scipy.signal import welch
from scipy.integrate import simps
from pywt import wavedec
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from mne.time_frequency import psd_array_multitaper
from scipy.io import loadmat


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def compute_DE(signal):
    variance = np.var(signal,ddof=1)
    return math.log(2*math.pi*math.e*variance)/2

def computeSliceMatrix(
        xdata,
        time_step_size=1,
        clip_len=60,
        is_fft=False,
        input_dim=16):
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
        #print(signal_array.shape)
        if is_fft:
            #_,curr_time_step = DE_PSD(curr_time_step)
            curr_time_step, _ = computeFFT(
                curr_time_step, n=physical_time_step_size)
            curr_time_step = curr_time_step[:,2:5]

        time_steps.append(curr_time_step)
        start_time_step = end_time_step

    eeg_clip = np.stack(time_steps, axis=0)
    return eeg_clip


class DrowsyDataset(Dataset):
    def __init__(
            self,
            time_step_size=1,
            max_seq_len=60,
            standardize=True,
            scaler=None,
            split='train',
            padding_val=0,
            data_augment=False,
            adj_mat_dir=None,
            graph_type=None,
            top_k=None,
            filter_type='laplacian',
            use_fft=False,
            preproc_dir=None,
            sub_num=None,
            input_dim=16):
        """
        Args:
            input_dir: dir to resampled signals h5 files
            raw_data_dir: dir to TUSZ edf files
            time_step_size: int, in seconds
            max_seq_len: int, EEG clip length, in seconds
            standardize: if True, will z-normalize wrt train set
            scaler: scaler object for standardization
            split: train, dev or test
            padding_val: int, value used for padding to max_seq_len
            data_augment: if True, perform random augmentation of EEG
            adj_mat_dir: dir to pre-computed distance graph adjacency matrix
            graph_type: 'combined' (i.e. distance graph) or 'individual' (correlation graph)
            top_k: int, top-k neighbors of each node to keep. For correlation graph only
            filter_type: 'laplacian' for distance graph, 'dual_random_walk' for correlation graph
            use_fft: whether perform Fourier transform
            preproc_dir: dir to preprocessed Fourier transformed data, optional
        """
        if (graph_type == 'individual') and (top_k is None):
            raise ValueError('Please specify top_k for individual graph.')

        self.time_step_size = time_step_size
        self.max_seq_len = max_seq_len
        self.standardize = standardize
        self.scaler = scaler
        self.split = split
        self.padding_val = padding_val
        self.data_augment = data_augment
        self.adj_mat_dir = adj_mat_dir
        self.graph_type = graph_type
        self.top_k = top_k
        self.filter_type = filter_type
        self.use_fft = use_fft
        self.preproc_dir = preproc_dir
        self.input_dim = input_dim
        leave_out_sub = sub_num

        filename = r"drowsy_11subs_balanced_1880_5s_128hz.mat"
        tmp = sio.loadmat(filename)
        xdata = np.array(tmp['EEG_sample'])
        label = np.array(tmp['labels']) - 1
        subIdx = np.array(tmp['sub_index'])

        trainindx = np.where(subIdx != leave_out_sub)[0]
        np.random.shuffle(trainindx)

        xtrain = xdata[trainindx]

        #       form the testing data
        #print(subIdx.shape)
        testindx = np.where(subIdx == leave_out_sub)[0]

        xtest = xdata[testindx]
        #print(xtrain)
        if self.split=='train':
            self.size=len(xtrain)
            self.xdata = xtrain
            self.label = label[trainindx]
            self.subIdx = subIdx[trainindx]
        if self.split=='dev':
            self.size=len(xtrain)
            self.xdata = xtrain
            self.label = label[trainindx]
            self.subIdx = subIdx[trainindx]
        if self.split == 'test':
            self.size = len(xtest)
            self.xdata = xtest
            self.label = label[testindx]
            self.subIdx = subIdx[testindx]
        
        if self.preproc_dir is not None:
            self.file_list = []
            if self.split=="train":
                for path, currentDirectory, files in os.walk(self.preproc_dir):
                    for file in files:
                        if not file.startswith("S{}_".format(leave_out_sub)):
                            self.file_list.append(file)
            else:
                for path, currentDirectory, files in os.walk(self.preproc_dir):
                    for file in files:
                        if file.startswith("S{}_".format(leave_out_sub)):
                            self.file_list.append(file)

    def __len__(self):
        return self.size

    def get_labels(self):
        return self.label.flatten()
        
    def _random_reflect(self, EEG_seq):
        """
        Randomly reflect EEG channels along the midline
        """
        swap_pairs = get_swap_pairs(INCLUDED_CHANNELS)
        EEG_seq_reflect = EEG_seq.copy()
        if (np.random.choice([True, False])):
            for pair in swap_pairs:
                EEG_seq_reflect[:, [pair[0], pair[1]],
                :] = EEG_seq[:, [pair[1], pair[0]], :]
        else:
            swap_pairs = None
        return EEG_seq_reflect, swap_pairs

    def _random_scale(self, EEG_seq):
        """
        Scale EEG signals by a random value between 0.8 and 1.2
        """
        scale_factor = np.random.uniform(0.8, 1.2)
        if self.use_fft:
            EEG_seq += np.log(scale_factor)
        else:
            EEG_seq *= scale_factor
        return EEG_seq

    def _get_indiv_graphs(self, eeg_clip, swap_nodes=None):
        """
        Compute adjacency matrix for correlation graph
        Args:
            eeg_clip: shape (seq_len, num_nodes, input_dim)
            swap_nodes: list of swapped node index
        Returns:
            adj_mat: adjacency matrix, shape (num_nodes, num_nodes)
        """
        self.sensor_ids = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7', 'FC3', 'FCZ', 'FC4', 'FT8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'Oz', 'O2']
        num_sensors = len(self.sensor_ids)

        adj_mat = np.eye(num_sensors, num_sensors,
                         dtype=np.float32)  # diagonal is 1

        # (num_nodes, seq_len, input_dim)
        eeg_clip = np.transpose(eeg_clip, (1, 0, 2))

        assert eeg_clip.shape[0] == num_sensors

        # (num_nodes, seq_len*input_dim)
        eeg_clip = eeg_clip.reshape((num_sensors, -1))

        sensor_id_to_ind = {}
        for i, sensor_id in enumerate(self.sensor_ids):
            sensor_id_to_ind[sensor_id] = i

        if swap_nodes is not None:
            for node_pair in swap_nodes:
                node_name0 = [
                    key for key,
                            val in sensor_id_to_ind.items() if val == node_pair[0]][0]
                node_name1 = [
                    key for key,
                            val in sensor_id_to_ind.items() if val == node_pair[1]][0]
                sensor_id_to_ind[node_name0] = node_pair[1]
                sensor_id_to_ind[node_name1] = node_pair[0]

        for i in range(0, num_sensors):
            for j in range(i + 1, num_sensors):
                xcorr = comp_xcorr(
                    eeg_clip[i, :], eeg_clip[j, :], mode='valid', normalize=True)
                adj_mat[i, j] = xcorr
                adj_mat[j, i] = xcorr

        adj_mat = abs(adj_mat)

        if (self.top_k is not None):
            adj_mat = keep_topk(adj_mat, top_k=self.top_k, directed=True)
        else:
            raise ValueError('Invalid top_k value!')

        return adj_mat

    def _get_combined_graph(self, swap_nodes=None):
        """
        Get adjacency matrix for pre-computed distance graph
        Returns:
            adj_mat_new: adjacency matrix, shape (num_nodes, num_nodes)
        """
        adj_mat = sio.loadmat('all_adj_score.mat')['adj_score']
        adj_mat = np.mean(adj_mat,axis=0)
        #print(adj_mat.shape)
        adj_mat = adj_mat.astype(np.float32)
        adj_mat_new = adj_mat.copy()
        if swap_nodes is not None:
            for node_pair in swap_nodes:
                for i in range(adj_mat.shape[0]):
                    adj_mat_new[node_pair[0], i] = adj_mat[node_pair[1], i]
                    adj_mat_new[node_pair[1], i] = adj_mat[node_pair[0], i]
                    adj_mat_new[i, node_pair[0]] = adj_mat[i, node_pair[1]]
                    adj_mat_new[i, node_pair[1]] = adj_mat[i, node_pair[0]]
                    adj_mat_new[i, i] = 1
                adj_mat_new[node_pair[0], node_pair[1]
                ] = adj_mat[node_pair[1], node_pair[0]]
                adj_mat_new[node_pair[1], node_pair[0]
                ] = adj_mat[node_pair[0], node_pair[1]]

        return adj_mat_new

    def _compute_supports(self, adj_mat):
        """
        Comput supports
        """
        supports = []
        supports_mat = []
        if self.filter_type == "laplacian":  # ChebNet graph conv
            supports_mat.append(
                utils.calculate_scaled_laplacian(adj_mat, lambda_max=None))
        elif self.filter_type == "random_walk":  # Forward random walk
            supports_mat.append(utils.calculate_random_walk_matrix(adj_mat).T)
        elif self.filter_type == "dual_random_walk":  # Bidirectional random walk
            supports_mat.append(utils.calculate_random_walk_matrix(adj_mat).T)
            supports_mat.append(
                utils.calculate_random_walk_matrix(adj_mat.T).T)
        else:
            supports_mat.append(utils.calculate_scaled_laplacian(adj_mat))
        for support in supports_mat:
            supports.append(torch.FloatTensor(support.toarray()))
        return supports

    def __getitem__(self, idx):
        """
        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]
        Returns:
            a tuple of (x, y, seq_len, supports, adj_mat, write_file_name)
        """


        ydata = self.label[idx]

        xdata = self.xdata[idx, :, :]
        
        
        # preprocess
        if self.preproc_dir is None:
            eeg_clip = computeSliceMatrix(
                xdata=xdata,
                time_step_size=self.time_step_size, clip_len=self.max_seq_len,
                is_fft=self.use_fft,
                input_dim = self.input_dim)
        else:
            #with h5py.File(os.path.join(self.preproc_dir, 'S'+str(self.subIdx[idx][0]) +'_' + str(idx) + '.h5'), 'r') as hf:
            with h5py.File(os.path.join(self.preproc_dir, self.file_list[idx]), 'r') as hf:
                eeg_clip = hf['clip'][()]
                eeg_clip = eeg_clip[:,:,2:8]

        # data augmentation
        if self.data_augment:
            #curr_feature, swap_nodes = self._random_reflect(eeg_clip)
            swap_nodes = None
            curr_feature = eeg_clip.copy()
            #print(curr_feature.shape)
            decision = np.random.randint(2, size=1)
            #print(decision)
            if decision == 1:
                channel = np.random.randint(30, size=1)
                curr_feature[:,channel,:]=0
            #curr_feature = self._random_scale(curr_feature)
        else:
            swap_nodes = None
            curr_feature = eeg_clip.copy()

        # standardize wrt train mean and std
        #curr_feature = (curr_feature - np.mean(curr_feature))/np.std(curr_feature)
        #if self.standardize:
        #    curr_feature = self.scaler.transform(curr_feature)

        # padding
        curr_len = curr_feature.shape[0]
        seq_len = np.minimum(curr_len, self.max_seq_len)
        if curr_len < self.max_seq_len:
            len_pad = self.max_seq_len - curr_len
            padded_feature = np.ones(
                (len_pad, curr_feature.shape[1], curr_feature.shape[2])) * self.padding_val
            padded_feature = np.concatenate(
                (curr_feature, padded_feature), axis=0)
        else:
            padded_feature = curr_feature.copy()

        if np.any(np.isnan(padded_feature)):
            raise ValueError("Nan found in x!")

        # convert to tensors
        # (max_seq_len, num_nodes, input_dim)
        x = torch.FloatTensor(padded_feature)
        y = torch.LongTensor(ydata)
        seq_len = torch.LongTensor([seq_len])

        # Get adjacency matrix for graph
        if self.graph_type == 'combined':
            #eeg_clip = eeg_clip.squeeze()
            indiv_adj_mat = self._get_indiv_graphs(eeg_clip, swap_nodes)
            indiv_supports = self._compute_supports(indiv_adj_mat)
            curr_support = np.concatenate(indiv_supports, axis=0)
            if np.any(np.isnan(curr_support)):
                raise ValueError("Nan found in indiv_supports!")
        elif self.adj_mat_dir is not None:
            indiv_adj_mat = self._get_combined_graph(swap_nodes)
            indiv_supports = self._compute_supports(indiv_adj_mat)
        else:
            indiv_supports = []
            indiv_adj_mat = []
        #print(seq_len)

        return (xdata,x, y, seq_len, indiv_supports, indiv_adj_mat)

def load_dataset_classification(
        train_batch_size,
        test_batch_size=None,
        time_step_size=1,
        max_seq_len=60,
        standardize=True,
        num_workers=8,
        padding_val=0.,
        augmentation=False,
        adj_mat_dir=None,
        graph_type='combined',
        top_k=None,
        filter_type='laplacian',
        use_fft=False,
        preproc_dir=None,
        sub_num=None,
        input_dim=16):
    """
    Args:
        input_dir: dir to resampled signals h5 files
        raw_data_dir: dir to TUSZ raw edf files
        train_batch_size: int
        test_batch_size: int
        time_step_size: int, in seconds
        max_seq_len: EEG clip length, in seconds
        standardize: if True, will z-normalize wrt train set
        num_workers: int
        padding_val: value used for padding
        augmentation: if True, perform random augmentation of EEG
        adj_mat_dir: dir to pre-computed distance graph adjacency matrix
        graph_type: 'combined' (i.e. distance graph) or 'individual' (correlation graph)
        top_k: int, top-k neighbors of each node to keep. For correlation graph only
        filter_type: 'laplacian' for distance graph, 'dual_random_walk' for correlation graph
        use_fft: whether perform Fourier transform
        preproc_dir: dir to preprocessed Fourier transformed data, optional
    Returns:
        dataloaders: dictionary of train/dev/test dataloaders
        datasets: dictionary of train/dev/test datasets
        scaler: standard scaler
    """
    if (graph_type is not None) and (
            graph_type not in ['individual', 'combined']):
        raise NotImplementedError

    scaler = None

    dataloaders = {}
    datasets = {}
    for split in ['train', 'dev', 'test']:
        if split == 'train':
            data_augment = augmentation
        else:
            data_augment = False  # no augmentation on dev/test sets

        dataset = DrowsyDataset(time_step_size=time_step_size,
                                 max_seq_len=max_seq_len,
                                 standardize=standardize,
                                 scaler=scaler,
                                 split=split,
                                 padding_val=padding_val,
                                 data_augment=data_augment,
                                 adj_mat_dir=adj_mat_dir,
                                 graph_type=graph_type,
                                 top_k=top_k,
                                 filter_type=filter_type,
                                 use_fft=use_fft,
                                 preproc_dir=preproc_dir,
                                 sub_num = sub_num,
                                 input_dim = input_dim)

        if split == 'train':
            shuffle = True
            batch_size = train_batch_size
            loader = DataLoader(dataset=dataset,
                                shuffle=shuffle,
                                batch_size=batch_size,
                                num_workers=num_workers)
            dataloaders[split] = loader
        else:
            shuffle = False
            batch_size = test_batch_size

            loader = DataLoader(dataset=dataset,
                                shuffle=shuffle,
                                batch_size=batch_size,
                                num_workers=num_workers)
            dataloaders[split] = loader
        datasets[split] = dataset


    return dataloaders, datasets