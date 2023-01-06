import numpy as np
import os
import pickle
import torch
import json
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import utils
from data_utils import *
from data_loader_drowsiness import load_dataset_classification
from args import get_args
from collections import OrderedDict
from json import dumps
from model import DCRNNModel_classification
from tensorboardX import SummaryWriter
from tqdm import tqdm
from dotted_dict import DottedDict
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy
from scipy.io import savemat
import pandas as pd

def main(args):

    # Get device
    args.cuda = torch.cuda.is_available()
    device = "cuda" if args.cuda else "cpu"

    # Set random seed
    utils.seed_torch(seed=args.rand_seed)

    # Get save directories
    args.save_dir = utils.get_save_dir(
        args.save_dir, training=True if args.do_train else False)
    # Save args
    args_file = os.path.join(args.save_dir, 'args.json')
    with open(args_file, 'w') as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)

    # Set up logger
    log = utils.get_logger(args.save_dir, 'train')
    tbx = SummaryWriter(args.save_dir)
    log.info('Args: {}'.format(dumps(vars(args), indent=4, sort_keys=True)))
    seed = 0
    utils.seed_torch(seed=seed)
    log.info('Input_dim: {}'.format(args.input_dim))
    for seed in range(5):
        utils.seed_torch(seed=seed)
        for num_features in range(3,4):
            for sub_num in range(1, 12):
                # Build dataset
                log.info('Building dataset...')
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
                    graph_type=args.graph_type,
                    top_k=args.top_k,
                    filter_type=args.filter_type,
                    use_fft=args.use_fft,
                    preproc_dir=args.preproc_dir,
                    sub_num = sub_num,
                    input_dim = num_features)
        
                # Build model
                log.info('Building model...')
                model = DCRNNModel_classification(
                    args=args, num_classes=args.num_classes, device=device)
                if args.do_train:
                    if args.load_model_path is not None:
                        model = utils.load_model_checkpoint(
                            args.load_model_path, model)
        
                    num_params = utils.count_parameters(model)
                    log.info('Total number of trainable parameters: {}'.format(num_params))
        
                    model = model.to(device)
        
                    # Train
                    train(model, dataloaders, args, device, args.save_dir, log, tbx,sub_num)
        
                    # Load best model after training finished
                    best_path = os.path.join(args.save_dir, 'last.pth.tar')
                    model = utils.load_model_checkpoint(best_path, model)
                    model = model.to(device)

def train(model, dataloaders, args, device, save_dir, log, tbx,sub_num):
    """
    Perform training and evaluate on val set
    """

    # Define loss function

    loss_fn = nn.BCEWithLogitsLoss().to(device)
    # Data loaders
    train_loader = dataloaders['train']
    dev_loader = dataloaders['dev']
    test_loader = dataloaders['test']
    # Get saver
    saver = utils.CheckpointSaver(save_dir,
                                  metric_name=args.metric_name,
                                  maximize_metric=args.maximize_metric,
                                  log=log)

    # To train mode
    model.train()

    # Get optimizer and scheduler
    optimizer = optim.Adam(params=model.parameters(),
                           lr=args.lr_init, weight_decay=args.l2_wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # average meter for validation loss
    nll_meter = utils.AverageMeter()

    # Train
    log.info('Training...')
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
                xdata = xdata.to(device)
                y = 1- y.view(-1).to(device)  # (batch_size,)
                seq_lengths = seq_lengths.view(-1).to(device)  # (batch_size,)
                for i in range(len(supports)):
                    supports[i] = supports[i].to(device)

                # Zero out optimizer first
                optimizer.zero_grad()

                # Forward
                logits,adj_ori_batch = model(xdata,x, seq_lengths, supports,adj_mat)
                if logits.shape[-1] == 1:
                    logits = logits.view(-1)  # (batch_size,)
                loss = loss_fn(logits, y.float())

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

                tbx.add_scalar('train/Loss', loss_val, step)
                tbx.add_scalar('train/LR',
                               optimizer.param_groups[0]['lr'],
                               step)

            if epoch % args.eval_every == 0:
                # Evaluate and save checkpoint
                log.info('Evaluating at epoch {}...'.format(epoch))

                eval_results,_ = evaluate(model,
                                        test_loader,
                                        args,
                                        save_dir,
                                        device,
                                        is_test=False,
                                        nll_meter=nll_meter,
                                        adj_score=None,
                                        epoch=epoch,
                                        sub_num=sub_num)

                best_path = saver.save(epoch,
                                       model,
                                       optimizer,
                                       eval_results[args.metric_name])

                # Accumulate patience for early stopping
                if eval_results['loss'] < prev_val_loss:
                    patience_count = 0
                else:
                    patience_count += 1
                prev_val_loss = eval_results['loss']

                # Back to train mode
                model.train()

                # Log to console
                results_str = ', '.join('{}: {:.3f}'.format(k, v)
                                        for k, v in eval_results.items())
                log.info('Test {}'.format(results_str))

                # Log to TensorBoard
                log.info('Visualizing in TensorBoard...')
                for k, v in eval_results.items():
                    tbx.add_scalar('eval/{}'.format(k), v, step)

        # Step lr scheduler
        scheduler.step()


def evaluate(
        model,
        dataloader,
        args,
        save_dir,
        device,
        is_test=False,
        nll_meter=None,
        eval_set='dev',
        best_thresh=0.5,
        adj_score=None,
        epoch=0,
        sub_num=0):
    # To evaluate mode
    model.eval()

    # Define loss function
    loss_fn = nn.BCEWithLogitsLoss().to(device)
    y_pred_all = []
    y_true_all = []
    y_prob_all = []
    file_name_all = []
    with torch.no_grad(), tqdm(total=len(dataloader.dataset)) as progress_bar:
        for xdata,x, y, seq_lengths, supports, adj_mat in dataloader:
            batch_size = x.shape[0]

            # Input seqs
            xdata = xdata.to(device)
            x = x.to(device)
            y = 1 - y.view(-1).to(device)  # (batch_size,)
            seq_lengths = seq_lengths.view(-1).to(device)  # (batch_size,)
            for i in range(len(supports)):
                supports[i] = supports[i].to(device)

            # Forward
            # (batch_size, num_classes)
            logits,adj = model(xdata,x, seq_lengths, supports,adj_mat)
            adj_score = []

            if logits.shape[-1] == 1:  # binary detection
                logits = logits.view(-1)  # (batch_size,)
                y_prob = torch.sigmoid(logits).cpu().numpy()  # (batch_size, )
                y_true = y.cpu().numpy().astype(int)
                y_pred = (y_prob > best_thresh).astype(int)  # (batch_size, )
            else:
                # (batch_size, num_classes)
                y_prob = F.softmax(logits, dim=1).cpu().numpy()
                y_pred = np.argmax(y_prob, axis=1).reshape(-1)  # (batch_size,)
                y_true = y.cpu().numpy().astype(int)

            # Update loss
            loss = loss_fn(logits, y.float())

            if nll_meter is not None:
                nll_meter.update(loss.item(), batch_size)

            y_pred_all.append(y_pred)
            y_true_all.append(y_true)
            y_prob_all.append(y_prob)
            file_name_all.extend('aa')

            # Log info
            progress_bar.update(batch_size)

    y_pred_all = np.concatenate(y_pred_all, axis=0)
    y_true_all = np.concatenate(y_true_all, axis=0)
    y_prob_all = np.concatenate(y_prob_all, axis=0)

    scores_dict, _, _ = utils.eval_dict(y_pred=y_pred_all,
                                        y=y_true_all,
                                        y_prob=y_prob_all,
                                        file_names=file_name_all,
                                        average="binary" if args.task == "detection" else "weighted")

    eval_loss = nll_meter.avg if (nll_meter is not None) else loss.item()
    results_list = [('loss', eval_loss),
                    ('sub',sub_num),
                    ('epoch',int(epoch)),
                    ('acc', scores_dict['acc']),
                    ('F1', scores_dict['F1']),
                    ('recall', scores_dict['recall']),
                    ('precision', scores_dict['precision'])]

    if 'auroc' in scores_dict.keys():
        results_list.append(('auroc', scores_dict['auroc']))
    results = OrderedDict(results_list)

    return results,adj_score


if __name__ == '__main__':
    main(get_args())
