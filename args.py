import argparse


def get_args():
    parser = argparse.ArgumentParser('Train GNN on driver drowsiness data.')

    # General args
    parser.add_argument('--save_dir',
                        type=str,
                        default=None,
                        help='Directory to save the outputs and checkpoints.')
    parser.add_argument(
        '--load_model_path',
        type=str,
        default=None,
        help='Model checkpoint to start training/testing from.')
    parser.add_argument('--do_train',
                        default=False,
                        action='store_true',
                        help='Whether perform training.')
    parser.add_argument('--rand_seed',
                        type=int,
                        default=123,
                        help='Random seed.')
    parser.add_argument('--fine_tune',
                        default=False,
                        action='store_true',
                        help='Whether to fine-tune pre-trained model.')

    # Input args
    parser.add_argument(
        '--graph_type',
        choices=(
            'individual',
            'combined'),
        default='individual',
        help='Whether use individual graphs (cross-correlation) or combined graph (distance).')
    parser.add_argument('--max_seq_len',
                        type=int,
                        default='60',
                        help='Maximum sequence length in seconds.')
    parser.add_argument('--node_to_keep',
                        type=int,
                        default='10',
                        help='Node to keep in GNN.')
                        
    parser.add_argument(
        '--output_seq_len',
        type=int,
        default=12,
        help='Output seq length in seconds.')
    parser.add_argument('--time_step_size',
                        type=int,
                        default=1,
                        help='Time step size in seconds.')
    parser.add_argument(
        '--top_k',
        type=int,
        default=3,
        help='Top-k neighbors of each node to keep, for graph sparsity.')

    # Model args
    parser.add_argument('--num_nodes',
                        type=int,
                        default=30,
                        help='Number of nodes in graph.')
    parser.add_argument('--num_rnn_layers',
                        type=int,
                        default=2,
                        help='Number of RNN layers in encoder and/or decoder.')
    parser.add_argument(
        '--pretrained_num_rnn_layers',
        type=int,
        default=3,
        help='Number of RNN layers in encoder and decoder for SS pre-training.')
    parser.add_argument('--rnn_units',
                        type=int,
                        default=64,
                        help='Number of hidden units in DCRNN.')
    parser.add_argument('--dcgru_activation',
                        type=str,
                        choices=('relu', 'tanh'),
                        default='tanh',
                        help='Nonlinear activation used in DCGRU cells.')
    parser.add_argument('--input_dim',
                        type=int,
                        default=100,
                        help='Input seq feature dim.')

    parser.add_argument(
        '--num_classes',
        type=int,
        default=1,
        help='Number of classes for seizure detection/classification.')
    parser.add_argument('--output_dim',
                        type=int,
                        default=100,
                        help='Output seq feature dim.')
    parser.add_argument('--max_diffusion_step',
                        type=int,
                        default=2,
                        help='Maximum diffusion step.')
    parser.add_argument('--cl_decay_steps',
                        type=int,
                        default=3000,
                        help='Scheduled sampling decay steps.')
    parser.add_argument(
        '--use_curriculum_learning',
        default=False,
        action='store_true',
        help='Whether to use curriculum training for seq-seq model.')
    parser.add_argument(
        '--use_fft',
        default=False,
        action='store_true',
        help='Whether the input data is Fourier transformed EEG signal or raw EEG.')

    # Training/test args
    parser.add_argument('--train_batch_size',
                        type=int,
                        default=40,
                        help='Training batch size.')
    parser.add_argument('--test_batch_size',
                        type=int,
                        default=128,
                        help='Dev/test batch size.')
    parser.add_argument('--num_workers',
                        type=int,
                        default=8,
                        help='Number of sub-processes to use per data loader.')
    parser.add_argument('--dropout',
                        type=float,
                        default=0.0,
                        help='Dropout rate for dropout layer before final FC.')
    parser.add_argument('--eval_every',
                        type=int,
                        default=1,
                        help='Evaluate on dev set every x epoch.')
    parser.add_argument(
        '--metric_name',
        type=str,
        default='auroc',
        choices=(
            'F1',
            'acc',
            'loss',
            'auroc'),
        help='Name of dev metric to determine best checkpoint.')
    parser.add_argument('--lr_init',
                        type=float,
                        default=3e-4,
                        help='Initial learning rate.')
    parser.add_argument('--l2_wd',
                        type=float,
                        default=5e-4,
                        help='L2 weight decay.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=100,
                        help='Number of epochs for training.')
    parser.add_argument('--max_grad_norm',
                        type=float,
                        default=5.0,
                        help='Maximum gradient norm for gradient clipping.')
    parser.add_argument('--metric_avg',
                        type=str,
                        default='weighted',
                        help='weighted, micro or macro.')
    parser.add_argument('--data_augment',
                        default=False,
                        action='store_true',
                        help='Whether perform data augmentation.')
    parser.add_argument('--keep_ratio',
                        default=1,
                        help='keep how many percentage of nodes.')
        
    parser.add_argument('--leave_one_out',
                    type=int,
                    default=1,
                    help='Leave one out subject index')

    args = parser.parse_args()

    # which metric to maximize
    if args.metric_name == 'loss':
        # Best checkpoint is the one that minimizes loss
        args.maximize_metric = False
    elif args.metric_name in ('F1', 'acc', 'auroc'):
        # Best checkpoint is the one that maximizes F1 or acc
        args.maximize_metric = True
    else:
        raise ValueError(
            'Unrecognized metric name: "{}"'.format(
                args.metric_name))

    # must provide load_model_path if testing only
    if (args.load_model_path is None) and not(args.do_train):
        raise ValueError(
            'For evaluation only, please provide trained model checkpoint in argument load_model_path.')

    # filter type
    if args.graph_type == "individual":
        args.filter_type = "dual_random_walk"
    if args.graph_type == "combined":
        args.filter_type = "laplacian"

    return args
