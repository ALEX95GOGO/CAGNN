# CAGNN
A graph neural network that can generate connectivity via end-to-end training

### Training
python train.py --save_dir ./save/ --max_seq_len 10 --do_train --num_epochs 30 --metric_name F1 --use_fft --lr_init 1e-3 --num_rnn_layers 2 --rnn_units 32 --max_diffusion_step 2 --num_classes 1 --dropout 0 --graph_type combined --num_nodes 30 --input_dim 3 --data_augment --test_batch_size 320 --train_batch_size 200 --eval_every 1--rand_seed 0 --num_epochs 30
Set the num_node to the number of channels in EEG, input_dim equal to the length of feature vector, num_classes equal to the number of classes in the input

### Visualization
python plot_explanability.py

This will plot the explanable figures highlighting the sequence most relevant to the network prediction
