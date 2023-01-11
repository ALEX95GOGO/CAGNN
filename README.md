# CAGNN
A graph neural network that can generate connectivity via end-to-end training

### Training
python train.py --save_dir ./save/ --max_seq_len 10 --do_train --num_epochs 30 --use_fft --lr_init 1e-3 --num_rnn_layers 2 --rnn_units 32 --num_classes 1  --num_nodes 30 --input_dim 3 --test_batch_size 1 --train_batch_size 200 --eval_every 30

Set the num_node to the number of channels in EEG, input_dim equal to the length of feature vector, num_classes equal to the number of classes in the input

### Visualization
python plot_explanable.py

This will plot the explanable figures highlighting the sequence most relevant to the network prediction
