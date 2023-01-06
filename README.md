# CAGNN

To train a model, run: python train.py --save_dir ./save/ --max_seq_len 10 --do_train --num_epochs 30 --use_fft --lr_init 1e-3 --num_rnn_layers 2 --rnn_units 32 --max_diffusion_step 2 --num_classes 1  --graph_type combined --num_nodes 30 --input_dim 3 --test_batch_size 320 --train_batch_size 200 --eval_every 30

To visualize the explanable results, run: python plot_explanable.py
