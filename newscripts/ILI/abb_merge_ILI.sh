export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ILI_RL_36_24_abb1 \
  --model B6iFast \
  --slow_model S1iSlow \
  --data custom \
  --features MS \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_learner 3 \
  --urt_heads 1 \
  --learning_rate 0.001 \
  --d_model 512 \
  --d_ff 2048 \
  --itr 1 \
  --fix_seed 2021 \
  --train_epochs 20 \
  --batch_size 32 \
  --patience 3 \
  --rl_seed 42 \
  --use_weight 1 \
  --use_td 1 \
  --use_extra 1 \
  --use_pretrain 1 \
  --epsilon 0.3 \
  --RL_epoch 1 \
  --RL_warmup_epoch 100 \
  --RL_pretrain_epoch 200 \
  --RL_step_size 450 \
  --RL_max_patience 5 \
  --gamma 0.99 \
  --tau 0.005 \
  --hidden_dim 128 \
  --checkpoints ./checkpoints/

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ILI_RL_36_36_abb1 \
  --model B6iFast \
  --slow_model S1iSlow \
  --data custom \
  --features MS \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 36 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_learner 3 \
  --urt_heads 1 \
  --learning_rate 0.001 \
  --d_model 512 \
  --d_ff 2048 \
  --itr 1 \
  --fix_seed 2021 \
  --train_epochs 20 \
  --batch_size 32 \
  --patience 3 \
  --rl_seed 42 \
  --use_weight 1 \
  --use_td 1 \
  --use_extra 1 \
  --use_pretrain 1 \
  --epsilon 0.3 \
  --RL_epoch 1 \
  --RL_warmup_epoch 100 \
  --RL_pretrain_epoch 200 \
  --RL_step_size 450 \
  --RL_max_patience 5 \
  --gamma 0.99 \
  --tau 0.005 \
  --hidden_dim 128 \
  --checkpoints ./checkpoints/

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ILI_RL_36_48_abb1 \
  --model B6iFast \
  --slow_model S1iSlow \
  --data custom \
  --features MS \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 48 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_learner 3 \
  --urt_heads 1 \
  --learning_rate 0.001 \
  --d_model 512 \
  --d_ff 2048 \
  --itr 1 \
  --fix_seed 2021 \
  --train_epochs 20 \
  --batch_size 32 \
  --patience 3 \
  --rl_seed 42 \
  --use_weight 1 \
  --use_td 1 \
  --use_extra 1 \
  --use_pretrain 1 \
  --epsilon 0.3 \
  --RL_epoch 1 \
  --RL_warmup_epoch 100 \
  --RL_pretrain_epoch 200 \
  --RL_step_size 450 \
  --RL_max_patience 5 \
  --gamma 0.99 \
  --tau 0.005 \
  --hidden_dim 128 \
  --checkpoints ./checkpoints/

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ILI_RL_36_60_abb1 \
  --model B6iFast \
  --slow_model S1iSlow \
  --data custom \
  --features MS \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 60 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_learner 3 \
  --urt_heads 1 \
  --learning_rate 0.001 \
  --d_model 512 \
  --d_ff 2048 \
  --itr 1 \
  --fix_seed 2021 \
  --train_epochs 20 \
  --batch_size 32 \
  --patience 3 \
  --rl_seed 42 \
  --use_weight 1 \
  --use_td 1 \
  --use_extra 1 \
  --use_pretrain 1 \
  --epsilon 0.3 \
  --RL_epoch 1 \
  --RL_warmup_epoch 100 \
  --RL_pretrain_epoch 200 \
  --RL_step_size 450 \
  --RL_max_patience 5 \
  --gamma 0.99 \
  --tau 0.005 \
  --hidden_dim 128 \
  --checkpoints ./checkpoints/

  python -u run.py \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ILI_RL_36_24_abb2 \
  --model B6iFast \
  --slow_model S1iSlow \
  --data custom \
  --features MS \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_learner 3 \
  --urt_heads 1 \
  --learning_rate 0.001 \
  --d_model 512 \
  --d_ff 2048 \
  --itr 1 \
  --fix_seed 2021 \
  --train_epochs 20 \
  --batch_size 32 \
  --patience 3 \
  --rl_seed 42 \
  --use_weight 1 \
  --use_td 1 \
  --use_extra 1 \
  --use_pretrain 1 \
  --epsilon 0.3 \
  --RL_epoch 1 \
  --RL_warmup_epoch 100 \
  --RL_pretrain_epoch 200 \
  --RL_step_size 256 \
  --RL_max_patience 5 \
  --gamma 0.99 \
  --tau 0.005 \
  --hidden_dim 128 \
  --checkpoints ./checkpoints/

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ILI_RL_36_36_abb2 \
  --model B6iFast \
  --slow_model S1iSlow \
  --data custom \
  --features MS \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 36 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_learner 3 \
  --urt_heads 1 \
  --learning_rate 0.001 \
  --d_model 512 \
  --d_ff 2048 \
  --itr 1 \
  --fix_seed 2021 \
  --train_epochs 20 \
  --batch_size 32 \
  --patience 3 \
  --rl_seed 42 \
  --use_weight 1 \
  --use_td 1 \
  --use_extra 1 \
  --use_pretrain 1 \
  --epsilon 0.3 \
  --RL_epoch 1 \
  --RL_warmup_epoch 100 \
  --RL_pretrain_epoch 200 \
  --RL_step_size 256 \
  --RL_max_patience 5 \
  --gamma 0.99 \
  --tau 0.005 \
  --hidden_dim 128 \
  --checkpoints ./checkpoints/

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ILI_RL_36_48_abb2 \
  --model B6iFast \
  --slow_model S1iSlow \
  --data custom \
  --features MS \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 48 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_learner 3 \
  --urt_heads 1 \
  --learning_rate 0.001 \
  --d_model 512 \
  --d_ff 2048 \
  --itr 1 \
  --fix_seed 2021 \
  --train_epochs 20 \
  --batch_size 32 \
  --patience 3 \
  --rl_seed 42 \
  --use_weight 1 \
  --use_td 1 \
  --use_extra 1 \
  --use_pretrain 1 \
  --epsilon 0.3 \
  --RL_epoch 1 \
  --RL_warmup_epoch 100 \
  --RL_pretrain_epoch 200 \
  --RL_step_size 256 \
  --RL_max_patience 5 \
  --gamma 0.99 \
  --tau 0.005 \
  --hidden_dim 128 \
  --checkpoints ./checkpoints/

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ILI_RL_36_60_abb2 \
  --model B6iFast \
  --slow_model S1iSlow \
  --data custom \
  --features MS \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 60 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_learner 3 \
  --urt_heads 1 \
  --learning_rate 0.001 \
  --d_model 512 \
  --d_ff 2048 \
  --itr 1 \
  --fix_seed 2021 \
  --train_epochs 20 \
  --batch_size 32 \
  --patience 3 \
  --rl_seed 42 \
  --use_weight 1 \
  --use_td 1 \
  --use_extra 1 \
  --use_pretrain 1 \
  --epsilon 0.3 \
  --RL_epoch 1 \
  --RL_warmup_epoch 100 \
  --RL_pretrain_epoch 200 \
  --RL_step_size 256 \
  --RL_max_patience 5 \
  --gamma 0.99 \
  --tau 0.005 \
  --hidden_dim 128 \
  --checkpoints ./checkpoints/

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ILI_RL_36_24_abb3 \
  --model B6iFast \
  --slow_model S1iSlow \
  --data custom \
  --features MS \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_learner 3 \
  --urt_heads 1 \
  --learning_rate 0.001 \
  --d_model 512 \
  --d_ff 2048 \
  --itr 1 \
  --fix_seed 2021 \
  --train_epochs 20 \
  --batch_size 32 \
  --patience 3 \
  --rl_seed 42 \
  --use_weight 1 \
  --use_td 1 \
  --use_extra 1 \
  --use_pretrain 1 \
  --epsilon 0.3 \
  --RL_epoch 3 \
  --RL_warmup_epoch 100 \
  --RL_pretrain_epoch 200 \
  --RL_step_size 450 \
  --RL_max_patience 5 \
  --gamma 0.99 \
  --tau 0.005 \
  --hidden_dim 128 \
  --checkpoints ./checkpoints/

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ILI_RL_36_36_abb3 \
  --model B6iFast \
  --slow_model S1iSlow \
  --data custom \
  --features MS \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 36 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_learner 3 \
  --urt_heads 1 \
  --learning_rate 0.001 \
  --d_model 512 \
  --d_ff 2048 \
  --itr 1 \
  --fix_seed 2021 \
  --train_epochs 20 \
  --batch_size 32 \
  --patience 3 \
  --rl_seed 42 \
  --use_weight 1 \
  --use_td 1 \
  --use_extra 1 \
  --use_pretrain 1 \
  --epsilon 0.3 \
  --RL_epoch 3 \
  --RL_warmup_epoch 100 \
  --RL_pretrain_epoch 200 \
  --RL_step_size 450 \
  --RL_max_patience 5 \
  --gamma 0.99 \
  --tau 0.005 \
  --hidden_dim 128 \
  --checkpoints ./checkpoints/

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ILI_RL_36_48_abb3 \
  --model B6iFast \
  --slow_model S1iSlow \
  --data custom \
  --features MS \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 48 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_learner 3 \
  --urt_heads 1 \
  --learning_rate 0.001 \
  --d_model 512 \
  --d_ff 2048 \
  --itr 1 \
  --fix_seed 2021 \
  --train_epochs 20 \
  --batch_size 32 \
  --patience 3 \
  --rl_seed 42 \
  --use_weight 1 \
  --use_td 1 \
  --use_extra 1 \
  --use_pretrain 1 \
  --epsilon 0.3 \
  --RL_epoch 3 \
  --RL_warmup_epoch 100 \
  --RL_pretrain_epoch 200 \
  --RL_step_size 450 \
  --RL_max_patience 5 \
  --gamma 0.99 \
  --tau 0.005 \
  --hidden_dim 128 \
  --checkpoints ./checkpoints/

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ILI_RL_36_60_abb3 \
  --model B6iFast \
  --slow_model S1iSlow \
  --data custom \
  --features MS \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 60 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_learner 3 \
  --urt_heads 1 \
  --learning_rate 0.001 \
  --d_model 512 \
  --d_ff 2048 \
  --itr 1 \
  --fix_seed 2021 \
  --train_epochs 20 \
  --batch_size 32 \
  --patience 3 \
  --rl_seed 42 \
  --use_weight 1 \
  --use_td 1 \
  --use_extra 1 \
  --use_pretrain 1 \
  --epsilon 0.3 \
  --RL_epoch 3 \
  --RL_warmup_epoch 100 \
  --RL_pretrain_epoch 200 \
  --RL_step_size 450 \
  --RL_max_patience 5 \
  --gamma 0.99 \
  --tau 0.005 \
  --hidden_dim 128 \
  --checkpoints ./checkpoints/

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ILI_RL_36_24_abb3 \
  --model B6iFast \
  --slow_model S1iSlow \
  --data custom \
  --features MS \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_learner 3 \
  --urt_heads 1 \
  --learning_rate 0.001 \
  --d_model 512 \
  --d_ff 2048 \
  --itr 1 \
  --fix_seed 2021 \
  --train_epochs 20 \
  --batch_size 32 \
  --patience 3 \
  --rl_seed 42 \
  --use_weight 1 \
  --use_td 1 \
  --use_extra 1 \
  --use_pretrain 1 \
  --epsilon 0.3 \
  --RL_epoch 3 \
  --RL_warmup_epoch 100 \
  --RL_pretrain_epoch 200 \
  --RL_step_size 256 \
  --RL_max_patience 5 \
  --gamma 0.99 \
  --tau 0.005 \
  --hidden_dim 128 \
  --checkpoints ./checkpoints/

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ILI_RL_36_36_abb3 \
  --model B6iFast \
  --slow_model S1iSlow \
  --data custom \
  --features MS \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 36 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_learner 3 \
  --urt_heads 1 \
  --learning_rate 0.001 \
  --d_model 512 \
  --d_ff 2048 \
  --itr 1 \
  --fix_seed 2021 \
  --train_epochs 20 \
  --batch_size 32 \
  --patience 3 \
  --rl_seed 42 \
  --use_weight 1 \
  --use_td 1 \
  --use_extra 1 \
  --use_pretrain 1 \
  --epsilon 0.3 \
  --RL_epoch 3 \
  --RL_warmup_epoch 100 \
  --RL_pretrain_epoch 200 \
  --RL_step_size 256 \
  --RL_max_patience 5 \
  --gamma 0.99 \
  --tau 0.005 \
  --hidden_dim 128 \
  --checkpoints ./checkpoints/

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ILI_RL_36_48_abb3 \
  --model B6iFast \
  --slow_model S1iSlow \
  --data custom \
  --features MS \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 48 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_learner 3 \
  --urt_heads 1 \
  --learning_rate 0.001 \
  --d_model 512 \
  --d_ff 2048 \
  --itr 1 \
  --fix_seed 2021 \
  --train_epochs 20 \
  --batch_size 32 \
  --patience 3 \
  --rl_seed 42 \
  --use_weight 1 \
  --use_td 1 \
  --use_extra 1 \
  --use_pretrain 1 \
  --epsilon 0.3 \
  --RL_epoch 3 \
  --RL_warmup_epoch 100 \
  --RL_pretrain_epoch 200 \
  --RL_step_size 256 \
  --RL_max_patience 5 \
  --gamma 0.99 \
  --tau 0.005 \
  --hidden_dim 128 \
  --checkpoints ./checkpoints/

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ILI_RL_36_60_abb3 \
  --model B6iFast \
  --slow_model S1iSlow \
  --data custom \
  --features MS \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 60 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_learner 3 \
  --urt_heads 1 \
  --learning_rate 0.001 \
  --d_model 512 \
  --d_ff 2048 \
  --itr 1 \
  --fix_seed 2021 \
  --train_epochs 20 \
  --batch_size 32 \
  --patience 3 \
  --rl_seed 42 \
  --use_weight 1 \
  --use_td 1 \
  --use_extra 1 \
  --use_pretrain 1 \
  --epsilon 0.3 \
  --RL_epoch 3 \
  --RL_warmup_epoch 100 \
  --RL_pretrain_epoch 200 \
  --RL_step_size 256 \
  --RL_max_patience 5 \
  --gamma 0.99 \
  --tau 0.005 \
  --hidden_dim 128 \
  --checkpoints ./checkpoints/