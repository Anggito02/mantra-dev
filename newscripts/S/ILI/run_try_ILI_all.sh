export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ILI_RL_36_24_try \
  --model B6iFast \
  --slow_model S1iSlow \
  --data custom \
  --features S \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 1 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
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
  --epsilon 1 \
  --RL_epoch 10 \
  --RL_warmup_epoch 100 \
  --RL_pretrain_epoch 200 \
  --RL_step_size 24 \
  --RL_max_patience 5 \
  --gamma 0.99 \
  --tau 0.001 \
  --hidden_dim 256 \
  --checkpoints ./checkpoints/