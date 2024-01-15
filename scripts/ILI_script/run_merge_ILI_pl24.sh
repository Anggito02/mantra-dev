export CUDA_VISIBLE_DEVICES=0

# python -u run_dualmode3k.py \
#   --is_training 1 \
#   --root_path ./dataset/illness/ \
#   --data_path national_illness.csv \
#   --model_id ili_invertE3k_36_24 \
#   --model B6autoformer \
#   --slow_model AutoformerS1 \
#   --data custom \
#   --features M \
#   --seq_len 36 \
#   --label_len 18 \
#   --pred_len 24 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --n_learner 3 \
#   --urt_heads 1 \
#   --learning_rate 0.001 \
#   --d_model 256 \
#   --itr 3 \
#   --fix_seed 2023 \
#   --train_epochs 20 \
#   --batch_size 32 \
#   --checkpoints ./checkpoints1/

python -u run_dualmode3k.py \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ili_invertE3k_36_24 \
  --model B6iFast \
  --slow_model S1iSlow \
  --data custom \
  --features M \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_learner 3 \
  --urt_heads 1 \
  --learning_rate 0.001 \
  --d_model 256 \
  --itr 3 \
  --fix_seed 2023 \
  --train_epochs 20 \
  --batch_size 32 \
  --checkpoints ./checkpoints1/