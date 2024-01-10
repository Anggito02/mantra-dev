python -u run_dualmode3k.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_E3k_96_96 \
  --model B6autoformer \
  --slow_model AutoformerS1 \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --n_learner 3 \
  --urt_heads 1 \
  --itr 3 \
  --learning_rate 0.0001 \
  --anomaly 1.0 \
  --train_epochs 20 \
  --fix_seed 2023 \
  --dropout 0.001 \
  --d_model 256 \
  --checkpoints ./checkpoints0/
