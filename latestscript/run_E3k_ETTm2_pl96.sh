#export CUDA_VISIBLE_DEVICES=2


python -u run_dualmode3k.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id E3k_ETTm2_96_96 \
  --model B6autoformer \
  --slow_model AutoformerS1 \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --n_learner 3 \
  --urt_head 1\
  --learning_rate 0.001 \
  --factor 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --train_epochs 20 \
  --itr 3 \
  --fix_seed 2023 \
  --dropout 0.1 \
  --d_model 256 \
  --checkpoints ./checkpoints0/





#train_epochs 1 \

