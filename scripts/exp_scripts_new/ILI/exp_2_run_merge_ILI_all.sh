export CUDA_VISIBLE_DEVICES=0

# e_layers 

##### MSE #####

python -u run_dualmode3k.py \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ili_E3k_MSE_36_24 \
  --model B6autoformer \
  --slow_model AutoformerS1 \
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
  --urt_heads 2 \
  --loss 'mse' \
  --corr_penalty 0.5 \
  --learning_rate 0.001 \
  --d_model 256 \
  --itr 3 \
  --fix_seed 2023 \
  --train_epochs 20 \
  --batch_size 32 \
  --checkpoints ./checkpoints2/

# python -u run_dualmode3k.py \
#   --is_training 1 \
#   --root_path ./dataset/illness/ \
#   --data_path national_illness.csv \
#   --model_id ili_invertE3k_MSE_36_24 \
#   --model B6iFast \
#   --slow_model S1iSlow \
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
#   --urt_heads 2 \
#   --loss 'mse' \
#   --corr_penalty 0.5 \
#   --learning_rate 0.001 \
#   --d_model 256 \
#   --itr 3 \
#   --fix_seed 2023 \
#   --train_epochs 20 \
#   --batch_size 32 \
#   --checkpoints ./checkpoints2/

# python -u run_dualmode3k.py \
#   --is_training 1 \
#   --root_path ./dataset/illness/ \
#   --data_path national_illness.csv \
#   --model_id ili_E3k_MSE_36_36 \
#   --model B6autoformer \
#   --slow_model AutoformerS1 \
#   --data custom \
#   --features M \
#   --seq_len 36 \
#   --label_len 18 \
#   --pred_len 36 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --n_learner 3 \
#   --urt_heads 2 \
#   --loss 'mse' \
#   --corr_penalty 0.5 \
#   --learning_rate 0.001 \
#   --d_model 256 \
#   --itr 3 \
#   --fix_seed 2023 \
#   --train_epochs 20 \
#   --batch_size 32 \
#   --checkpoints ./checkpoints2/

# python -u run_dualmode3k.py \
#   --is_training 1 \
#   --root_path ./dataset/illness/ \
#   --data_path national_illness.csv \
#   --model_id ili_invertE3k_MSE_MSE_36_36 \
#   --model B6iFast \
#   --slow_model S1iSlow \
#   --data custom \
#   --features M \
#   --seq_len 36 \
#   --label_len 18 \
#   --pred_len 36 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --n_learner 3 \
#   --urt_heads 2 \
#   --loss 'mse' \
#   --corr_penalty 0.5 \
#   --learning_rate 0.001 \
#   --d_model 256 \
#   --itr 3 \
#   --fix_seed 2023 \
#   --train_epochs 20 \
#   --batch_size 32 \
#   --checkpoints ./checkpoints2/

# python -u run_dualmode3k.py \
#   --is_training 1 \
#   --root_path ./dataset/illness/ \
#   --data_path national_illness.csv \
#   --model_id ili_E3k_MSE_36_48 \
#   --model B6autoformer \
#   --slow_model AutoformerS1 \
#   --data custom \
#   --features M \
#   --seq_len 36 \
#   --label_len 18 \
#   --pred_len 48 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --n_learner 3 \
#   --urt_heads 2 \
#   --loss 'mse' \
#   --corr_penalty 0.5 \
#   --learning_rate 0.001 \
#   --d_model 256 \
#   --itr 3 \
#   --fix_seed 2023 \
#   --train_epochs 20 \
#   --batch_size 32 \
#   --checkpoints ./checkpoints2/

# python -u run_dualmode3k.py \
#   --is_training 1 \
#   --root_path ./dataset/illness/ \
#   --data_path national_illness.csv \
#   --model_id ili_invertE3k_MSE_36_48 \
#   --model B6iFast \
#   --slow_model S1iSlow \
#   --data custom \
#   --features M \
#   --seq_len 36 \
#   --label_len 18 \
#   --pred_len 48 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --n_learner 3 \
#   --urt_heads 2 \
#   --loss 'mse' \
#   --corr_penalty 0.5 \
#   --learning_rate 0.001 \
#   --d_model 256 \
#   --itr 3 \
#   --fix_seed 2023 \
#   --train_epochs 20 \
#   --batch_size 32 \
#   --checkpoints ./checkpoints2/

# python -u run_dualmode3k.py \
#   --is_training 1 \
#   --root_path ./dataset/illness/ \
#   --data_path national_illness.csv \
#   --model_id ili_E3k_MSE_36_60 \
#   --model B6autoformer \
#   --slow_model AutoformerS1 \
#   --data custom \
#   --features M \
#   --seq_len 36 \
#   --label_len 18 \
#   --pred_len 60 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --n_learner 3 \
#   --urt_heads 2 \
#   --loss 'mse' \
#   --corr_penalty 0.5 \
#   --learning_rate 0.001 \
#   --d_model 256 \
#   --itr 3 \
#   --fix_seed 2023 \
#   --train_epochs 20 \
#   --batch_size 32 \
#   --checkpoints ./checkpoints2/

# python -u run_dualmode3k.py \
#   --is_training 1 \
#   --root_path ./dataset/illness/ \
#   --data_path national_illness.csv \
#   --model_id ili_invertE3k_MSE_36_60 \
#   --model B6iFast \
#   --slow_model S1iSlow \
#   --data custom \
#   --features M \
#   --seq_len 36 \
#   --label_len 18 \
#   --pred_len 60 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --n_learner 3 \
#   --urt_heads 2 \
#   --loss 'mse' \
#   --corr_penalty 0.5 \
#   --learning_rate 0.001 \
#   --d_model 256 \
#   --itr 3 \
#   --fix_seed 2023 \
#   --train_epochs 20 \
#   --batch_size 32 \
#   --checkpoints ./checkpoints2/

# python -u run_dualmode3k.py \
#   --is_training 1 \
#   --root_path ./dataset/illness/ \
#   --data_path national_illness.csv \
#   --model_id ili_E3k_MSE_36_72 \
#   --model B6autoformer \
#   --slow_model AutoformerS1 \
#   --data custom \
#   --features M \
#   --seq_len 36 \
#   --label_len 18 \
#   --pred_len 72 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --n_learner 3 \
#   --urt_heads 2 \
#   --loss 'mse' \
#   --corr_penalty 0.5 \
#   --learning_rate 0.001 \
#   --d_model 256 \
#   --itr 3 \
#   --fix_seed 2023 \
#   --train_epochs 20 \
#   --batch_size 32 \
#   --checkpoints ./checkpoints2/

# python -u run_dualmode3k.py \
#   --is_training 1 \
#   --root_path ./dataset/illness/ \
#   --data_path national_illness.csv \
#   --model_id ili_invertE3k_MSE_36_72 \
#   --model B6iFast \
#   --slow_model S1iSlow \
#   --data custom \
#   --features M \
#   --seq_len 36 \
#   --label_len 18 \
#   --pred_len 72 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --n_learner 3 \
#   --urt_heads 2 \
#   --loss 'mse' \
#   --corr_penalty 0.5 \
#   --learning_rate 0.001 \
#   --d_model 256 \
#   --itr 3 \
#   --fix_seed 2023 \
#   --train_epochs 20 \
#   --batch_size 32 \
#   --checkpoints ./checkpoints2/

# python -u run_dualmode3k.py \
#   --is_training 1 \
#   --root_path ./dataset/illness/ \
#   --data_path national_illness.csv \
#   --model_id ili_E3k_MSE_36_90 \
#   --model B6autoformer \
#   --slow_model AutoformerS1 \
#   --data custom \
#   --features M \
#   --seq_len 36 \
#   --label_len 18 \
#   --pred_len 90 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --n_learner 3 \
#   --urt_heads 2 \
#   --loss 'mse' \
#   --corr_penalty 0.5 \
#   --learning_rate 0.001 \
#   --d_model 256 \
#   --itr 3 \
#   --fix_seed 2023 \
#   --train_epochs 20 \
#   --batch_size 32 \
#   --checkpoints ./checkpoints2/

# python -u run_dualmode3k.py \
#   --is_training 1 \
#   --root_path ./dataset/illness/ \
#   --data_path national_illness.csv \
#   --model_id ili_invertE3k_MSE_36_90 \
#   --model B6iFast \
#   --slow_model S1iSlow \
#   --data custom \
#   --features M \
#   --seq_len 36 \
#   --label_len 18 \
#   --pred_len 90 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --n_learner 3 \
#   --urt_heads 2 \
#   --loss 'mse' \
#   --corr_penalty 0.5 \
#   --learning_rate 0.001 \
#   --d_model 256 \
#   --itr 3 \
#   --fix_seed 2023 \
#   --train_epochs 20 \
#   --batch_size 32 \
#   --checkpoints ./checkpoints2/

# ##### MSE #####

# ##### NEGATIVE CORRELATION #####

# python -u run_dualmode3k.py \
#   --is_training 1 \
#   --root_path ./dataset/illness/ \
#   --data_path national_illness.csv \
#   --model_id ili_E3k_Neg_36_24 \
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
#   --urt_heads 2 \
#   --loss 'neg_corr' \
#   --corr_penalty 0.5 \
#   --learning_rate 0.001 \
#   --d_model 256 \
#   --itr 3 \
#   --fix_seed 2023 \
#   --train_epochs 20 \
#   --batch_size 32 \
#   --checkpoints ./checkpoints2/

# python -u run_dualmode3k.py \
#   --is_training 1 \
#   --root_path ./dataset/illness/ \
#   --data_path national_illness.csv \
#   --model_id ili_invertE3k_MSE_36_24 \
#   --model B6iFast \
#   --slow_model S1iSlow \
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
#   --urt_heads 2 \
#   --loss 'mse' \
#   --corr_penalty 0.5 \
#   --learning_rate 0.001 \
#   --d_model 256 \
#   --itr 3 \
#   --fix_seed 2023 \
#   --train_epochs 20 \
#   --batch_size 32 \
#   --checkpoints ./checkpoints2/

# python -u run_dualmode3k.py \
#   --is_training 1 \
#   --root_path ./dataset/illness/ \
#   --data_path national_illness.csv \
#   --model_id ili_E3k_Neg_36_36 \
#   --model B6autoformer \
#   --slow_model AutoformerS1 \
#   --data custom \
#   --features M \
#   --seq_len 36 \
#   --label_len 18 \
#   --pred_len 36 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --n_learner 3 \
#   --urt_heads 2 \
#   --loss 'neg_corr' \
#   --corr_penalty 0.5 \
#   --learning_rate 0.001 \
#   --d_model 256 \
#   --itr 3 \
#   --fix_seed 2023 \
#   --train_epochs 20 \
#   --batch_size 32 \
#   --checkpoints ./checkpoints2/

# python -u run_dualmode3k.py \
#   --is_training 1 \
#   --root_path ./dataset/illness/ \
#   --data_path national_illness.csv \
#   --model_id ili_invertE3k_MSE_MSE_36_36 \
#   --model B6iFast \
#   --slow_model S1iSlow \
#   --data custom \
#   --features M \
#   --seq_len 36 \
#   --label_len 18 \
#   --pred_len 36 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --n_learner 3 \
#   --urt_heads 2 \
#   --loss 'mse' \
#   --corr_penalty 0.5 \
#   --learning_rate 0.001 \
#   --d_model 256 \
#   --itr 3 \
#   --fix_seed 2023 \
#   --train_epochs 20 \
#   --batch_size 32 \
#   --checkpoints ./checkpoints2/

# python -u run_dualmode3k.py \
#   --is_training 1 \
#   --root_path ./dataset/illness/ \
#   --data_path national_illness.csv \
#   --model_id ili_E3k_Neg_36_48 \
#   --model B6autoformer \
#   --slow_model AutoformerS1 \
#   --data custom \
#   --features M \
#   --seq_len 36 \
#   --label_len 18 \
#   --pred_len 48 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --n_learner 3 \
#   --urt_heads 2 \
#   --loss 'neg_corr' \
#   --corr_penalty 0.5 \
#   --learning_rate 0.001 \
#   --d_model 256 \
#   --itr 3 \
#   --fix_seed 2023 \
#   --train_epochs 20 \
#   --batch_size 32 \
#   --checkpoints ./checkpoints2/

# python -u run_dualmode3k.py \
#   --is_training 1 \
#   --root_path ./dataset/illness/ \
#   --data_path national_illness.csv \
#   --model_id ili_invertE3k_MSE_36_48 \
#   --model B6iFast \
#   --slow_model S1iSlow \
#   --data custom \
#   --features M \
#   --seq_len 36 \
#   --label_len 18 \
#   --pred_len 48 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --n_learner 3 \
#   --urt_heads 2 \
#   --loss 'mse' \
#   --corr_penalty 0.5 \
#   --learning_rate 0.001 \
#   --d_model 256 \
#   --itr 3 \
#   --fix_seed 2023 \
#   --train_epochs 20 \
#   --batch_size 32 \
#   --checkpoints ./checkpoints2/

# python -u run_dualmode3k.py \
#   --is_training 1 \
#   --root_path ./dataset/illness/ \
#   --data_path national_illness.csv \
#   --model_id ili_E3k_Neg_36_60 \
#   --model B6autoformer \
#   --slow_model AutoformerS1 \
#   --data custom \
#   --features M \
#   --seq_len 36 \
#   --label_len 18 \
#   --pred_len 60 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --n_learner 3 \
#   --urt_heads 2 \
#   --loss 'neg_corr' \
#   --corr_penalty 0.5 \
#   --learning_rate 0.001 \
#   --d_model 256 \
#   --itr 3 \
#   --fix_seed 2023 \
#   --train_epochs 20 \
#   --batch_size 32 \
#   --checkpoints ./checkpoints2/

# python -u run_dualmode3k.py \
#   --is_training 1 \
#   --root_path ./dataset/illness/ \
#   --data_path national_illness.csv \
#   --model_id ili_invertE3k_MSE_36_60 \
#   --model B6iFast \
#   --slow_model S1iSlow \
#   --data custom \
#   --features M \
#   --seq_len 36 \
#   --label_len 18 \
#   --pred_len 60 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --n_learner 3 \
#   --urt_heads 2 \
#   --loss 'mse' \
#   --corr_penalty 0.5 \
#   --learning_rate 0.001 \
#   --d_model 256 \
#   --itr 3 \
#   --fix_seed 2023 \
#   --train_epochs 20 \
#   --batch_size 32 \
#   --checkpoints ./checkpoints2/

# python -u run_dualmode3k.py \
#   --is_training 1 \
#   --root_path ./dataset/illness/ \
#   --data_path national_illness.csv \
#   --model_id ili_E3k_Neg_36_72 \
#   --model B6autoformer \
#   --slow_model AutoformerS1 \
#   --data custom \
#   --features M \
#   --seq_len 36 \
#   --label_len 18 \
#   --pred_len 72 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --n_learner 3 \
#   --urt_heads 2 \
#   --loss 'neg_corr' \
#   --corr_penalty 0.5 \
#   --learning_rate 0.001 \
#   --d_model 256 \
#   --itr 3 \
#   --fix_seed 2023 \
#   --train_epochs 20 \
#   --batch_size 32 \
#   --checkpoints ./checkpoints2/

# python -u run_dualmode3k.py \
#   --is_training 1 \
#   --root_path ./dataset/illness/ \
#   --data_path national_illness.csv \
#   --model_id ili_invertE3k_MSE_36_72 \
#   --model B6iFast \
#   --slow_model S1iSlow \
#   --data custom \
#   --features M \
#   --seq_len 36 \
#   --label_len 18 \
#   --pred_len 72 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --n_learner 3 \
#   --urt_heads 2 \
#   --loss 'mse' \
#   --corr_penalty 0.5 \
#   --learning_rate 0.001 \
#   --d_model 256 \
#   --itr 3 \
#   --fix_seed 2023 \
#   --train_epochs 20 \
#   --batch_size 32 \
#   --checkpoints ./checkpoints2/

# python -u run_dualmode3k.py \
#   --is_training 1 \
#   --root_path ./dataset/illness/ \
#   --data_path national_illness.csv \
#   --model_id ili_E3k_Neg_36_90 \
#   --model B6autoformer \
#   --slow_model AutoformerS1 \
#   --data custom \
#   --features M \
#   --seq_len 36 \
#   --label_len 18 \
#   --pred_len 90 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --n_learner 3 \
#   --urt_heads 2 \
#   --loss 'neg_corr' \
#   --corr_penalty 0.5 \
#   --learning_rate 0.001 \
#   --d_model 256 \
#   --itr 3 \
#   --fix_seed 2023 \
#   --train_epochs 20 \
#   --batch_size 32 \
#   --checkpoints ./checkpoints2/

# python -u run_dualmode3k.py \
#   --is_training 1 \
#   --root_path ./dataset/illness/ \
#   --data_path national_illness.csv \
#   --model_id ili_invertE3k_MSE_36_90 \
#   --model B6iFast \
#   --slow_model S1iSlow \
#   --data custom \
#   --features M \
#   --seq_len 36 \
#   --label_len 18 \
#   --pred_len 90 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --n_learner 3 \
#   --urt_heads 2 \
#   --loss 'mse' \
#   --corr_penalty 0.5 \
#   --learning_rate 0.001 \
#   --d_model 256 \
#   --itr 3 \
#   --fix_seed 2023 \
#   --train_epochs 20 \
#   --batch_size 32 \
#   --checkpoints ./checkpoints2/

# ##### NEGATIVE CORRELATION #####