export CUDA_VISIBLE_DEVICES=0

# 96/96
# python -u run_open_net.py \
#     --is_training 1 \
#     --root_path ./dataset/exchange_rate/ \
#     --data_path exchange_rate.csv \
#     --model_id Exchange_96_96 \
#     --model B6iFast \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --label_len 48 \
#     --pred_len 96 \
#     --e_layers 3 \
#     --d_layers 1 \
#     --factor 1 \
#     --enc_in 8 \
#     --dec_in 8 \
#     --c_out 8 \
#     --des 'normal_0' \
#     --n_learner 3 \
#     --urt_heads 1 \
#     --learning_rate 0.001 \
#     --d_model 512 \
#     --d_ff 512 \
#     --itr 1 \
#     --fix_seed 2021 \
#     --train_epochs 20 \
#     --batch_size 32 \
#     --patience 5 \
#     --checkpoints ./checkpoints/ \

# python -u run_rlmc.py \
#     --model_id Exchange_96_96 \
#     --model B6iFast \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --label_len 48 \
#     --pred_len 96 \
#     --e_layers 3 \
#     --d_layers 1 \
#     --des 'normal_0' \
#     --d_model 512 \
#     --seed 42 \
#     --use_weight 0 \
#     --use_td 1 \
#     --use_extra 1 \
#     --use_pretrain 1 \
#     --epsilon 1 \
#     --gamma 0.99 \
#     --tau 0.005 \
#     --exp_name Exchange_96_96

# python -u run_open_net.py \
#     --is_training 1 \
#     --root_path ./dataset/exchange_rate/ \
#     --data_path exchange_rate.csv \
#     --model_id Exchange_96_96 \
#     --model B6iFast \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --label_len 48 \
#     --pred_len 96 \
#     --e_layers 3 \
#     --d_layers 1 \
#     --factor 1 \
#     --enc_in 8 \
#     --dec_in 8 \
#     --c_out 8 \
#     --des 'normal_1' \
#     --n_learner 3 \
#     --urt_heads 1 \
#     --learning_rate 0.001 \
#     --d_model 512 \
#     --d_ff 512 \
#     --itr 1 \
#     --fix_seed 2022 \
#     --train_epochs 20 \
#     --batch_size 32 \
#     --patience 5 \
#     --checkpoints ./checkpoints/

# python -u run_rlmc.py \
#     --model_id Exchange_96_96 \
#     --model B6iFast \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --label_len 48 \
#     --pred_len 96 \
#     --e_layers 3 \
#     --d_layers 1 \
#     --des 'normal_1' \
#     --d_model 512 \
#     --seed 2021 \
#     --use_weight 0 \
#     --use_td 1 \
#     --use_extra 1 \
#     --use_pretrain 1 \
#     --epsilon 1 \
#     --gamma 0.99 \
#     --tau 0.005 \
#     --exp_name Exchange_96_96

# python -u run_open_net.py \
#     --is_training 1 \
#     --root_path ./dataset/exchange_rate/ \
#     --data_path exchange_rate.csv \
#     --model_id Exchange_96_96 \
#     --model B6iFast \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --label_len 48 \
#     --pred_len 96 \
#     --e_layers 3 \
#     --d_layers 1 \
#     --factor 1 \
#     --enc_in 8 \
#     --dec_in 8 \
#     --c_out 8 \
#     --des 'normal_2' \
#     --n_learner 3 \
#     --urt_heads 1 \
#     --learning_rate 0.001 \
#     --d_model 512 \
#     --d_ff 512 \
#     --itr 1 \
#     --fix_seed 2023 \
#     --train_epochs 20 \
#     --batch_size 32 \
#     --patience 5 \
#     --checkpoints ./checkpoints/

# python -u run_rlmc.py \
#     --model_id Exchange_96_96 \
#     --model B6iFast \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --label_len 48 \
#     --pred_len 96 \
#     --e_layers 3 \
#     --d_layers 1 \
#     --des 'normal_2' \
#     --d_model 512 \
#     --seed 2022 \
#     --use_weight 0 \
#     --use_td 1 \
#     --use_extra 1 \
#     --use_pretrain 1 \
#     --epsilon 1 \
#     --gamma 0.99 \
#     --tau 0.005 \
#     --exp_name Exchange_96_96

# python -u run_open_net.py \
#     --is_training 1 \
#     --root_path ./dataset/exchange_rate/ \
#     --data_path exchange_rate.csv \
#     --model_id Exchange_96_192 \
#     --model B6iFast \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --label_len 48 \
#     --pred_len 192 \
#     --e_layers 3 \
#     --d_layers 1 \
#     --factor 1 \
#     --enc_in 8 \
#     --dec_in 8 \
#     --c_out 8 \
#     --des 'normal_0' \
#     --n_learner 3 \
#     --urt_heads 1 \
#     --learning_rate 0.0001 \
#     --d_model 512 \
#     --d_ff 512 \
#     --itr 1 \
#     --fix_seed 2020 \
#     --train_epochs 20 \
#     --batch_size 32 \
#     --patience 5 \
#     --checkpoints ./checkpoints/

# python -u run_rlmc.py \
#     --model_id Exchange_96_192 \
#     --model B6iFast \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --label_len 48 \
#     --pred_len 192 \
#     --e_layers 3 \
#     --d_layers 1 \
#     --des 'normal_0' \
#     --d_model 512 \
#     --seed 2020 \
#     --use_weight 0 \
#     --use_td 1 \
#     --use_extra 1 \
#     --use_pretrain 1 \
#     --epsilon 0.7 \
#     --gamma 0.99 \
#     --tau 0.001 \
#     --exp_name Exchange_96_192

# python -u run_open_net.py \
#     --is_training 1 \
#     --root_path ./dataset/exchange_rate/ \
#     --data_path exchange_rate.csv \
#     --model_id Exchange_96_192 \
#     --model B6iFast \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --label_len 48 \
#     --pred_len 192 \
#     --e_layers 3 \
#     --d_layers 1 \
#     --factor 1 \
#     --enc_in 8 \
#     --dec_in 8 \
#     --c_out 8 \
#     --des 'normal_1' \
#     --n_learner 3 \
#     --urt_heads 1 \
#     --learning_rate 0.0001 \
#     --d_model 512 \
#     --d_ff 512 \
#     --itr 1 \
#     --fix_seed 2021 \
#     --train_epochs 20 \
#     --batch_size 32 \
#     --patience 5 \
#     --checkpoints ./checkpoints/

# python -u run_rlmc.py \
#     --model_id Exchange_96_192 \
#     --model B6iFast \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --label_len 48 \
#     --pred_len 192 \
#     --e_layers 3 \
#     --d_layers 1 \
#     --des 'normal_1' \
#     --d_model 512 \
#     --seed 2021 \
#     --use_weight 0 \
#     --use_td 1 \
#     --use_extra 1 \
#     --use_pretrain 1 \
#     --epsilon 0.7 \
#     --gamma 0.99 \
#     --tau 0.001 \
#     --exp_name Exchange_96_192

# python -u run_open_net.py \
#     --is_training 1 \
#     --root_path ./dataset/exchange_rate/ \
#     --data_path exchange_rate.csv \
#     --model_id Exchange_96_192 \
#     --model B6iFast \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --label_len 48 \
#     --pred_len 192 \
#     --e_layers 3 \
#     --d_layers 1 \
#     --factor 1 \
#     --enc_in 8 \
#     --dec_in 8 \
#     --c_out 8 \
#     --des 'normal_2' \
#     --n_learner 3 \
#     --urt_heads 1 \
#     --learning_rate 0.0001 \
#     --d_model 512 \
#     --d_ff 512 \
#     --itr 1 \
#     --fix_seed 2022 \
#     --train_epochs 20 \
#     --batch_size 32 \
#     --patience 5 \
#     --checkpoints ./checkpoints/

# python -u run_rlmc.py \
#     --model_id Exchange_96_192 \
#     --model B6iFast \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --label_len 48 \
#     --pred_len 192 \
#     --e_layers 3 \
#     --d_layers 1 \
#     --des 'normal_2' \
#     --d_model 512 \
#     --seed 2022 \
#     --use_weight 0 \
#     --use_td 1 \
#     --use_extra 1 \
#     --use_pretrain 1 \
#     --epsilon 0.7 \
#     --gamma 0.99 \
#     --tau 0.001 \
#     --exp_name Exchange_96_192

# python -u run_open_net.py \
#     --is_training 1 \
#     --root_path ./dataset/exchange_rate/ \
#     --data_path exchange_rate.csv \
#     --model_id Exchange_96_336 \
#     --model B6iFast \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --label_len 48 \
#     --pred_len 336 \
#     --e_layers 3 \
#     --d_layers 1 \
#     --factor 1 \
#     --enc_in 8 \
#     --dec_in 8 \
#     --c_out 8 \
#     --des 'normal_0' \
#     --n_learner 3 \
#     --urt_heads 1 \
#     --learning_rate 0.0001 \
#     --d_model 512 \
#     --d_ff 512 \
#     --itr 1 \
#     --fix_seed 2021 \
#     --train_epochs 20 \
#     --batch_size 32 \
#     --patience 5 \
#     --checkpoints ./checkpoints/

# python -u run_rlmc.py \
#     --model_id Exchange_96_336 \
#     --model B6iFast \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --label_len 48 \
#     --pred_len 336 \
#     --e_layers 3 \
#     --d_layers 1 \
#     --des 'normal_0' \
#     --d_model 512 \
#     --seed 2021 \
#     --use_weight 0 \
#     --use_td 1 \
#     --use_extra 1 \
#     --use_pretrain 1 \
#     --epsilon 0.7 \
#     --gamma 0.99 \
#     --tau 0.001 \
#     --exp_name Exchange_96_336

python -u run_open_net.py \
    --is_training 1 \
    --root_path ./dataset/exchange_rate/ \
    --data_path exchange_rate.csv \
    --model_id Exchange_96_336 \
    --model B6iFast \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 336 \
    --e_layers 3 \
    --d_layers 1 \
    --factor 1 \
    --enc_in 8 \
    --dec_in 8 \
    --c_out 8 \
    --des 'normal_1' \
    --n_learner 3 \
    --urt_heads 1 \
    --learning_rate 0.0001 \
    --d_model 512 \
    --d_ff 512 \
    --itr 1 \
    --fix_seed 2021 \
    --train_epochs 20 \
    --batch_size 32 \
    --patience 5 \
    --checkpoints ./checkpoints/

python -u run_rlmc.py \
    --model_id Exchange_96_336 \
    --model B6iFast \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 336 \
    --e_layers 3 \
    --d_layers 1 \
    --des 'normal_1' \
    --d_model 512 \
    --seed 2021 \
    --use_weight 0 \
    --use_td 1 \
    --use_extra 1 \
    --use_pretrain 1 \
    --epsilon 0.6 \
    --gamma 0.99 \
    --tau 0.001 \
    --exp_name Exchange_96_336

# python -u run_open_net.py \
#     --is_training 1 \
#     --root_path ./dataset/exchange_rate/ \
#     --data_path exchange_rate.csv \
#     --model_id Exchange_96_336 \
#     --model B6iFast \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --label_len 48 \
#     --pred_len 336 \
#     --e_layers 3 \
#     --d_layers 1 \
#     --factor 1 \
#     --enc_in 8 \
#     --dec_in 8 \
#     --c_out 8 \
#     --des 'normal_2' \
#     --n_learner 3 \
#     --urt_heads 1 \
#     --learning_rate 0.0001 \
#     --d_model 512 \
#     --d_ff 512 \
#     --itr 1 \
#     --fix_seed 2022 \
#     --train_epochs 20 \
#     --batch_size 32 \
#     --patience 5 \
#     --checkpoints ./checkpoints/

# python -u run_rlmc.py \
#     --model_id Exchange_96_336 \
#     --model B6iFast \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --label_len 48 \
#     --pred_len 336 \
#     --e_layers 3 \
#     --d_layers 1 \
#     --des 'normal_2' \
#     --d_model 512 \
#     --seed 2022 \
#     --use_weight 0 \
#     --use_td 1 \
#     --use_extra 1 \
#     --use_pretrain 1 \
#     --epsilon 0.7 \
#     --gamma 0.99 \
#     --tau 0.001 \
#     --exp_name Exchange_96_336
