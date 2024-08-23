export CUDA_VISIBLE_DEVICES=0

# 36/24
python -u run_open_net.py \
    --is_training 1 \
    --root_path ./dataset/illness/ \
    --data_path national_illness.csv \
    --model_id ILI_36_24 \
    --model B6iFast \
    --data custom \
    --features M \
    --seq_len 36 \
    --label_len 18 \
    --pred_len 24 \
    --e_layers 3 \
    --d_layers 1 \
    --factor 1 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'normal_0' \
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
    --model_id ILI_36_24 \
    --model B6iFast \
    --data custom \
    --features M \
    --seq_len 36 \
    --label_len 18 \
    --pred_len 24 \
    --e_layers 3 \
    --d_layers 1 \
    --des 'normal_0' \
    --d_model 512 \
    --seed 42 \
    --use_weight 0 \
    --use_td 1 \
    --use_extra 1 \
    --use_pretrain 1 \
    --epsilon 1 \
    --gamma 0.99 \
    --tau 0.005 \
    --exp_name ILI_36_24