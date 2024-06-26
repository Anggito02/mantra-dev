export CUDA_VISIBLE_DEVICES=0

# 96/96
python -u run_open_net.py \
    --is_training 1 \
    --root_path ./dataset/ETT/ \
    --data_path ETTm2.csv \
    --model_id ETTh1_96_96 \
    --model B6iFast \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 1 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'normal_0' \
    --n_learner 3 \
    --urt_heads 1 \
    --learning_rate 0.0001 \
    --d_model 128 \
    --d_ff 128 \
    --itr 1 \
    --fix_seed 2021 \
    --train_epochs 20 \
    --batch_size 32 \
    --patience 5 \
    --checkpoints ./checkpoints/

python -u run_rlmc.py \
    --model_id ETTh1_96_96 \
    --model B6iFast \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --e_layers 2 \
    --d_layers 1 \
    --des 'normal_0' \
    --d_model 128 \
    --use_weight 0 \
    --use_td 1 \
    --use_extra 1 \
    --use_pretrain 1 \
    --epsilon 0.7 \
    --gamma 0.99 \
    --tau 0.005 \
    --exp_name ILI_36_24