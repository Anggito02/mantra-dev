import argparse
import os
import torch
from exp.exp_main_3k_updated import Exp_Main_DualmodE3K
# from exp.opt_urt import Opt_URT

from exp.exp_rl import OPT_RL_Mantra
import random
import numpy as np

import gc

def main():
    # fix_seed = 2021
    # random.seed(fix_seed)
    # torch.manual_seed(fix_seed)
    # np.random.seed(fix_seed)
    # torch.cuda.manual_seed(fix_seed)
    # torch.backends.cudnn.deterministic = True
    # os.environ['PYTHONHASHSEED'] = str(fix_seed)
    
    parser = argparse.ArgumentParser(description='iTransformer for Time Series Forecasting')

    # basic config
    parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=False, default='tes', help='model id')
    parser.add_argument('--model', type=str, required=False, default='B6iFast',
                        help='model name, options: [Autoformer, Informer, Transformer]')
    parser.add_argument('--slow_model', type=str, required=False, default='S1iSlow',
                        help='slow model name, options: [Autoformer, Informer, Transformer, etc]')

    # data loader
    parser.add_argument('--data', type=str, required=False, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/illness', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='national_illness.csv', help='data file')
    parser.add_argument('--features', type=str, default='S',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=36, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=18, help='start token length')
    parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')

    # model define
    parser.add_argument('--bucket_size', type=int, default=4, help='for Reformer')
    parser.add_argument('--n_hashes', type=int, default=4, help='for Reformer')
    parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_learner', type=int, default=3, help='number of learner')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--urt_heads', type=int, default=1, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--anomaly', type=float, default=10.0, help='anomaly limit')
    parser.add_argument('--des', type=str, default='Test', help='exp description')
    parser.add_argument('--loss', type=str, default='neg_corr', help='loss function')
    parser.add_argument('--corr_penalty', type=float, default=0.5, help='correlation penalty for negative correlation loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')
    parser.add_argument('--fix_seed', type=str, default='2021,2022,2023', help='Fix seed for iterations')

    # RL
    parser.add_argument('--rl_seed', default=42, type=int)
    parser.add_argument('--use_weight',   default=1, type=int)
    parser.add_argument('--use_td',       default=1, type=int)
    parser.add_argument('--use_extra',    default=1, type=int)
    parser.add_argument('--use_pretrain', default=1, type=int)

    parser.add_argument('--epsilon', default=0.5, type=float)
    parser.add_argument('--exp_name', default='rlmc', type=str)

    parser.add_argument('--learn_rate_RL', default=3e-4, type=float, help='learning rate for reinforcement learning')
    parser.add_argument('--RL_epochs', default=1, type=int, help='epoch for reinforcement learning')
    parser.add_argument('--RL_warmup_epoch', default=100, type=int, help='warmup epoch for reinforcement learning')
    parser.add_argument('--RL_pretrain_epoch', default=200, type=int, help='pretrain epoch for reinforcement learning')
    parser.add_argument('--RL_step_size', default=450, type=int, help='step size for reinforcement learning')
    parser.add_argument('--RL_max_patience', default=5, type=int, help='max patience for reinforcement learning')

    parser.add_argument('--gamma', default=0.99, type=float, help='discount factor')
    parser.add_argument('--tau', default=0.005, type=float, help='soft update rate')
    parser.add_argument('--hidden_dim', default=256, type=int, help='hidden dimension for RL CNN')

    # parser.add_argument('--num_fastlearners', type=int, default=2, help='number of fast_learner')


    args = parser.parse_args()

    # fix_seed = 2021
    # fix_seed=args.fix_seed.split(",")
    fix_seed=int(args.fix_seed.split(",")[0])
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    torch.cuda.manual_seed(fix_seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(fix_seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
        # args.gpu = args.device_ids
        # args.gpu = args.devices
        print("args.gpu: ")
        print(args.gpu)

    print('Args in experiment:')
    print(args)

    
    # Exp = Exp_Main_Dualmod
    Exp = Exp_Main_DualmodE3K

    if args.is_training:
        for ii in range(args.itr):
            
            fix_seed=args.fix_seed.split(",")
            fix_seed=[int(i) for i in fix_seed]
            random.seed(fix_seed[ii])
            torch.manual_seed(fix_seed[ii])
            np.random.seed(fix_seed[ii])
            torch.cuda.manual_seed(fix_seed[ii])
            torch.backends.cudnn.deterministic = True
            os.environ['PYTHONHASHSEED'] = str(fix_seed[ii])

            # setting record of experiments
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            exp = Exp(args)  # set experiments
            # opt = OptURT(args)  # set experiments

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            # print('>>>>>>>start training URT: {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            # opt.train_urt(setting)

            # Testing only Mantra
            print('>>>>>>>testing only mantra : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            # print('>>>>>>>testing Model+URT : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            # opt.test2(setting)

            # if args.do_predict:
            #     print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            #     exp.predict(setting, True)

            # RL Experiment
            rl_exp = OPT_RL_Mantra(args, setting)
            print(">>>>>>>RL_Train: {}>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting))
            rl_exp.forward(setting)

            del exp
            del rl_exp
            gc.collect()
            
            torch.cuda.empty_cache()


        # OptRL = OPT_RL_Mantra(args)
        # for ii in range(args.itr):
            
        #     setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
        #         args.model_id,
        #         args.model,
        #         args.data,
        #         args.features,
        #         args.seq_len,
        #         args.label_len,
        #         args.pred_len,
        #         args.d_model,
        #         args.n_heads,
        #         args.e_layers,
        #         args.d_layers,
        #         args.d_ff,
        #         args.factor,
        #         args.embed,
        #         args.distil,
        #         args.des, ii)

        #     # exp = Exp(args)  # set experiments
        #     # opt = OptURT(args)  # set experiments

        #     # print('>>>>>>>start training URT: {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        #     # opt.train_urt(setting)

        #     # print('>>>>>>>testing FastSlow+URT : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        #     # opt.test2(setting)

        #     torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                      args.model,
                                                                                                      args.data,
                                                                                                      args.features,
                                                                                                      args.seq_len,
                                                                                                      args.label_len,
                                                                                                      args.pred_len,
                                                                                                      args.d_model,
                                                                                                      args.n_heads,
                                                                                                      args.e_layers,
                                                                                                      args.d_layers,
                                                                                                      args.d_ff,
                                                                                                      args.factor,
                                                                                                      args.embed,
                                                                                                      args.distil,
                                                                                                      args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
