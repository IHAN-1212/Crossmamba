import argparse
import os
import torch
import random
import numpy as np

from cross_exp.exp_crossmamba import Exp_crossmamba
from utils.tools import string_split

parser = argparse.ArgumentParser(description='CrossMamba')

parser.add_argument('--data', type=str, required=True, default='ETTh2', help='data')
parser.add_argument('--root_path', type=str, default='./datasets/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh2.csv', help='data file')  
parser.add_argument('--data_split', type=str, default='0.7,0.1,0.2',help='train/val/test split, can be ratio or number')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location to store model checkpoints')

parser.add_argument('--in_len', type=int, default=96, help='input MTS length (T)')
parser.add_argument('--out_len', type=int, default=96, help='output MTS length (\tau)')
parser.add_argument('--data_dim', type=int, default=7, help='Number of dimensions of the MTS data (D)')

parser.add_argument('--t_cycle', type=int, default=6, help='segment length (t_cycle)')
parser.add_argument('--d_model', type=int, default=128, help='dimension of hidden states (d_model)')
parser.add_argument('--d_ff', type=int, default=128, help='dimension of MLP in transformer')
parser.add_argument('--d_state', type=int, default=1, help='num of heads')

parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer initial learning rate')
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
parser.add_argument('--itr', type=int, default=1, help='experiments times')

parser.add_argument('--save_pred', action='store_true', help='whether to save the predicted future MTS', default=False)

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multiple gpus')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]
    print(args.gpu)

data_parser = {
    'ETTh1':{'data':'ETTh1.csv', 'data_dim':7, 'split':[0.6, 0.2, 0.2]},
    'ETTh2':{'data':'ETTh2.csv', 'data_dim':7, 'split':[0.6, 0.2, 0.2]},
    'ETTm1':{'data':'ETTm1.csv', 'data_dim':7, 'split':[0.6, 0.2, 0.2]},
    'ETTm2':{'data':'ETTm2.csv', 'data_dim':7, 'split':[0.6, 0.2, 0.2]},
    'ECL':{'data':'ECL.csv', 'data_dim':321, 'split':[0.7, 0.1, 0.2]},
    'WTH':{'data':'WTH.csv', 'data_dim':12, 'split':[0.6, 0.2, 0.2]},
    'Weather':{'data':'weather.csv', 'data_dim':21, 'split':[0.7, 0.1, 0.2]},
    'ILI':{'data':'national_illness.csv', 'data_dim':7, 'split':[0.7, 0.1, 0.2]},
    'Traffic':{'data':'traffic.csv', 'data_dim':862, 'split':[0.7, 0.1, 0.2]},
    'Exchange':{'data':'exchange_rate.csv', 'data_dim':8, 'split':[0.7, 0.1, 0.2]},
}
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.data_dim = data_info['data_dim']
    args.data_split = data_info['split']
else:
    args.data_split = string_split(args.data_split)

print('Args in experiment:')
print(args)

Exp = Exp_crossmamba

for ii in range(args.itr):
    # setting record of experiments
    setting = ('Crossmamba_{}{}__in{}_seg{}__dmodel-{}_dstate-{}_dff-{}_dropout{}_batch{}___lr{}_itr{}'.
                format(args.data,args.out_len,
                args.in_len, args.t_cycle,
                args.d_model, args.d_state, args.d_ff, args.dropout, args.batch_size, args.lradj, ii))

    exp = Exp(args) # set experiments
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)
    
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, args.save_pred)
