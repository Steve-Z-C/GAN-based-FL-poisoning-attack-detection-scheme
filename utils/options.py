#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse




def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    # parser.add_argument('--epochs', type=int, default=5, help="rounds of training")
    parser.add_argument('--start_epoch', type=int, default=1, help='Start epoch for training')
    parser.add_argument('--end_epoch', type=int, default=101, help='End epoch for training')
    parser.add_argument('--num_users', type=int, default=10, help="number of users: K")
    # parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=20, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=128, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    # parser.add_argument('--lr_decay', type=float, default=0.995, help="learning rate decay each round")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum (default: 0.5)")
    # parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--aggregation', type=str, default='fedavg', help="name of aggregation rule")
    parser.add_argument('--dp', type=str, default='none', help="differential privacy method")
    parser.add_argument('--attack', type=str, default='none', help="attack method")
    parser.add_argument('--detect', type=str, default='none', help="aggregation rule")


    args = parser.parse_args()
    return args
