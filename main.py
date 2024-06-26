import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.compat.v1 import ConfigProto
from tensorflow.keras import backend as K
import numpy as np
from glob import glob
import random
from xmlrpc import client
from utils import *
import time
import argparse

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

train_data_path = "./dataset/mnist/Train_data"
target_data_path = "./dataset/mnist/Target_data"
test_data_path = "./dataset/mnist/Test_data"

accpath = "./result/mnist/accuracy"
losspath = "./result/mnist/loss"

client_model_path = "./models/mnist/client model"
global_model_path = "./models/mnist/global model"

def load_data(train_data_path, test_data_path):
    x_test_server = np.load(test_data_path + r"/x_test_0.npy")
    y_test_server = np.load(test_data_path + r"/y_test_0.npy")

    x_train = []
    y_train = []
    x = glob(train_data_path + "/x*.npy")
    y = glob(train_data_path + "/y*.npy")
    x = sorted(x, key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))
    y = sorted(y, key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))
    for i in x:
        x_train.append(np.load(i))
    for i in y:
        y_train.append(np.load(i))

    x_test_client = np.load(test_data_path + "/x_test_1.npy")
    y_test_client = np.load(test_data_path + "/y_test_1.npy")

    return x_train, y_train, x_test_server, y_test_server, x_test_client, y_test_client

def seed_everything(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    
if __name__ == "__main__":
    clients_num = 10
    args = args_parser()
    if args.attack == 'target':
        x_target, y_target, x_test_server, y_test_server, x_test_client, y_test_client = load_data(target_data_path, test_data_path)
    else:
        x_train, y_train, x_test_server, y_test_server, x_test_client, y_test_client = load_data(train_data_path, test_data_path)
    if args.attack == 'none':
        accpath = accpath + r"/acc(normal).txt"
        losspath = losspath + r"/loss(normal).txt"
        client_model_path = client_model_path + '/normal'
        global_model_path = global_model_path + '/normal'
    elif args.attack == 'random':
        accpath = accpath + r"/acc(random).txt"
        losspath = losspath + r"/loss(random).txt"
        client_model_path = client_model_path + '/random'
        global_model_path = global_model_path + '/random'
    elif args.attack == 'target':
        accpath = accpath + r"/acc(target).txt"
        losspath = losspath + r"/loss(target).txt"
        client_model_path = client_model_path + '/target'
        global_model_path = global_model_path + '/target'
    elif args.attack == 'detect':
        accpath = accpath + r"/acc(detect).txt"
        losspath = losspath + r"/loss(detect).txt"
        client_model_path = client_model_path + '/detect'
        global_model_path = global_model_path + '/detect'     
          
    f1_loss = 0.05
    acc_loss = 0.05
    safety_matrix = np.zeros(clients_num)
    safety_param = 5
    total_start_time = time.time() # total training start time
        
    for round in range(args.start_epoch, args.end_epoch):
        global_start_time = time.time() # global training start time
        K.clear_session()
        i = 0
        while(clients_num > i):
            print("The " + str(i+1) + " client: ")
            # normal fl
            if args.attack == 'none':
                client_model = train_client_model(x_train[i], y_train[i], x_test_client, y_test_client, round, args.local_ep, args.local_bs, args.lr, global_model_path)
                evaluate_client_model(client_model, x_test_client, y_test_client)
            # random attack
            if args.attack == 'random':
                client_model = model_random_attack(x_train[i], y_train[i], x_test_client, y_test_client, round, args.local_ep, args.local_bs, args.lr, global_model_path)
                evaluate_client_model(client_model, x_test_client, y_test_client)
            # target attack
            elif args.attack == 'target':
                client_model = train_client_model(x_target[i], y_target[i], x_test_client, y_test_client, round, args.local_ep, args.local_bs, args.lr, global_model_path)
                evaluate_client_model(client_model, x_test_client, y_test_client)
            save_model_update(client_model, i+1, round, client_model_path)
            i += 1

        
        if args.detect == 'none':
            model_aggregation(x_test_server, y_test_server, global_model_path, client_model_path, round, accpath, losspath)
        elif args.detect == 'gan':
            if round == 1:
                model_aggregation(x_test_server, y_test_server, global_model_path, client_model_path, round, accpath, losspath)
            else:
                safety_matrix = model_detect_aggregation(x_test_server, y_test_server, global_model_path, client_model_path, round, accpath, losspath, f1_loss, acc_loss, safety_matrix, safety_param)

        global_end_time = time.time() # global training end time
        elapsed_global_time = global_end_time - global_start_time
        print(f"Round {round+1:3d} training time: {elapsed_global_time:.2f}  ")
        
    total_end_time = time.time() # total training end time
    elapsed_total_time = total_end_time - total_start_time
    print(f"Total training time: {elapsed_total_time:.2f} seconds")


