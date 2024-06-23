from xmlrpc import client
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from glob import glob
from utils import *
import time
import argparse
import random

# from tensorflow.keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True   #不全部占满显存，按需分配
# set_session(tf.Session(config=config))

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

train_data_path = r"./dataset/non_iid/Train_data"
test_data_path = r"./dataset/non_iid/Test_data"

accpath = r"./result/mnist/non_iid/model/accuracy"
losspath = r"./result/mnist/non_iid/model/loss"

client_model_path = './models(additional_experiment)/mnist/non_iid/client model'
global_model_path = r"./models(additional_experiment)/mnist/non_iid/global model/models"

def load_data(train_data_path, test_data_path):
    x_test_server = np.load(test_data_path + r"/x_test_0.npy")
    y_test_server = np.load(test_data_path + r"/y_test_0.npy")

    x_train = []
    y_train = []
    x = glob(train_data_path + r"/x*.npy")
    y = glob(train_data_path + r"/y*.npy")
    x = sorted(x, key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))
    y = sorted(y, key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))
    for i in x:
        x_train.append(np.load(i))
    for i in y:
        y_train.append(np.load(i))

    x_test_client = np.load(test_data_path + r"/x_test_1.npy")
    y_test_client = np.load(test_data_path + r"/y_test_1.npy")

    return x_train, y_train, x_test_server, y_test_server, x_test_client, y_test_client

def seed_everything(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

if __name__ == "__main__":
    x_train, y_train, x_test_server, y_test_server, x_test_client, y_test_client = load_data(train_data_path, test_data_path)
    clients_num = 10
    args = args_parser()
    if args.dp == 'none':
        accpath = accpath + r"/acc(FL).txt"
        losspath = losspath + r"/loss(FL).txt"
        print(losspath)
        client_model_path = client_model_path + '/FL'
        global_model_path = global_model_path + '/FL'
    elif args.dp == 'ldp':
        accpath = accpath + r"/acc(LDP).txt"
        losspath = losspath + r"/loss(LDP).txt"
        client_model_path = client_model_path + '/LDP'
        global_model_path = global_model_path + '/LDP'
    elif args.dp == 'qpldp':
        accpath = accpath + r"/acc(QPLDP).txt"
        losspath = losspath + r"/loss(QPLDP).txt"
        client_model_path = client_model_path + '/QPLDP'
        global_model_path = global_model_path + '/QPLDP'     
    elif args.dp == 'psi':
        accpath = accpath + r"/acc(PSI).txt"
        losspath = losspath + r"/loss(PSI).txt"
        client_model_path = client_model_path + '/PSI'
        global_model_path = global_model_path + '/PSI'
          
    total_start_time = time.time() # total training start time
        
    for round in range(args.start_epoch, args.end_epoch):
        global_start_time = time.time() # global training start time
        K.clear_session()
        i = 0
        while(clients_num > i):
            print("The " + str(i+1) + " client: ")
            client_model = train_client_model(x_train[i], y_train[i], x_test_client, y_test_client, round, args.local_ep, args.local_bs, args.lr, global_model_path)
            evaluate_client_model(client_model, x_test_client, y_test_client)
            #不添加噪声
            # if args.dp == 'none':
            #向模型参数添加噪声(FL+FDP)
            if args.dp == 'ldp':
                client_model = model_noise_add(client_model, 0.05)
                evaluate_client_model(client_model, x_test_client, y_test_client)
            # 向模型参数添加噪声(FL+Q+PSI+FDP)
            elif args.dp == 'qpldp':
                client_model = model_noise_add(client_model, 0.05)
                evaluate_client_model(client_model, x_test_client, y_test_client)
            elif args.dp == 'psi':
                client_model = model_noise_add(client_model, 0.04)
                evaluate_client_model(client_model, x_test_client, y_test_client)
            save_model_update(client_model, i+1, round, client_model_path)
            i += 1
        
        # psi_set = model_psi(client_model_path, round)
        # models = glob.glob(client_model_path + r"/client_model_" + str(round) + r"_" + r"*.h5")
        # models = sorted(models, key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))
        # num = 0
        # for j in models:
        #     arr = load_model(j)
        #     md1 = fed_learn.model_noise_add(arr, 0.2, psi_set)
        #     fed_learn.evaluate_client_model(md1, x_test_client, y_test_client)
        #     fed_learn.save_model_update(md1, num+1, round, client_model_path)
        #     num += 1
        

        model_aggregation(x_test_server, y_test_server, global_model_path, client_model_path, round, accpath, losspath)

        global_end_time = time.time() # global training end time
        elapsed_global_time = global_end_time - global_start_time
        print(f"Round {round+1:3d} training time: {elapsed_global_time:.2f}  ")
        
    total_end_time = time.time() # total training end time
    elapsed_total_time = total_end_time - total_start_time
    print(f"Total training time: {elapsed_total_time:.2f} seconds")


