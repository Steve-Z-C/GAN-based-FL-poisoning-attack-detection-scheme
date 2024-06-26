import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
# from create_mnist_iid_datasets import *

def mnist_noniid():
    # 加载MNIST数据集
    (x_train, y_train), _ = mnist.load_data()
    print(x_train.shape, y_train.shape)
    # 按类别排序训练数据
    sorted_indices = np.argsort(y_train)
    x_train_sorted = x_train[sorted_indices]
    y_train_sorted = y_train[sorted_indices]
    print(x_train_sorted.shape, y_train_sorted.shape)

    # 划分为200组，每组300条数据
    num_groups = 200
    group_size = 300

    x_train_groups = np.array_split(x_train_sorted, num_groups)
    y_train_groups = np.array_split(y_train_sorted, num_groups)

    # 设置10个客户端
    num_clients = 10
    groups_per_client = 20

    # 每个客户端按顺序分配20组数据
    client_data = []
    for i in range(num_clients):
        start_idx = i * groups_per_client
        end_idx = start_idx + groups_per_client
        client_x = np.concatenate(x_train_groups[start_idx:end_idx])
        client_y = np.concatenate(y_train_groups[start_idx:end_idx])
        # client_x, client_y = data_processing(client_x, client_y)
        print(client_x.shape, client_y.shape)
        client_data.append((client_x, client_y))

    # 保存数据为npy格式
    for i, (client_x, client_y) in enumerate(client_data):
        np.save(f'../mnist/non_iid/Train_data/x_train_{i+1}.npy', client_x)
        np.save(f'../mnist/non_iid/Train_data/y_train_{i+1}.npy', client_y)
        
if __name__ == "__main__":
    mnist_noniid()
