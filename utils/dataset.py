import numpy as np
from glob import glob

class MNIST():
    def __init__(self, datapath):
        super(MNIST, self).__init__()
        self.datapath = datapath
        x_train = []
        y_train = []
        x = glob(self.datapath + r"/non_iid/Train_data/x*.npy")
        y = glob(self.datapath + r"/non_iid/Train_data/y*.npy")
        x = sorted(x, key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))
        y = sorted(y, key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))
        for i in x:
            x_train.append(np.load(i))
        for i in y:
            y_train.append(np.load(i))

        x_test_server = np.load(self.datapath + r"/non_iid/Test_data/x_test_0.npy")
        y_test_server = np.load(self.datapath + r"/non_iid/Test_data/y_test_0.npy")
        x_test_client = np.load(self.datapath + r"/non_iid/Test_data/x_test_1.npy")
        y_test_client = np.load(self.datapath + r"/non_iid/Test_data/y_test_1.npy")

        return x_train, y_train, x_test_server, y_test_server, x_test_client, y_test_client
    
    
class CIFAR100():
    def __init__(self, datapath, iid_status=False):
        super(CIFAR100, self).__init__()
        self.iid_status = iid_status
        self.datapath = datapath
        
        if self.iid_status:
            x_train = []
            y_train = []
            x = glob(self.datapath + r"/iid/Train_data/x*.npy")
            y = glob(self.datapath + r"/iid/Train_data/y*.npy")
            x = sorted(x, key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))
            y = sorted(y, key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))
            for i in x:
                x_train.append(np.load(i))
            for i in y:
                y_train.append(np.load(i))

            x_test_server = np.load(self.datapath + r"/iid/Test_data/x_test_0.npy")
            y_test_server = np.load(self.datapath + r"/iid/Test_data/y_test_0.npy")
            x_test_client = np.load(self.datapath + r"/iid/Test_data/x_test_1.npy")
            y_test_client = np.load(self.datapath + r"/iid/Test_data/y_test_1.npy")

            return x_train, y_train, x_test_server, y_test_server, x_test_client, y_test_client
        else:
            x_train = []
            y_train = []
            x = glob(self.datapath + r"/non_iid/Train_data/x*.npy")
            y = glob(self.datapath + r"/non_iid/Train_data/y*.npy")
            x = sorted(x, key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))
            y = sorted(y, key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))
            for i in x:
                x_train.append(np.load(i))
            for i in y:
                y_train.append(np.load(i))

            x_test_server = np.load(self.datapath + r"/non_iid/Test_data/x_test_0.npy")
            y_test_server = np.load(self.datapath + r"/non_iid/Test_data/y_test_0.npy")
            x_test_client = np.load(self.datapath + r"/non_iid/Test_data/x_test_1.npy")
            y_test_client = np.load(self.datapath + r"/non_iid/Test_data/y_test_1.npy")

            return x_train, y_train, x_test_server, y_test_server, x_test_client, y_test_client