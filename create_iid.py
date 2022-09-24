import numpy as np
import os
import torch


class GetWearDataSet(object):
    def __init__(self, action_num, client_num, shard_num, divide_num):
        # action_num:选取的动作数量
        # client_num:客户机数量
        # shard_num:每个客户机分到的数据label数量
        # divide_num:txt文件中125行数据(5s)要分成几份
        self.action_num = action_num
        self.client_num = client_num
        self.shard_num = shard_num
        self.divide_num = divide_num
        self.train_data = None
        self.train_label = None
        self.train_data_size = None
        self.test_data = None
        self.test_label = None
        self.test_data_size = None
        self.train_data_combine, self.valid_data_combine = [0]*5, [0]*5
        self.train_label_combine, self.valid_label_combine = [0]*5, [0]*5

        self.DataSetConstruct()

    def DataSetConstruct(self):
        data_dir = r'../data'
        # action_index = [ i+1 for i in np.random.permutation(19)][:self.action_num]
        action_index = [i + 1 for i in range(self.action_num)]
        file_paths = []
        for file in os.listdir(data_dir):
            if int(file[1:]) in action_index:
                file_paths.append(os.path.join(data_dir, file))
        # print(file_paths)

        # 初始化数据

        all_data = [0] * self.action_num
        label = [0] * self.action_num
        conbine_data = 0
        conbine_label = 0
        for index in range(self.action_num):
            for root, dirs, files in os.walk(file_paths[index]):
                for i, file in enumerate(files):
                    path = os.path.join(root, file)
                    a = np.loadtxt(path, delimiter=',')
                    a = torch.FloatTensor(a)
                    a = a.reshape(self.divide_num, -1, 45)
                    if not torch.is_tensor(all_data[index]):
                        all_data[index] = a
                    else:
                        all_data[index] = torch.cat((all_data[index], a), 0)
            all_data[index] = all_data[index].reshape(100, -1, 125//self.divide_num, 45)
            label[index] = torch.ones(all_data[index].shape)*index
            print(all_data[index].size())
            l = all_data.shape[1]// 6
            train_data, test_data = conbine_data.split([5 * l, 1 * l], dim=1)
            train_label, test_label = conbine_label.split([5 * l, 1 * l], dim=1)
            if not torch.is_tensor(conbine_data):
                self.train_data = train_data
                self.test_data = test_data
                self.train_label = train_label
                self.test_label = test_label
            else:
                self.train_data = torch.cat((self.train_data, train_data), 1)
                self.test_data = torch.cat((self.test_data, test_data), 1)
                self.train_label = torch.cat((self.train_label, train_label), 1)
                self.test_label = torch.cat((self.test_label, test_label), 1)

        # #
        self.train_data_size = self.train_data.shape[1]
        self.test_data_size = self.test_data.shape[1]
        print(self.train_data.shape)
        print(self.train_label.shape)
        print(self.test_data.shape)
        print(self.test_label.shape)


if __name__ == "__main__":
    wearabel_data = GetWearDataSet(15, 100, 3, 5)