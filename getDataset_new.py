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
            all_data[index] = all_data[index].reshape(self.client_num * self.shard_num//self.action_num, -1, 125//self.divide_num, 45)
            # print(all_data[index].size())

        conbine_data = 0
        conbine_label = 0
        for i in range(self.action_num // self.shard_num):
            for j in range(all_data[0].shape[0]):
                # 遍历一个动作里待分配的数据切片，每个切片分给一个client，这里做的就是把分给一个client的切片都合并成b，就是一份数据
                for k in range(self.shard_num):
                    # 将分给一个client的shard数据合并出来
                    if k == 0:
                        label = torch.zeros(all_data[0].shape[1], 19).index_fill(1, torch.tensor([i * self.shard_num + k]), 1)
                        # print(label)
                        b = all_data[i * self.shard_num + k][j]
                    else:
                        b = torch.cat((b, all_data[i * self.shard_num + k][j]), 0)
                        # label = torch.cat((label,torch.ones(120,1)*(i*self.shard_num+k)),0)
                        label = torch.cat(
                            (label, torch.zeros(all_data[0].shape[1], 19).index_fill(1, torch.tensor([i * self.shard_num + k]), 1)), 0)
                b = b.squeeze().unsqueeze(0)
                label = label.unsqueeze(0)
                # print(label.size)
                # print(b.size())
                if not torch.is_tensor(conbine_data):
                    conbine_data = b
                    conbine_label = label
                else:
                    conbine_data = torch.cat((conbine_data, b), 0)
                    conbine_label = torch.cat((conbine_label, label), 0)

        # print(conbine_data.size())
        # print(conbine_label.size())
        train1, test1, train2, test2, train3, test3, train4, test4 = conbine_data.split([20*self.divide_num, 4*self.divide_num, 20*self.divide_num, 4*self.divide_num, 20*self.divide_num, 4*self.divide_num, 20*self.divide_num, 4*self.divide_num], dim=1)
        train_l1, test_l1, train_l2, test_l2, train_l3, test_l3, train_l4, test_l4 = conbine_label.split([20*self.divide_num, 4*self.divide_num, 20*self.divide_num, 4*self.divide_num, 20*self.divide_num, 4*self.divide_num, 20*self.divide_num, 4*self.divide_num], dim=1)
        print(np.unique((torch.argmax(train_l1.reshape(-1, 19), dim=1).numpy())))
        print(np.unique((torch.argmax(train_l2.reshape(-1, 19), dim=1).numpy())))
        print(np.unique((torch.argmax(train_l3.reshape(-1, 19), dim=1).numpy())))
        print(np.unique((torch.argmax(train_l4.reshape(-1, 19), dim=1).numpy())))
        self.train_data, self.test_data = torch.cat((train1, train2, train3, train4), 1), torch.cat((test1, test2, test3, test4), 1)
        self.train_label, self.test_label = torch.cat((train_l1, train_l2, train_l3, train_l4), 1), torch.cat((test_l1, test_l2, test_l3, test_l4), 1)
        # self.train_data,self.test_data = conbine_data.split([200,40],dim=1)
        # self.train_label,self.test_label = conbine_label.split([200,40],dim=1)
        self.train_data_size = self.train_data.shape[1]
        self.test_data_size = self.test_data.shape[1]
        # print()

        train1, valid1, train2, valid2, train3, valid3, train4, valid4 = self.train_data.split(
            [16 * self.divide_num, 4 * self.divide_num, 16 * self.divide_num, 4 * self.divide_num, 16 * self.divide_num,
             4 * self.divide_num, 16 * self.divide_num, 4 * self.divide_num], dim=1)
        train_l1, valid_l1, train_l2, valid_l2, train_l3, valid_l3, train_l4, valid_l4 = self.train_label.split(
            [16 * self.divide_num, 4 * self.divide_num, 16 * self.divide_num, 4 * self.divide_num, 16 * self.divide_num,
             4 * self.divide_num, 16 * self.divide_num, 4 * self.divide_num], dim=1)
        self.train_data_combine[0], self.valid_data_combine[0] = torch.cat((train1, train2, train3, train4), 1), torch.cat(
            (valid1, valid2, valid3, valid4), 1)
        self.train_label_combine[0], self.valid_label_combine[0] = torch.cat((train_l1, train_l2, train_l3, train_l4), 1), torch.cat(
            (valid_l1, valid_l2, valid_l3, valid_l4), 1)

        print(np.unique((torch.argmax(valid_l1.reshape(-1, 19), dim=1).numpy())))
        print(np.unique((torch.argmax(valid_l2.reshape(-1, 19), dim=1).numpy())))
        print(np.unique((torch.argmax(valid_l3.reshape(-1, 19), dim=1).numpy())))
        print(np.unique((torch.argmax(valid_l4.reshape(-1, 19), dim=1).numpy())))

        print(self.train_data.shape)
        train1, valid1, train2, valid2, train3, valid3, train4, valid4 = self.train_data.split(
            [4 * self.divide_num, 16 * self.divide_num, 4 * self.divide_num, 16 * self.divide_num, 4 * self.divide_num,
             16 * self.divide_num, 4 * self.divide_num, 16 * self.divide_num], dim=1)
        train_l1, valid_l1, train_l2, valid_l2, train_l3, valid_l3, train_l4, valid_l4 = self.train_label.split(
            [4 * self.divide_num, 16 * self.divide_num, 4 * self.divide_num, 16 * self.divide_num, 4 * self.divide_num,
             16 * self.divide_num, 4 * self.divide_num, 16 * self.divide_num], dim=1)
        self.valid_data_combine[1], self.train_data_combine[1] = torch.cat((train1, train2, train3, train4), 1), torch.cat(
            (valid1, valid2, valid3, valid4), 1)
        self.valid_label_combine[1], self.train_label_combine[1] = torch.cat((train_l1, train_l2, train_l3, train_l4), 1), torch.cat(
            (valid_l1, valid_l2, valid_l3, valid_l4), 1)


        self.train_data_combine[2], self.valid_data_combine[2] = torch.cat((self.train_data[:,:4,:], self.train_data[:,8:24,:], self.train_data[:,28:44,:], self.train_data[:,48:64,:], self.train_data[:,68:,:]), 1), torch.cat(
            (self.train_data[:,4:8,:], self.train_data[:,24:28,:], self.train_data[:,44:48,:], self.train_data[:,64:68,:]), 1)
        self.train_label_combine[2], self.valid_label_combine[2] = torch.cat((self.train_label[:,:4,:], self.train_label[:,8:24,:], self.train_label[:,28:44,:], self.train_label[:,48:64,:], self.train_label[:,68:,:]), 1), torch.cat(
            (self.train_label[:,4:8,:], self.train_label[:,24:28,:], self.train_label[:,44:48,:], self.train_label[:,64:68,:]), 1)


        self.train_data_combine[3], self.valid_data_combine[3] = torch.cat((self.train_data[:,:8,:], self.train_data[:,12:28,:], self.train_data[:,32:48,:], self.train_data[:,52:68,:], self.train_data[:,72:,:]), 1), torch.cat(
            (self.train_data[:,8:12,:], self.train_data[:,28:32,:], self.train_data[:,48:52,:], self.train_data[:,68:72,:]), 1)
        self.train_label_combine[3], self.valid_label_combine[3] = torch.cat((self.train_label[:,:8,:], self.train_label[:,12:28,:], self.train_label[:,32:48,:], self.train_label[:,52:68,:], self.train_label[:,72:,:]), 1), torch.cat(
            (self.train_label[:,8:12,:], self.train_label[:,28:32,:], self.train_label[:,48:52,:], self.train_label[:,68:72,:]), 1)


        self.train_data_combine[4], self.valid_data_combine[4] = torch.cat((self.train_data[:,:12,:], self.train_data[:,16:32,:], self.train_data[:,36:52,:], self.train_data[:,56:72,:], self.train_data[:,76:,:]), 1), torch.cat(
            (self.train_data[:,12:16,:], self.train_data[:,32:36,:], self.train_data[:,52:56,:], self.train_data[:,72:76,:]), 1)
        self.train_label_combine[4], self.valid_label_combine[4] = torch.cat((self.train_label[:,:12,:], self.train_label[:,16:32,:], self.train_label[:,36:52,:], self.train_label[:,56:72,:], self.train_label[:,76:,:]), 1), torch.cat(
            (self.train_label[:,12:16,:], self.train_label[:,32:36,:], self.train_label[:,52:56,:], self.train_label[:,72:76,:]), 1)

        a = 4*self.divide_num
        b = 20*self.divide_num
        for i in range(5):
            self.train_data_combine[i], self.valid_data_combine[i] = torch.cat((self.train_data[:,:a*i,:], self.train_data[:,a*(i+1):b+a*i,:], self.train_data[:,b+a*(i+1):2*b+a*i,:], self.train_data[:,2*b+a*(i+1):3*b+a*i,:], self.train_data[:,3*b+a*(i+1):,:]), 1), \
                                                                     torch.cat((self.train_data[:,a*i:a*(i+1),:], self.train_data[:,b+a*i:b+a*(i+1),:], self.train_data[:,2*b+a*i:2*b+a*(i+1),:], self.train_data[:,3*b+a*i:3*b+a*(i+1),:]), 1)
            self.train_label_combine[i], self.valid_label_combine[i] = torch.cat((self.train_label[:,:a*i,:], self.train_label[:,a*(i+1):b+a*i,:], self.train_label[:,b+a*(i+1):2*b+a*i,:], self.train_label[:,2*b+a*(i+1):3*b+a*i,:], self.train_label[:,3*b+a*(i+1):,:]), 1), \
                                                                     torch.cat((self.train_label[:,a*i:a*(i+1),:], self.train_label[:,b+a*i:b+a*(i+1),:], self.train_label[:,2*b+a*i:2*b+a*(i+1),:], self.train_label[:,3*b+a*i:3*b+a*(i+1),:]), 1)


if __name__ == "__main__":
    wearabel_data = GetWearDataSet(16, 80, 4, 125)
    print(wearabel_data.train_data.size())
    # print(wearabel_data.train_data.reshape(5,20,-1,25, 45)[0,:,:,:,18:27])
    print(wearabel_data.train_label.size())
    print(wearabel_data.test_label.size())
    print(wearabel_data.test_label[0][0])
    print(wearabel_data.test_label[0][-1])
    print(wearabel_data.test_label[-1][0])
    print(wearabel_data.test_label[-1][-1])
