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
                        label = torch.zeros(all_data[0].shape[1], self.action_num).index_fill(1, torch.tensor(
                            [i * self.shard_num + k]), 1)
                        # print(label)
                        b = all_data[i * self.shard_num + k][j]
                    else:
                        b = torch.cat((b, all_data[i * self.shard_num + k][j]), 0)
                        # label = torch.cat((label,torch.ones(120,1)*(i*self.shard_num+k)),0)
                        label = torch.cat(
                            (label, torch.zeros(all_data[0].shape[1], self.action_num).index_fill(1, torch.tensor(
                                [i * self.shard_num + k]), 1)), 0)
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
        train1, test1, train2, test2, train3, test3 = conbine_data.split([20*self.divide_num, 4*self.divide_num, 20*self.divide_num, 4*self.divide_num, 20*self.divide_num, 4*self.divide_num], dim=1)
        train_l1, test_l1, train_l2, test_l2, train_l3, test_l3 = conbine_label.split([20*self.divide_num, 4*self.divide_num, 20*self.divide_num, 4*self.divide_num, 20*self.divide_num, 4*self.divide_num], dim=1)
        # print(torch.argmax(train_l1.reshape(-1, self.action_num), dim=1))
        # print(np.unique((torch.argmax(train_l1.reshape(-1, self.action_num), dim=1).numpy())))
        # print(np.unique((torch.argmax(train_l2.reshape(-1, self.action_num), dim=1).numpy())))
        # print(np.unique((torch.argmax(train_l3.reshape(-1, self.action_num), dim=1).numpy())))
        self.train_data, self.test_data = torch.cat((train1, train2, train3), 1), torch.cat((test1, test2, test3), 1)
        self.train_label, self.test_label = torch.cat((train_l1, train_l2, train_l3), 1), torch.cat((test_l1, test_l2, test_l3), 1)
        # conbine_label = torch.argmax(conbine_label, dim=2)

        # train1, test1, train2, test2 = conbine_data.split([20*self.divide_num, 4*self.divide_num, 20*self.divide_num, 4*self.divide_num], dim=1)
        # train_l1, test_l1, train_l2, test_l2 = conbine_label.split([20*self.divide_num, 4*self.divide_num, 20*self.divide_num, 4*self.divide_num], dim=1)
        # self.train_data, self.test_data = torch.cat((train1, train2), 1), torch.cat((test1, test2), 1)
        # self.train_label, self.test_label = torch.cat((train_l1, train_l2), 1), torch.cat((test_l1, test_l2), 1)
        # #
        self.train_data_size = self.train_data.shape[1]
        self.test_data_size = self.test_data.shape[1]

        # train1, valid1, train2, valid2 = self.train_data.split(
        #     [16 * self.divide_num, 4 * self.divide_num, 16 * self.divide_num, 4 * self.divide_num], dim=1)
        # train_l1, valid_l1, train_l2, valid_l2 = self.train_label.split(
        #     [16 * self.divide_num, 4 * self.divide_num, 16 * self.divide_num, 4 * self.divide_num], dim=1)
        # self.train_data_combine, self.valid_data_combine = torch.cat((train1, train2), 1), torch.cat((valid1, valid2), 1)
        # self.train_label_combine, self.valid_label_combine = torch.cat((train_l1, train_l2), 1), torch.cat((valid_l1, valid_l2), 1)

        # train1, valid1, train2, valid2, train3, valid3 = self.train_data.split(
        #     [16 * self.divide_num, 4 * self.divide_num, 16 * self.divide_num, 4 * self.divide_num, 16 * self.divide_num, 4 * self.divide_num], dim=1)
        # train_l1, valid_l1, train_l2, valid_l2, train_l3, valid_l3 = self.train_label.split(
        #     [16 * self.divide_num, 4 * self.divide_num, 16 * self.divide_num, 4 * self.divide_num, 16 * self.divide_num, 4 * self.divide_num], dim=1)


        # for i in range(5):
        #     self.train_data_combine[i], self.valid_data_combine[i] = torch.cat((self.train_data[:,:20*i,:], self.train_data[:,20*(i+1):100+20*i,:], self.train_data[:,100+20*(i+1):200+20*i,:], self.train_data[:,200+20*(i+1):,:]), 1), \
        #                                                              torch.cat((self.train_data[:,20*i:20*(i+1),:], self.train_data[:,100+20*i:100+20*(i+1),:], self.train_data[:,200+20*i:200+20*(i+1),:]), 1)
        #     self.train_label_combine[i], self.valid_label_combine[i] = torch.cat((self.train_label[:,:20*i,:], self.train_label[:,20*(i+1):100+20*i,:], self.train_label[:,100+20*(i+1):200+20*i,:], self.train_label[:,200+20*(i+1):,:]), 1), \
        #                                                              torch.cat((self.train_label[:,20*i:20*(i+1),:], self.train_label[:,100+20*i:100+20*(i+1),:], self.train_label[:,200+20*i:200+20*(i+1),:]), 1)
        #
        a = 4 * self.divide_num
        b = 20 * self.divide_num
        # for i in range(5):
        #     self.train_data_combine[i], self.valid_data_combine[i] = torch.cat((self.train_data[:, :a * i, :],
        #                                                                         self.train_data[:,
        #                                                                         a * (i + 1):b + a * i, :],
        #                                                                         self.train_data[:,
        #                                                                         b + a * (i + 1):2 * b + a * i, :],
        #                                                                         self.train_data[:,
        #                                                                         2 * b + a * (i + 1):, :]
        #                                                                         ), 1), \
        #                                                              torch.cat((
        #                                                                        self.train_data[:, a * i:a * (i + 1), :],
        #                                                                        self.train_data[:,
        #                                                                        b + a * i:b + a * (i + 1), :],
        #                                                                        self.train_data[:,
        #                                                                        2 * b + a * i:2 * b + a * (i + 1), :]
        #                                                                         ),
        #                                                                        1)
        #     self.train_label_combine[i], self.valid_label_combine[i] = torch.cat((self.train_label[:, :a * i, :],
        #                                                                           self.train_label[:,
        #                                                                           a * (i + 1):b + a * i, :],
        #                                                                           self.train_label[:,
        #                                                                           b + a * (i + 1):2 * b + a * i, :],
        #                                                                           self.train_label[:,
        #                                                                           2 * b + a * (i + 1):, :]
        #                                                                           ), 1), \
        #                                                                torch.cat((self.train_label[:, a * i:a * (i + 1),
        #                                                                           :], self.train_label[:,
        #                                                                               b + a * i:b + a * (i + 1), :],
        #                                                                           self.train_label[:,
        #                                                                           2 * b + a * i:2 * b + a * (i + 1), :]
        #                                                                           ), 1)


        # for i in range(5):
        #     self.train_data_combine[i], self.valid_data_combine[i] = torch.cat((self.train_data[:, :a * i, :],
        #                                                                         self.train_data[:,
        #                                                                         a * (i + 1):b + a * i, :],
        #                                                                         self.train_data[:,
        #                                                                         b + a * (i + 1):, :],
        #
        #                                                                         ), 1), \
        #                                                              torch.cat((
        #                                                                        self.train_data[:, a * i:a * (i + 1), :],
        #                                                                        self.train_data[:,
        #                                                                        b + a * i:b + a * (i + 1), :]
        #
        #                                                                         ),
        #                                                                        1)
        #     self.train_label_combine[i], self.valid_label_combine[i] = torch.cat((self.train_label[:, :a * i, :],
        #                                                                           self.train_label[:,
        #                                                                           a * (i + 1):b + a * i, :],
        #                                                                           self.train_label[:,
        #                                                                           b + a * (i + 1):, :],
        #
        #                                                                           ), 1), \
        #                                                                torch.cat((self.train_label[:, a * i:a * (i + 1),
        #                                                                           :], self.train_label[:,
        #                                                                               b + a * i:b + a * (i + 1), :],
        #
        #                                                                           ), 1)
        # print()


if __name__ == "__main__":
    wearabel_data = GetWearDataSet(3, 20, 3, 5)
    print(wearabel_data.train_data.size())
    # print(wearabel_data.train_data.reshape(5,20,-1,25, 45)[0,:,:,:,18:27])
    print(wearabel_data.train_label.size())
    print(wearabel_data.test_label.size())
    print(wearabel_data.test_label[0][0])
    print(wearabel_data.test_label[0][-1])
    print(wearabel_data.test_label[-1][0])
    print(wearabel_data.test_label[-1][-1])