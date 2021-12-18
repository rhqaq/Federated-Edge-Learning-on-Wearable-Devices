import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from getDataset import GetWearDataSet


class client(object):
    def __init__(self, trainDataSet,train_label_set, dev):
        self.train_ds = trainDataSet
        self.train_label_set = train_label_set
        self.dev = dev
        self.train_dl = None
        self.local_parameters = None

    def localUpdate(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters,alpha):
        Net.load_state_dict(global_parameters, strict=True)
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)
        for epoch in range(localEpoch):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = Net(data)
                # 修改为FedRS
                for i in range(preds.shape[1]):
                    if i not in self.train_label_set:
                        preds[:,i] = preds[:,i]* alpha
                loss = lossFun(preds, label)
                loss.backward()
                opti.step()
                opti.zero_grad()

        return Net.state_dict()

    def local_val(self):
        pass


class ClientsGroup(object):
    def __init__(self, action_num,client_num, shard_num, divide_num, dev):
        self.action_num = action_num
        self.client_num = client_num
        self.shard_num = shard_num
        self.divide_num = divide_num
        self.dev = dev
        self.clients_set = {}

        self.test_data_loader = None

        self.dataSetBalanceAllocation()

    def dataSetBalanceAllocation(self):
        mnistDataSet = GetWearDataSet(self.action_num,self.client_num, self.shard_num,self. divide_num)

        # test_data = mnistDataSet.test_data.reshape(-1,25,45)[:,:,18:27]
        test_data = mnistDataSet.test_data.reshape(-1,45)[:,18:27] # for MLP
        # print(test_data.size())
        test_label = torch.argmax(mnistDataSet.test_label.reshape(-1,15), dim=1)
        # print(test_label.size())
        self.test_data_loader = DataLoader(TensorDataset( test_data, test_label), batch_size=100, shuffle=False)

        train_data = mnistDataSet.train_data
        train_label = mnistDataSet.train_label

        # shard_size = mnistDataSet.train_data_size // self.num_of_clients // 2
        # shards_id = np.random.permutation(mnistDataSet.train_data_size // shard_size)
        for i in range(self.client_num):
            # shards_id1 = shards_id[i * 2]
            # shards_id2 = shards_id[i * 2 + 1]
            # data_shards1 = train_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            # data_shards2 = train_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            # label_shards1 = train_label[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            # label_shards2 = train_label[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]

            # local_data, local_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
            local_data, local_label = train_data[i],train_label[i]
            local_data = local_data[:,18:27]
            # local_data = local_data[:,:, 18:27]
            # print(local_data.size())
            # print(local_label.size())
            local_label = np.argmax(local_label, axis=1)
            local_label_set = local_label
            local_label_set = np.unique((local_label_set.numpy()))
            # print(local_label.size())
            someone = client(TensorDataset(local_data, local_label), local_label_set,self.dev)
            self.clients_set['client{}'.format(i)] = someone

if __name__=="__main__":
    MyClients = ClientsGroup( 15,100, 2, 5, 1)
    print(MyClients.clients_set['client21'].train_ds[0:200])
    print(len(MyClients.clients_set['client11'].train_ds))