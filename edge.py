import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from getDataset import GetWearDataSet
from Models import LSTM,MLP,Simple_LSTM
import torch.nn.functional as F
import os
from torch import optim
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix
from tqdm import tqdm


def localUpdate(localEpoch, train_dl, Net, lossFun, opti, global_parameters, dev):
    Net.load_state_dict(global_parameters, strict=True)
    # train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)
    for epoch in range(localEpoch):
        for data, label in train_dl:
            data, label = data.to(dev), label.to(dev)
            preds = Net(data)
            # 修改为FedRS
            # for i in range(preds.shape[1]):
            #     if i not in train_label_set:
            #         preds[:, i] = preds[:, i]
            loss = lossFun(preds, label)
            loss.backward()
            opti.step()
            opti.zero_grad()

    return Net.state_dict()


if __name__=="__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print(torch.cuda.device_count())
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # net = MLP()
    net = LSTM()
    net = net.to(dev)
    opti = optim.SGD(net.parameters(), lr=0.05)

    loss_func = F.cross_entropy

    global_parameters = {}
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()
    wearDataSet = GetWearDataSet(15, 100, 3, 5)
    # wearDataSet = GetWearDataSet(15, 100, 3, 125)
    test_data = wearDataSet.test_data.reshape(-1, 25, 45)[:,:,18:27]

    # test_data = wearDataSet.test_data.reshape(-1, 45)[:,18:27]
    # print(test_data.size())
    test_label = torch.argmax(wearDataSet.test_label.reshape(-1, 15), dim=1)
    test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=1000, shuffle=False)

    # train_data = wearDataSet.train_data.reshape(-1, 25, 45)[:,:,18:27]
    train_data_all = torch.chunk(input=wearDataSet.train_data,
                chunks=5,
                dim=0)
    train_label_all = torch.chunk(input=wearDataSet.train_label,
                chunks=5,
                dim=0)

    train_data_loader = list(range(5))
    # train_data = train_data_all[0].reshape(-1, 45)[:,18:27]
    train_data = train_data_all[0].reshape(-1, 25, 45)[:,:,18:27]
    print(train_data.size())
    train_label = torch.argmax(train_label_all[0].reshape(-1, 15), dim=1)
    print(train_label.size())
    train_data_loader[0] = DataLoader(TensorDataset(train_data,train_label), batch_size=100, shuffle=True)

    # train_data = train_data_all[0].reshape(-1, 45)[:,18:27]
    train_data = train_data_all[0].reshape(-1, 25, 45)[:, :, 18:27]
    print(train_data.size())
    train_label = torch.argmax(train_label_all[1].reshape(-1, 15), dim=1)
    train_data_loader[1] = DataLoader(TensorDataset(train_data, train_label), batch_size=100, shuffle=True)

    # train_data = train_data_all[2].reshape(-1, 45)[:,18:27]
    train_data = train_data_all[2].reshape(-1, 25, 45)[:, :, 18:27]
    print(train_data.size())
    train_label = torch.argmax(train_label_all[2].reshape(-1, 15), dim=1)
    train_data_loader[2] = DataLoader(TensorDataset(train_data, train_label), batch_size=100, shuffle=True)

    # train_data = train_data_all[3].reshape(-1, 45)[:,18:27]
    train_data = train_data_all[3].reshape(-1, 25, 45)[:, :, 18:27]
    print(train_data.size())
    train_label = torch.argmax(train_label_all[3].reshape(-1, 15), dim=1)
    train_data_loader[3] = DataLoader(TensorDataset(train_data, train_label), batch_size=100, shuffle=True)

    # train_data = train_data_all[4].reshape(-1, 45)[:,18:27]
    train_data = train_data_all[4].reshape(-1, 25, 45)[:, :, 18:27]
    print(train_data.size())
    train_label = torch.argmax(train_label_all[4].reshape(-1, 15), dim=1)
    train_data_loader[4] = DataLoader(TensorDataset(train_data, train_label), batch_size=100, shuffle=True)

    epoch_num = 2000
    result = []
    for i in range(epoch_num):
        print("communicate round {}".format(i + 1))
        net.train()
        sum_parameters = None
        for j in range(5):
            local_parameters = localUpdate(5, train_data_loader[j], net, loss_func, opti, global_parameters, dev)
            if sum_parameters is None:
                sum_parameters = {}
                for key, var in local_parameters.items():
                    sum_parameters[key] = var.clone()
            else:
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] + local_parameters[var]
        for var in global_parameters:
            global_parameters[var] = (sum_parameters[var] / 5)

        net.eval()
        with torch.no_grad():
            if (i + 1) % 1 == 0:
                net.load_state_dict(global_parameters, strict=True)
                sum_accu = 0
                num = 0
                label_list, preds_list = [], []
                for data, label in test_data_loader:
                    data, label = data.to(dev), label.to(dev)
                    preds = net(data)
                    preds = torch.argmax(preds, dim=1)
                    sum_accu += (preds == label).float().mean()
                    num += 1
                    label_list += label.cuda().data.cpu().numpy().tolist()
                    preds_list += preds.cuda().data.cpu().numpy().tolist()
                # print('accuracy: {}'.format(sum_accu / num))
                a_s = accuracy_score(label_list, preds_list)
                p_s = precision_score(label_list, preds_list, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                                      average='macro')
                r_s = recall_score(label_list, preds_list, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                                   average='macro')
                f1_s = f1_score(label_list, preds_list, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                                average='macro')
                print('accuracy_score: {}'.format(a_s))
                print('precision_score: {}'.format(p_s))
                print('recall_score: {}'.format(r_s))
                print('f1_score: {}'.format(f1_s))

        if i == 0:
            result = np.array([[a_s, p_s, r_s, f1_s]])
        else:
            result = np.vstack((result, [[a_s, p_s, r_s, f1_s]]))
        if (i + 1) % 500 == 0:
            np.save('lstm{}epoch_result.npy'.format(i + 1),result)



