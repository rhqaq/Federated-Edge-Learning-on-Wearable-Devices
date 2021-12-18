import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from getDataset import GetWearDataSet
from Models import Mnist_2NN,Mnist_CNN,Simple_LSTM
import torch.nn.functional as F
import os
from torch import optim
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix
from tqdm import tqdm

if __name__=="__main__":
    # wearDataSet = GetWearDataSet(15, 100, 3, 5)
    wearDataSet = GetWearDataSet(15, 100, 3, 125)
    # test_data = wearDataSet.test_data.reshape(-1, 25, 45)[:,:,18:27]
    test_data = wearDataSet.test_data.reshape(-1, 45)[:,18:27]
    print(test_data.size())
    test_label = torch.argmax(wearDataSet.test_label.reshape(-1, 15), dim=1)
    test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=100, shuffle=False)

    # train_data = wearDataSet.train_data.reshape(-1, 25, 45)[:,:,18:27]
    train_data = wearDataSet.train_data.reshape(-1, 45)[:,18:27]
    print(train_data.size())
    train_label = torch.argmax(wearDataSet.train_label.reshape(-1, 15), dim=1)
    train_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=1000, shuffle=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print(torch.cuda.device_count())
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net = Mnist_CNN()
    net = net.to(dev)
    opti = optim.SGD(net.parameters(), lr=0.05)

    loss_func = F.cross_entropy
    epoch_num = 1000
    net.train()
    for i in tqdm(range(epoch_num)):
        for data, label in train_data_loader:
            data, label = data.to(dev), label.to(dev)
            preds = net(data)
            loss = loss_func(preds, label)
            loss.backward()
            opti.step()
            opti.zero_grad()
    net.eval()
    with torch.no_grad():
        label_list, preds_list = [], []
        for data, label in test_data_loader:
            data, label = data.to(dev), label.to(dev)
            preds = net(data)
            preds = torch.argmax(preds, dim=1)
            label_list += label.cuda().data.cpu().numpy().tolist()
            preds_list += preds.cuda().data.cpu().numpy().tolist()
        # print('accuracy: {}'.format(sum_accu / num))
        a_s = accuracy_score(label_list, preds_list)
        p_s = precision_score(label_list, preds_list, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], average='macro')
        r_s = recall_score(label_list, preds_list, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], average='macro')
        f1_s = f1_score(label_list, preds_list, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], average='macro')
        print('accuracy_score: {}'.format(a_s))
        print('precision_score: {}'.format(p_s))
        print('recall_score: {}'.format(r_s))
        print('f1_score: {}'.format(f1_s))
