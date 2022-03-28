import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from Models import LSTM, MLP
from clients import ClientsGroup, client
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net = LSTM()

    print(torch.cuda.is_available())
    net = net.to(dev)
    loss_func = F.cross_entropy
    opti = optim.SGD(net.parameters(), lr=0.05)
    myClients = ClientsGroup(15, 100, 3, 5, dev)
    testDataLoader = myClients.test_data_loader

    num_in_comm = int(max(100 * 0.1, 1))

    global_parameters = {}
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()

    result = []

    with torch.no_grad():
        net.load_state_dict(torch.load('D:\wearable\code\checkpoints\9feature_lstm_num_comm1999_an15_D5_al1_E5_B10_lr0.05_num_clients100_cf0.1.pth'), strict=True)
        sum_accu = 0
        num = 0
        label_list, preds_list = [], []
        for data, label in testDataLoader:
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