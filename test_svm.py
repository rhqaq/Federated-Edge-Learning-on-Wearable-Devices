import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from getDataset import GetWearDataSet
from Models import LSTM,MLP,Simple_LSTM,SVM
import torch.nn.functional as F
import os
from torch import optim
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix
from tqdm import tqdm
import torch.nn as nn
from create_data import create_new_data,create_pamap_data
import warnings
from sklearn import preprocessing
from sklearn import svm

if __name__=="__main__":
    # min_max_scaler = preprocessing.MinMaxScaler()
    train_data_all, train_label_all, test_data_all, test_label_all = create_pamap_data()
    # test_data = test_data_all.reshape(test_data_all.shape[0], -1)
    test_data = test_data_all
    test_label = test_label_all
    warnings.filterwarnings("ignore", category=UserWarning)
    # wearDataSet = GetWearDataSet(15, 100, 3, 5)
    # wearDataSet = GetWearDataSet(10, 100, 2, 125)
    # test_data = wearDataSet.test_data.reshape(-1, 25, 45)[:,:,18:27]
    # test_data = wearDataSet.test_data.reshape(-1, 45)[:,18:27]
    # test_data = wearDataSet.test_data.reshape(-1, 45)
    # test_data = wearDataSet.test_data.reshape(-1, 25, 45)
    print(test_data.size())
    # test_label = torch.argmax(wearDataSet.test_label.reshape(-1, 10), dim=1)
    # test_label = wearDataSet.test_label.reshape(-1, 15)
    print(test_label.size())


    # train_data = wearDataSet.train_data.reshape(-1, 25, 45)[:,:,18:27]
    # train_data = wearDataSet.train_data.reshape(-1, 45)[:,18:27]
    # train_data = wearDataSet.train_data.reshape(-1, 25, 45)
    train_data = 0
    train_label = 0
    for i in range(len(train_data_all)):
        # local_data, local_label = train_data_all[i].reshape(train_data_all[i].shape[0], -1), train_label_all[i]
        local_data, local_label = train_data_all[i], train_label_all[i]
        if not torch.is_tensor(train_data):
            train_data = local_data
            train_label = local_label
        else:
            train_data = torch.cat((train_data, local_data), 0)
            train_label = torch.cat((train_label, local_label), 0)
    print(train_data.shape)
    print(np.unique((train_label.numpy())))
    print(np.unique((test_label.numpy())))
    # train_data = wearDataSet.train_data.reshape(-1, 45)
    # print(train_data.size())
    # train_label = torch.argmax(wearDataSet.train_label.reshape(-1, 10), dim=1)
    # train_data = torch.FloatTensor(min_max_scaler.fit_transform(train_data))
    # test_data = torch.FloatTensor(min_max_scaler.transform(test_data))
    print(train_data)

    train_data = np.array(train_data)
    train_label = np.array(train_label)
    test_data = np.array(test_data)
    test_label = np.array(test_label)
    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(train_data, train_label)
    preds_list = clf.predict(test_data)
    label_list = test_label

    num_list = [i for i in range(14)]
    a_s = accuracy_score(label_list, preds_list)
    p_s = precision_score(label_list, preds_list, labels=num_list, average='macro')
    r_s = recall_score(label_list, preds_list, labels=num_list, average='macro')
    f1_s = f1_score(label_list, preds_list, labels=num_list, average='macro')

    print('accuracy_score: {}'.format('%.4f' % a_s))
    print('precision_score: {}'.format('%.4f' % p_s))
    print('recall_score: {}'.format('%.4f' % r_s))
    print('f1_score: {}'.format('%.4f' % f1_s))
    # train_data_loader = DataLoader(TensorDataset(train_data, train_label), batch_size=1024, shuffle=True)
    # test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=1024, shuffle=False)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # print(torch.cuda.device_count())
    # dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # print(dev)
    #
    # classnum = 14
    # print(train_data.shape[1])
    # # net = MLP(train_data.shape[1],classnum)
    # net = LSTM(43,classnum)
    # net = net.to(dev)
    # opti = optim.SGD(net.parameters(), lr=0.05)
    #
    # # loss_func = F.multi_margin_loss
    # loss_func = F.cross_entropy
    # epoch_num = 1000
    # c = 0.001
    # net.train()
    # for i in tqdm(range(epoch_num)):
    #     for data, label in train_data_loader:
    #         data, label = data.to(dev), label.to(dev)
    #         preds = net(data)
    #         # print(preds.size())
    #         # print(label.size())
    #         loss = loss_func(preds, label.long())
    #         # weight = net.fc.weight.squeeze()
    #         # loss += c * torch.sum(torch.abs(weight))
    #         # print(loss.size())
    #         loss.backward()
    #         opti.step()
    #         opti.zero_grad()
    # net.eval()
    # with torch.no_grad():
    #     label_list, preds_list, prob = [], [], []
    #     for data, label in test_data_loader:
    #         label = label.long()
    #         data, label = data.to(dev), label.to(dev)
    #         preds = net(data)
    #         if len(prob) == 0:
    #             prob = F.softmax(preds, dim=1).cuda().data.cpu().numpy()
    #         else:
    #             prob = np.append(prob, F.softmax(preds, dim=1).cuda().data.cpu().numpy(), axis=0)
    #         preds = torch.argmax(preds, dim=1)
    #         label_list += label.cuda().data.cpu().numpy().tolist()
    #         preds_list += preds.cuda().data.cpu().numpy().tolist()
    #     # print('accuracy: {}'.format(sum_accu / num))
    #     num_list = [i for i in range(classnum)]
    #     a_s = accuracy_score(label_list, preds_list)
    #     p_s = precision_score(label_list, preds_list, labels=num_list, average='macro')
    #     r_s = recall_score(label_list, preds_list, labels=num_list, average='macro')
    #     f1_s = f1_score(label_list, preds_list, labels=num_list, average='macro')
    #     # auc_s = roc_auc_score(label_list, prob, multi_class='ovo',
    #     #                       labels=num_list, average='macro')
    #
    #     # p_s = precision_score(label_list, preds_list, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,14], average='macro')
    #     # r_s = recall_score(label_list, preds_list, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,14], average='macro')
    #     # f1_s = f1_score(label_list, preds_list, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,14], average='macro')
    #     # auc_s = roc_auc_score(label_list, prob, multi_class='ovo',
    #     #                       labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], average='macro')
    #
    #     print('accuracy_score: {}'.format('%.4f' %a_s))
    #     print('precision_score: {}'.format('%.4f' %p_s))
    #     print('recall_score: {}'.format('%.4f' %r_s))
    #     print('f1_score: {}'.format('%.4f' %f1_s))
    #     # print('auc_score: {}'.format(auc_s))
    #
    #     label_list, preds_list, prob = [], [], []
    #     for data, label in train_data_loader:
    #         label = label.long()
    #         data, label = data.to(dev), label.to(dev)
    #         preds = net(data)
    #         if len(prob) == 0:
    #             prob = F.softmax(preds, dim=1).cuda().data.cpu().numpy()
    #         else:
    #             prob = np.append(prob, F.softmax(preds, dim=1).cuda().data.cpu().numpy(), axis=0)
    #         preds = torch.argmax(preds, dim=1)
    #         label_list += label.cuda().data.cpu().numpy().tolist()
    #         preds_list += preds.cuda().data.cpu().numpy().tolist()
    #     # print('accuracy: {}'.format(sum_accu / num))
    #     num_list = [i for i in range(classnum)]
    #     a_s = accuracy_score(label_list, preds_list)
    #     p_s = precision_score(label_list, preds_list, labels=num_list, average='macro')
    #     r_s = recall_score(label_list, preds_list, labels=num_list, average='macro')
    #     f1_s = f1_score(label_list, preds_list, labels=num_list, average='macro')
    #     # auc_s = roc_auc_score(label_list, prob, multi_class='ovo',
    #     #                       labels=num_list, average='macro')
    #
    #     # p_s = precision_score(label_list, preds_list, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,14], average='macro')
    #     # r_s = recall_score(label_list, preds_list, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,14], average='macro')
    #     # f1_s = f1_score(label_list, preds_list, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,14], average='macro')
    #     # auc_s = roc_auc_score(label_list, prob, multi_class='ovo',
    #     #                       labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], average='macro')
    #
    #     print('accuracy_score: {}'.format('%.4f' % a_s))
    #     print('precision_score: {}'.format('%.4f' % p_s))
    #     print('recall_score: {}'.format('%.4f' % r_s))
    #     print('f1_score: {}'.format('%.4f' % f1_s))