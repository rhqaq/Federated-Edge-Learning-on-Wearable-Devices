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
# from joblib import Parallel,delayed
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-an', '--action_num', type=int, default=15, help='numer of the actions')
parser.add_argument('-nc', '--num_of_clients', type=int, default=100, help='numer of the clients')
parser.add_argument('-sn', '--shard_num', type=int, default=3, help='numer of the shard')
parser.add_argument('-dn', '--divide_num', type=int, default=125, help='numer of the time sequence division')
parser.add_argument('-cf', '--cfraction', type=float, default=0.1,
                    help='C fraction, 0 means 1 client, 1 means total clients')
parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')
parser.add_argument('-B', '--batchsize', type=int, default=250, help='local train batch size')
parser.add_argument('-mn', '--model_name', type=str, default='mlp', help='the model to train')
parser.add_argument('-al', '--alpha', type=float, default=1, help='numer of the shard')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.05, help="learning rate, \
                    use value from origin paper as default")
parser.add_argument('-vf', "--val_freq", type=int, default=1, help="model validation frequency(of communications)")
parser.add_argument('-sf', '--save_freq', type=int, default=500, help='global model save frequency(of communication)')
parser.add_argument('-ncomm', '--num_comm', type=int, default=2000, help='number of communications')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')


def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


if __name__ == "__main__":
    args = parser.parse_args()
    args = args.__dict__

    test_mkdir(args['save_path'])

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # dev = torch.device("cpu")
    net = None
    if args['model_name'] == 'lstm':
        net = LSTM()
        # net = torch.load('/root/rh/wearable_FL/code/FedAvg/use_pytorch/checkpoints/lstm_num_comm999_D5_al0.5_E5_B10_lr0.05_num_clients100_cf0.1.pth') #继续训练通信1000轮

    elif args['model_name'] == 'mlp':
        net = MLP()
        # net = torch.load('/root/rh/wearable_FL/code/FedAvg/use_pytorch/checkpoints/mlp_num_comm999_D125_E5_B10_lr0.05_num_clients100_cf0.1.pth')
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)
    print(torch.cuda.is_available())
    net = net.to(dev)
    loss_func = F.cross_entropy
    opti = optim.SGD(net.parameters(), lr=args['learning_rate'])
    myClients = ClientsGroup(args['action_num'], args['num_of_clients'], args['shard_num'], args['divide_num'], dev)
    testDataLoader = myClients.test_data_loader

    num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))

    global_parameters = {}
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()

    result = []
    for i in range(args['num_comm']):
        # print("communicate round {}".format(i+1))
        print("communicate round {}".format(i + 1))  # 继续训练
        # net.train()
        order = np.random.permutation(args['num_of_clients'])
        clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]

        sum_parameters = None
        for client in tqdm(clients_in_comm):
            local_parameters = myClients.clients_set[client].localUpdate(args['epoch'], args['batchsize'], net,
                                                                         loss_func, opti, global_parameters,
                                                                         args['alpha'])
            if sum_parameters is None:
                sum_parameters = {}
                for key, var in local_parameters.items():
                    sum_parameters[key] = var.clone()
            else:
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] + local_parameters[var]

        for var in global_parameters:
            global_parameters[var] = (sum_parameters[var] / num_in_comm)

        # net.eval()
        with torch.no_grad():
            if (i + 1) % args['val_freq'] == 0:
                net.load_state_dict(global_parameters, strict=True)
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
                p_s = precision_score(label_list, preds_list, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,14], average='macro')
                r_s = recall_score(label_list, preds_list, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,14], average='macro')
                f1_s = f1_score(label_list, preds_list, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,14], average='macro')
                print('accuracy_score: {}'.format(a_s))
                print('precision_score: {}'.format(p_s))
                print('recall_score: {}'.format(r_s))
                print('f1_score: {}'.format(f1_s))

                # print('roc_auc_score: {}'.format(roc_auc_score(label_list,preds_list,labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], average='macro')))

        if i == 0:
            result = np.array([[a_s, p_s, r_s, f1_s]])
        else:
            result = np.vstack((result, [[a_s, p_s, r_s, f1_s]]))
        # print(result)
        if (i + 1) % args['save_freq'] == 0:
            np.save(os.path.join(args['save_path'],
                                 '9feature_{}_an{}_sn{}_D{}_al{}_E{}_B{}_lr{}_num_clients{}_cf{}.npy'.format(
                                     args['model_name'],
                                     args['action_num'],
                                     args['shard_num'],
                                     args['divide_num'],
                                     args['alpha'],
                                     args['epoch'],
                                     args['batchsize'],
                                     args['learning_rate'],
                                     args['num_of_clients'],
                                     args['cfraction'])), result)
            torch.save(net, os.path.join(args['save_path'],
                                         '9feature_{}_num_comm{}_an{}_sn_D{}_al{}_E{}_B{}_lr{}_num_clients{}_cf{}.pth'.format(
                                             args['model_name'],
                                             i,
                                             args['action_num'],
                                             args['shard_num'],
                                             args['divide_num'],
                                             args['alpha'],
                                             args['epoch'],
                                             args['batchsize'],
                                             args['learning_rate'],
                                             args['num_of_clients'],
                                             args['cfraction'])))

