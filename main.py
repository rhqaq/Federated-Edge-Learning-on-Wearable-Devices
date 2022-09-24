import fedavg_api
import my_model_trainer_classification
from Models import LogisticRegression
import argparse
from my_model_trainer_classification import MyModelTrainer
from fedavg_api import FedAvgAPI
import os
import torch
from getDataset import GetWearDataSet


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='resnet56', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='./../../../data/cifar10',
                        help='data directory')

    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')

    parser.add_argument('--batch_size', type=int, default=1024, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='SGD with momentum; adam')

    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)

    parser.add_argument('--epochs', type=int, default=5, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--client_num_in_total', type=int, default=100, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=10, metavar='NN',
                        help='number of workers')

    parser.add_argument('--comm_round', type=int, default=1000,
                        help='how many round of communications we shoud use')

    parser.add_argument('--frequency_of_the_test', type=int, default=1,
                        help='the frequency of the algorithms')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    classnum = 10
    featurenum = 45
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
    args = add_args(parser)
    model = LogisticRegression(featurenum,classnum)
    model_trainer = MyModelTrainer(model,args)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print(torch.cuda.device_count())
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(dev)

    wearDataSet = GetWearDataSet(10, 100, 2, 125)
    test_data = wearDataSet.test_data
    test_label = wearDataSet.test_label
    train_data = wearDataSet.train_data
    train_label = wearDataSet.train_label
    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, classnum]
    FedAvgAPI(dataset,dev,args,model_trainer)