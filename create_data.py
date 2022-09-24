import pandas as pd
import torch
import numpy as np
import os
from tqdm import tqdm
from sklearn import preprocessing


def dim_turn(ds,feature_num,scale):
    label = ds[:,0]
    # data = torch.FloatTensor(scale.transform(ds[:,1:]))
    data = ds[:,1:]
    data = data.reshape(-1, 100, feature_num)
    label = label[:data.shape[0]]
    return data,label


def reproduce_tensor(t1,t2,td):
    l1 = t1.shape[0]//12
    # print(t2.shape[0])
    # print(l1)
    train11, test11, train12, test12 = t1.split([5 * l1,  l1, 5* l1,  l1], dim=0)
    l2 = t2.shape[0]//12
    # print(l2)
    train21, test21, train22, test22 = t2.split([5 * l2,  l2, 5* l2,  l2], dim=0)

    train1 = torch.cat((train11,train21),0)
    train2 = torch.cat((train12, train22), 0)
    test = torch.cat((test11,test12,test21,test22),0)
    if not torch.is_tensor(td):
        td = test
    else:
        td = torch.cat((td, test), 0)
    return train1,train2,td

def reproduce_tensor_3(t1,t2,t3,td):
    l1 = t1.shape[0]//18
    # print(l1)
    train11, test11, train12, test12, train13, test13 = t1.split([5 * l1,  l1, 5* l1,  l1, 5* l1,  l1], dim=0)
    l2 = t2.shape[0]//18
    # print(l2)
    train21, test21, train22, test22, train23, test23 = t2.split([5 * l2,  l2, 5* l2,  l2, 5* l2,  l2], dim=0)
    l3 = t3.shape[0]//18
    # print(l2)
    train31, test31, train32, test32, train33, test33 = t3.split([5 * l3,  l3, 5* l3,  l3, 5* l3,  l3], dim=0)

    train1 = torch.cat((train11,train21,train31),0)
    train2 = torch.cat((train12, train22,train32), 0)
    train3 = torch.cat((train13, train23, train33), 0)
    test = torch.cat((test11,test12,test13,test21,test22,test23,test31,test32,test33),0)
    if not torch.is_tensor(td):
        td = test
    else:
        td = torch.cat((td, test), 0)
    return train1,train2,train3,td


def reproduce_tensor_2(t1,t2,t3,td):
    l1 = t1.shape[0]//12
    # print(l1)
    train11, test11, train12, test12 = t1.split([5 * l1,  l1, 5* l1,  l1], dim=0)
    l2 = t2.shape[0]//12
    # print(l2)
    train21, test21, train22, test22 = t2.split([5 * l2,  l2, 5* l2,  l2], dim=0)
    l3 = t3.shape[0]//12
    # print(l2)
    train31, test31, train32, test32 = t3.split([5 * l3,  l3, 5* l3,  l3], dim=0)

    train1 = torch.cat((train11,train21,train31),0)
    train2 = torch.cat((train12, train22,train32), 0)
    test = torch.cat((test11,test12,test21,test22,test31,test32),0)
    if not torch.is_tensor(td):
        td = test
    else:
        td = torch.cat((td, test), 0)
    return train1,train2,td


def create_new_data():
    all_data = [0] * 100
    test_data = 0
    nums = [0]*8
    # max1 =0
    # min1 = 100000000000000
    # max2 =0
    # min2 = 100000000000000
    # max3 =0
    # min3 = 100000000000000
    vl1 = list()
    vl2 = list()
    vl3 = list()
    for i in range(15):
        df = pd.read_csv(
            r'D:\Activity Recognition from Single Chest-Mounted Accelerometer\Activity Recognition from Single Chest-Mounted Accelerometer\{}.csv'.format(
                i + 1), header=None)
        df[4] = df[4].map(lambda x: int(x-1))
        df[3] = df[3].map(lambda x: (x-1970) / 94)
        df[2] = df[2].map(lambda x: (x-2382) / 100)
        df[1] = df[1].map(lambda x: (x-1987) / 111)
        # vl1.extend( df[1].to_numpy())
        # vl2.extend(df[2].to_numpy())
        # vl3.extend(df[3].to_numpy())
        # print(nums)
        # print(df.head)
        # for index,row in df.iterrows():
        #     # if row[1]>max1:
        #     #     max1 = row[1]
        #     # if row[1]<min1:
        #     #     min1 = row[1]
        #     if row[2]>max2:
        #         max2 = row[2]
        #     if row[2]<min2:
        #         min2 = row[2]
        #     if row[3]>max3:
        #         max3 = row[3]
        #     if row[3]<min3:
        #         min3 = row[3]

        if i+1<=10:
            # df = pd.read_csv(r'D:\Activity Recognition from Single Chest-Mounted Accelerometer\Activity Recognition from Single Chest-Mounted Accelerometer\{}.csv'.format(i+1),header=None)
            # nums = [0]*8
            # # print(nums)
            # # print(df.head)
            # for index,row in df.iterrows():
            #     nums[int(row[4])-1] += 1
            df1 = df[df[4]==0]
            np1 = torch.FloatTensor(np.array(df1)[:df1.shape[0]-df1.shape[0]%300,:])
            df2 = df[df[4]==1]
            np2 = torch.FloatTensor(np.array(df2)[:df2.shape[0]-df2.shape[0]%300,:])
            all_data[7*i],all_data[7*i+1], test_data = reproduce_tensor(np1,np2,test_data)

            df3 = df[df[4]==2]
            np3 = torch.FloatTensor(np.array(df3)[:df3.shape[0]-df3.shape[0]%300,:])
            df4 = df[df[4]==3]
            np4 = torch.FloatTensor(np.array(df4)[:df4.shape[0]-df4.shape[0]%300,:])
            # print(np3.shape)
            all_data[7*i+2],all_data[7*i+3], test_data = reproduce_tensor(np3,np4,test_data)

            df5 = df[df[4]==4]
            np5 = torch.FloatTensor(np.array(df5)[:df5.shape[0]-df5.shape[0]%450,:])
            df6 = df[df[4]==5]
            np6 = torch.FloatTensor(np.array(df6)[:df6.shape[0]-df6.shape[0]%450,:])
            df7 = df[df[4]==6]
            np7 = torch.FloatTensor(np.array(df7)[:df7.shape[0]-df7.shape[0]%450,:])
            all_data[7*i+4],all_data[7*i+5],all_data[7*i+6], test_data = reproduce_tensor_3(np5,np6,np7,test_data)
        else:
            # df = pd.read_csv(
            #     r'D:\Activity Recognition from Single Chest-Mounted Accelerometer\Activity Recognition from Single Chest-Mounted Accelerometer\{}.csv'.format(
            #         i + 1), header=None)
            # nums = [0] * 8
            # # print(nums)
            # # print(df.head)
            # for index, row in df.iterrows():
            #     nums[int(row[4]) - 1] += 1
            df1 = df[df[4] == 0]
            np1 = torch.FloatTensor(np.array(df1)[:df1.shape[0] - df1.shape[0] % 300, :])
            df2 = df[df[4] == 1]
            np2 = torch.FloatTensor(np.array(df2)[:df2.shape[0] - df2.shape[0] % 300, :])
            all_data[6 * i + 10], all_data[6 * i + 11], test_data = reproduce_tensor(np1, np2, test_data)

            df3 = df[df[4] == 2]
            np3 = torch.FloatTensor(np.array(df3)[:df3.shape[0] - df3.shape[0] % 300, :])
            df4 = df[df[4] == 3]
            np4 = torch.FloatTensor(np.array(df4)[:df4.shape[0] - df4.shape[0] % 300, :])
            all_data[6 * i + 12], all_data[6 * i + 13], test_data = reproduce_tensor(np3, np4, test_data)

            df5 = df[df[4] == 4]
            np5 = torch.FloatTensor(np.array(df5)[:df5.shape[0] - df5.shape[0] % 300, :])
            df6 = df[df[4] == 5]
            np6 = torch.FloatTensor(np.array(df6)[:df6.shape[0] - df6.shape[0] % 300, :])
            df7 = df[df[4] == 6]
            np7 = torch.FloatTensor(np.array(df7)[:df7.shape[0] - df7.shape[0] % 300, :])
            all_data[6 * i + 14], all_data[6 * i + 15], test_data = reproduce_tensor_2(np5, np6, np7, test_data)
    # print(nums)
    #     # print(np1.shape)
    # print(all_data[98]) 282 3828
    # print(min2)
    # print(max2)
    # print(min3)
    # print(max3)
    # print(np.mean(vl1))
    # print(np.std(vl1))
    # print(np.mean(vl2))
    # print(np.std(vl2))
    # print(np.mean(vl3))
    # print(np.std(vl3))
    return all_data,test_data

    # for j in range(100):
    #     print(all_data[j].shape)
# 2+2+3
# 前10个人的数据分成7份，前两种类型2份，中两种2份，后三种3份。前四种数据要取50倍数，后三种要取75的倍数
# 后5人数据分成6份，都是取50倍数
# 一共得到100份数据，每份拥有的类型数是2或3
# 把每类类型的数据取最大倍数的数，然后均分。再把要s'd合并的类型数据合并，得到一个client的数据，再s'd
# create_new_data()

def create_MotionSense_data():
    all_data = []
    all_label = []
    test_data = 0
    test_label = 0
    index = 0
    for k in range(6):
        data_dir = r'..\archive\A_DeviceMotion_data\{}'.format(k+1)
        label_dir = {'dws':0,'ups':1,'sit':2,'std':3,'wlk':4,'jog':5}
        file_paths = []

        for file in os.listdir(data_dir):
            l = label_dir[file[:3]]
            path = os.path.join(data_dir, file)
            for root, dirs, files in os.walk(path):
                for i, file1 in enumerate(files):
                    path = os.path.join(root, file1)
                    # print(file1[4:-4])
                    a= pd.read_csv(path, header=None)
                    a = np.array(a)
                    a = a[1:,1:]
                    # print(a[0,])
                    a = a[:a.shape[0]-a.shape[0]%150, :]
                    a = torch.FloatTensor(a.astype(float))
                    tz = a.shape[0]//6
                    a_train, a_test = a.split([5*tz,tz],dim=0)
                    # print(a_train.shape)
                    a_train_data,a_train_label = dim_turn(a_train,12)
                    a_test_data,a_test_label = dim_turn(a_test,12)



                    all_data.append(a_train_data)
                    all_label.append(a_train_label)


                    # index = int(file1[4:-4]) - 1
                    # print(index)
                    # if not torch.is_tensor(all_data[24*k+index]):
                    #     all_data[24*k+index] = a_train_data
                    #     all_label[24*k+index] = a_train_label
                    # else:
                    #     all_data[24*k+index] = torch.cat((all_data[24*k+index], a_train_data), 0)
                    #     all_label[24*k+index] = torch.cat((all_label[24*k+index],a_train_label), 0)


                    if not torch.is_tensor(test_data):
                        test_data = a_test_data
                        test_label = a_test_label
                    else:
                        test_data = torch.cat((test_data,a_test_data),0)
                        test_label = torch.cat((test_label,a_test_label),0)
                # a['label'] = [l]*a.shape[0]
                # print(a.shape)
                # a.to_csv(path, index=False, header=False)
    print(len(all_data))
    print(test_data.shape)
    return all_data,all_label,test_data,test_label



def add_label():
    k=5
    data_dir = r'..\archive\A_DeviceMotion_data\{}'.format(k + 1)
    label_dir = {'dws': 0, 'ups': 1, 'sit': 2, 'std': 3, 'wlk': 4, 'jog': 5}
    file_paths = []
    for file in os.listdir(data_dir):
        l = label_dir[file[:3]]
        path = os.path.join(data_dir, file)
        for root, dirs, files in os.walk(path):
            for i, file1 in enumerate(files):
                path = os.path.join(root, file1)
                a = pd.read_csv(path, header=None)
                a['label'] = [l]*a.shape[0]
                print(a.shape)
                a.to_csv(path, index=False, header=False)


def read_dat(fp):
    with open(fp, encoding='utf-8') as f:
        lines = f.readlines()
    for i in range(len(lines)):
        lines[i] = np.array(lines[i].split(' '),dtype=np.float64)
    a = np.array(lines)
    return a


def create_pamap_data():
    all_data = []
    all_label = []
    test_data = 0
    test_label = 0
    min_max_scaler = preprocessing.MinMaxScaler()
    # x1 = read_dat('D:\wearable\PAMAP_Dataset\PAMAP_Dataset\Indoor\subject1.dat')
    # x2 = read_dat('D:\wearable\PAMAP_Dataset\PAMAP_Dataset\Outdoor\subject2.dat')
    # x = np.append(x1,x2,axis=0)
    # print(x.shape)
    # x = x[:,2:]
    # min_max_scaler = min_max_scaler.fit(x)
    flag = True
    client_list = [0]*24
    # data_dir = 'D:\wearable\PAMAP_Dataset\PAMAP_Dataset\Indoor'
    for data_dir in ['D:\wearable\PAMAP_Dataset\PAMAP_Dataset\Indoor','D:\wearable\PAMAP_Dataset\PAMAP_Dataset\Outdoor']:
        for file in os.listdir(data_dir):
            path = os.path.join(data_dir, file)
            a = read_dat(path)
            a = np.delete(a, 0, axis=1)
            # a = np.delete(a, [0,16,15,14,13,30,29,28,27,44,43,42,41], axis=1)
            # print(a.shape)
            client_num = 1
            # 按顺序将label一样的矩阵提取出来
            while(a.shape[0]):
                label = int(a[0, 0])
                for i in range(a.shape[0]):
                    # print(a[i, 1])
                    a[i,0] = int(a[i,0])
                    # if a[0,1]==0:
                    #     a = a[1:, :]
                    #     continue
                    if a[i,0]!=label:
                        if label != 0:
                            client_list[label] += 1
                            client_num += 1
                            b = a[:i,:]
                            b = b[:b.shape[0] - b.shape[0] % 600, :]
                            b = torch.FloatTensor(b.astype(float))
                            tz = b.shape[0] // 6
                            b_train, b_test = b.split([5 * tz, tz], dim=0)
                            # print(a_train.shape)
                            b_train_data, b_train_label = dim_turn(b_train, 43,min_max_scaler)
                            b_test_data, b_test_label = dim_turn(b_test, 43,min_max_scaler)

                            all_data.append(b_train_data)
                            all_label.append(b_train_label)
                            if not torch.is_tensor(test_data):
                                test_data = b_test_data
                                test_label = b_test_label
                            else:
                                test_data = torch.cat((test_data, b_test_data), 0)
                                test_label = torch.cat((test_label, b_test_label), 0)
                        label = a[i, 0]
                        a = a[i:,:]
                        break
                if i+1 == a.shape[0]:
                    if label != 0:
                        b=a
                        b = b[:b.shape[0] - b.shape[0] % 600, :]
                        b = torch.FloatTensor(b.astype(float))
                        tz = b.shape[0] // 6
                        b_train, b_test = b.split([5 * tz, tz], dim=0)
                        # print(a_train.shape)
                        b_train_data, b_train_label = dim_turn(b_train, 43,min_max_scaler)
                        b_test_data, b_test_label = dim_turn(b_test, 43,min_max_scaler)

                        all_data.append(b_train_data)
                        all_label.append(b_train_label)
                        if not torch.is_tensor(test_data):
                            test_data = b_test_data
                            test_label = b_test_label
                        else:
                            test_data = torch.cat((test_data, b_test_data), 0)
                            test_label = torch.cat((test_label, b_test_label), 0)
                    break
            print('该文件{}'.format(client_num))

    print(client_list)
    for i in range(len(all_data)):
        print(all_data[i].shape)
    print(all_data[0][0,:])
    print(len(all_data))
    enc = preprocessing.LabelEncoder()  # 获取一个LabelEncoder
    enc = enc.fit([ 1,  2, 3, 10, 11, 12, 13, 14, 15, 16, 20, 21, 22, 23])  # 训练LabelEncoder
    test_label = torch.LongTensor(enc.transform(test_label))
    for i in range(len(all_label)):
        all_label[i] = torch.LongTensor(enc.transform(all_label[i]))
    print(test_data.shape)
    print(test_label.shape)
    return all_data, all_label, test_data, test_label
# add_label()
# create_MotionSense_data()

if __name__=="__main__":
    create_pamap_data()