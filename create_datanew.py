import pandas as pd
import torch
import numpy as np


def dim_turn(ds):
    label = ds[:,4]
    data = ds[:,1:4]
    data = data.reshape(-1, 25, 3)
    label = label[:data.shape[0]]
    return data,label

def reproduce_tensor(t1,t2,td,tl):
    l1 = t1.shape[0]//12
    # print(t2.shape[0])
    # print(l1)
    train11, test11, train12, test12 = t1.split([5 * l1,  l1, 5* l1,  l1], dim=0)
    l2 = t2.shape[0]//12
    # print(l2)
    train21, test21, train22, test22 = t2.split([5 * l2,  l2, 5* l2,  l2], dim=0)

    train11,label11 = dim_turn(train11)
    train12, label12 = dim_turn(train12)
    train21, label21 = dim_turn(train21)
    train22, label22 = dim_turn(train22)

    label1 = torch.cat((label11,label21),0)
    label2 = torch.cat((label12, label22), 0)
    train1 = torch.cat((train11,train21),0)
    train2 = torch.cat((train12, train22), 0)

    test11, tlabel11 = dim_turn(test11)
    test12, tlabel12 = dim_turn(test12)
    test21, tlabel21 = dim_turn(test21)
    test22, tlabel22 = dim_turn(test22)
    test = torch.cat((test11,test12,test21,test22),0)
    testlabel = torch.cat((tlabel11,tlabel12,tlabel21,tlabel22),0)

    if not torch.is_tensor(td):
        td = test
        tl = testlabel
    else:
        td = torch.cat((td, test), 0)
        tl = torch.cat((tl, testlabel), 0)
    return train1,train2,label1,label2,td,tl

def reproduce_tensor_3(t1,t2,t3,td,tl):
    l1 = t1.shape[0]//18
    # print(l1)
    train11, test11, train12, test12, train13, test13 = t1.split([5 * l1,  l1, 5* l1,  l1, 5* l1,  l1], dim=0)
    l2 = t2.shape[0]//18
    # print(l2)
    train21, test21, train22, test22, train23, test23 = t2.split([5 * l2,  l2, 5* l2,  l2, 5* l2,  l2], dim=0)
    l3 = t3.shape[0]//18
    # print(l2)
    train31, test31, train32, test32, train33, test33 = t3.split([5 * l3,  l3, 5* l3,  l3, 5* l3,  l3], dim=0)

    train11,label11 = dim_turn(train11)
    train12, label12 = dim_turn(train12)
    train13, label13 = dim_turn(train13)
    train21, label21 = dim_turn(train21)
    train22, label22 = dim_turn(train22)
    train23, label23 = dim_turn(train23)
    train31, label31 = dim_turn(train31)
    train32, label32 = dim_turn(train32)
    train33, label33 = dim_turn(train33)

    train1 = torch.cat((train11,train21,train31),0)
    train2 = torch.cat((train12, train22,train32), 0)
    train3 = torch.cat((train13, train23, train33), 0)

    label1 = torch.cat((label11,label21,label31),0)
    label2 = torch.cat((label12, label22,label32), 0)
    label3 = torch.cat((label13, label23, label33), 0)

    test11, tlabel11 = dim_turn(test11)
    test12, tlabel12 = dim_turn(test12)
    test13, tlabel13 = dim_turn(test13)
    test21, tlabel21 = dim_turn(test21)
    test22, tlabel22 = dim_turn(test22)
    test23, tlabel23 = dim_turn(test23)
    test31, tlabel31 = dim_turn(test31)
    test32, tlabel32 = dim_turn(test32)
    test33, tlabel33 = dim_turn(test33)

    test = torch.cat((test11,test12,test13,test21,test22,test23,test31,test32,test33),0)
    testlabel = torch.cat((tlabel11,tlabel12,tlabel13,tlabel21,tlabel22,tlabel23,tlabel31,tlabel32,tlabel33),0)
    if not torch.is_tensor(td):
        td = test
        tl = testlabel
    else:
        td = torch.cat((td, test), 0)
        tl = torch.cat((tl, testlabel), 0)
    return train1,train2,train3,label1,label2,label3,td,tl


def reproduce_tensor_2(t1,t2,t3,td,tl):
    l1 = t1.shape[0]//12
    # print(l1)
    train11, test11, train12, test12 = t1.split([5 * l1,  l1, 5* l1,  l1], dim=0)
    l2 = t2.shape[0]//12
    # print(l2)
    train21, test21, train22, test22 = t2.split([5 * l2,  l2, 5* l2,  l2], dim=0)
    l3 = t3.shape[0]//12
    # print(l2)
    train31, test31, train32, test32 = t3.split([5 * l3,  l3, 5* l3,  l3], dim=0)

    train11,label11 = dim_turn(train11)
    train12, label12 = dim_turn(train12)
    train21, label21 = dim_turn(train21)
    train22, label22 = dim_turn(train22)
    train31, label31 = dim_turn(train31)
    train32, label32 = dim_turn(train32)


    train1 = torch.cat((train11,train21,train31),0)
    train2 = torch.cat((train12, train22,train32), 0)
    label1 = torch.cat((label11,label21,label31),0)
    label2 = torch.cat((label12, label22,label32), 0)

    test11, tlabel11 = dim_turn(test11)
    test12, tlabel12 = dim_turn(test12)
    test21, tlabel21 = dim_turn(test21)
    test22, tlabel22 = dim_turn(test22)
    test31, tlabel31 = dim_turn(test31)
    test32, tlabel32 = dim_turn(test32)

    test = torch.cat((test11,test12,test21,test22,test31,test32),0)
    testlabel = torch.cat((tlabel11,tlabel12,tlabel21,tlabel22,tlabel31,tlabel32),0)
    if not torch.is_tensor(td):
        td = test
        tl = testlabel
    else:
        td = torch.cat((td, test), 0)
        tl = torch.cat((tl, testlabel), 0)
    return train1,train2,label1,label2,td,tl


def create_new_data():
    all_data = [0] * 100
    all_label = [0] *100
    test_data = 0
    test_label = 0
    nums = [0]*8
    # max1 =0
    # min1 = 100000000000000
    # max2 =0
    # min2 = 100000000000000
    # max3 =0

    for i in range(15):
        df = pd.read_csv(
            r'../Activity Recognition from Single Chest-Mounted Accelerometer/{}.csv'.format(
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
            all_data[7*i],all_data[7*i+1],all_label[7*i],all_label[7*i+1],  test_data, test_label = reproduce_tensor(np1,np2,test_data,test_label)

            df3 = df[df[4]==2]
            np3 = torch.FloatTensor(np.array(df3)[:df3.shape[0]-df3.shape[0]%300,:])
            df4 = df[df[4]==3]
            np4 = torch.FloatTensor(np.array(df4)[:df4.shape[0]-df4.shape[0]%300,:])
            # print(np3.shape)
            all_data[7*i+2],all_data[7*i+3],all_label[7*i+2],all_label[7*i+3], test_data, test_label = reproduce_tensor(np3,np4,test_data,test_label)

            df5 = df[df[4]==4]
            np5 = torch.FloatTensor(np.array(df5)[:df5.shape[0]-df5.shape[0]%450,:])
            df6 = df[df[4]==5]
            np6 = torch.FloatTensor(np.array(df6)[:df6.shape[0]-df6.shape[0]%450,:])
            df7 = df[df[4]==6]
            np7 = torch.FloatTensor(np.array(df7)[:df7.shape[0]-df7.shape[0]%450,:])
            all_data[7*i+4],all_data[7*i+5],all_data[7*i+6], all_label[7*i+4],all_label[7*i+5],all_label[7*i+6], test_data, test_label = reproduce_tensor_3(np5,np6,np7,test_data,test_label)
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
            all_data[6 * i + 10], all_data[6 * i + 11],all_label[6 * i + 10], all_label[6 * i + 11], test_data, test_label = reproduce_tensor(np1, np2, test_data,test_label)

            df3 = df[df[4] == 2]
            np3 = torch.FloatTensor(np.array(df3)[:df3.shape[0] - df3.shape[0] % 300, :])
            df4 = df[df[4] == 3]
            np4 = torch.FloatTensor(np.array(df4)[:df4.shape[0] - df4.shape[0] % 300, :])
            all_data[6 * i + 12], all_data[6 * i + 13], all_label[6 * i + 12], all_label[6 * i + 13], test_data, test_label = reproduce_tensor(np3, np4, test_data,test_label)

            df5 = df[df[4] == 4]
            np5 = torch.FloatTensor(np.array(df5)[:df5.shape[0] - df5.shape[0] % 300, :])
            df6 = df[df[4] == 5]
            np6 = torch.FloatTensor(np.array(df6)[:df6.shape[0] - df6.shape[0] % 300, :])
            df7 = df[df[4] == 6]
            np7 = torch.FloatTensor(np.array(df7)[:df7.shape[0] - df7.shape[0] % 300, :])
            all_data[6 * i + 14], all_data[6 * i + 15], all_label[6 * i + 14], all_label[6 * i + 15], test_data, test_label = reproduce_tensor_2(np5, np6, np7, test_data,test_label)
    # print(nums)
    #     # print(np1.shape)
    # print(all_data[98])
    # print(test_data.shape)
    # print(test_label.shape)
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
create_new_data()