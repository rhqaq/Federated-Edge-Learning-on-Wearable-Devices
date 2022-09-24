import numpy as np
import matplotlib.pyplot as plt

data_acc = [0]*11
label_name = [0]*11

data_acc[0] = np.load(r'./checkpoints/added/adam45feature_lstm_an15_sn3_D5_al0.9_E1_Ball_lr0.05_num_clients100_cf0.1.npy')
data_acc[1] = np.load(r'./checkpoints/added/adam45feature_lstm_an15_sn3_D5_al0.9_E1_B10_lr0.05_num_clients100_cf0.1.npy')
data_acc[2] = np.load(r'./checkpoints/added/adam45feature_lstm_an15_sn3_D5_al0.9_E1_B50_lr0.05_num_clients100_cf0.1.npy')
data_acc[3] = np.load(r'./checkpoints/added/adam45feature_lstm_an15_sn3_D5_al0.9_E5_B10_lr0.05_num_clients100_cf0.1.npy')
data_acc[4] = np.load(r'./checkpoints/added/adam45feature_lstm_an15_sn3_D5_al0.9_E5_B50_lr0.05_num_clients100_cf0.1.npy')
data_acc[5] = np.load(r'./checkpoints/added/adam45feature_lstm_an15_sn3_D5_al0.9_E20_B10_lr0.05_num_clients100_cf0.1.npy')
data_acc[6] = np.load(r'./checkpoints/added/adam45feature_lstm_an15_sn3_D5_al0.9_E20_B50_lr0.05_num_clients100_cf0.1.npy')
data_acc[7] = np.load(r'./checkpoints/added/adam45feature_lstm_an15_sn3_D5_al0.9_E5_B50_lr0.05_num_clients100_cf0.npy')
data_acc[8] = np.load(r'./checkpoints/added/adam45feature_lstm_an15_sn3_D5_al0.9_E5_B50_lr0.05_num_clients100_cf0.1.npy')
data_acc[9] = np.load(r'./checkpoints/added/adam45feature_lstm_an15_sn3_D5_al0.9_E5_B50_lr0.05_num_clients100_cf0.5.npy')
data_acc[10] = np.load(r'./checkpoints/added/adam45feature_lstm_an15_sn3_D5_al0.9_E5_B50_lr0.05_num_clients100_cf1.npy')


# data_acc[7] = np.load(r'./checkpoints/added/cfadam4515feature_mlp_an15_sn3_D125_al1_E5_B1250_lr0.05_num_clients100_cf0.0.npy')
# data_acc[8] = np.load(r'./checkpoints/added/cfadam4515feature_mlp_an15_sn3_D125_al1_E5_B1250_lr0.05_num_clients100_cf0.1.npy')
# data_acc[9] = np.load(r'./checkpoints/added/cfadam4515feature_mlp_an15_sn3_D125_al1_E5_B1250_lr0.05_num_clients100_cf0.5.npy')
# data_acc[10] = np.load(r'./checkpoints/added/cfadam4515feature_mlp_an15_sn3_D125_al1_E5_B1250_lr0.05_num_clients100_cf1.0.npy')
#
#
# data_acc[0] = np.load(r'./checkpoints/added/adam45feature_lstm_an15_sn3_D5_al1_E1_Ball_lr0.05_num_clients100_cf0.1.npy')
# data_acc[1] = np.load(r'./checkpoints/added/adam45feature_lstm_an15_sn3_D5_al1_E1_B10_lr0.05_num_clients100_cf0.1.npy')
# data_acc[2] = np.load(r'./checkpoints/added/adam45feature_lstm_an15_sn3_D5_al1_E1_B50_lr0.05_num_clients100_cf0.1.npy')
# data_acc[3] = np.load(r'./checkpoints/added/adam45feature_lstm_an15_sn3_D5_al1_E5_B10_lr0.05_num_clients100_cf0.1.npy')
# data_acc[4] = np.load(r'./checkpoints/added/adam45feature_lstm_an15_sn3_D5_al1_E5_B50_lr0.05_num_clients100_cf0.1.npy')
# data_acc[5] = np.load(r'./checkpoints/added/adam45feature_lstm_an15_sn3_D5_al1_E20_B10_lr0.05_num_clients100_cf0.1.npy')
# data_acc[6] = np.load(r'./checkpoints/added/adam45feature_lstm_an15_sn3_D5_al1_E20_B50_lr0.05_num_clients100_cf0.1.npy')
# data_acc[7] = np.load(r'./checkpoints/added/adam45feature_lstm_an15_sn3_D5_al1_E5_B50_lr0.05_num_clients100_cf0.npy')
# data_acc[8] = np.load(r'./checkpoints/added/adam45feature_lstm_an15_sn3_D5_al1_E5_B50_lr0.05_num_clients100_cf0.1.npy')
# data_acc[9] = np.load(r'./checkpoints/added/adam45feature_lstm_an15_sn3_D5_al1_E5_B50_lr0.05_num_clients100_cf0.5.npy')
# data_acc[10] = np.load(r'./checkpoints/added/adam45feature_lstm_an15_sn3_D5_al1_E5_B50_lr0.05_num_clients100_cf1.npy')

label_name[0] = 'B=∞ E=1'
label_name[1] = 'B=10 E=1'
label_name[2] = 'B=50 E=1'
label_name[3] = 'B=10 E=5'
label_name[4] = 'B=50 E=5'
label_name[5] = 'B=50 E=10'
label_name[6] = 'B=50 E=20'
label_name[7] = 'cf=0'
label_name[8] = 'cf=0.1'
label_name[9] = 'cf=0.5'
label_name[10] = 'cf=1'
x = [0]+[25*(i+1) for i in range(1000//25)]
plt.figure(figsize=(10, 8))
plt.xlim(0,1000)
plt.ylim(0, 1)
for i in range(7):
    # for j in range(999):
    #     if data_acc[i][:,0][j+1]<data_acc[i][:,0][j]:
    #         data_acc[i][:, 0][j + 1] = data_acc[i][:,0][j]
    data = [0]+[data_acc[i][:,0][25*(k+1)-1] for k in range(1000//25) ]
    plt.plot(x, data, linestyle='-',  linewidth=1.5, label=label_name[i])
    # plt.plot(data_acc[i][:,0], linestyle='-',  linewidth=1.5, label=label_name[i])

# for i in range(4):
#     data = [0] + [data_acc[i+7][:, 0][25 * (k + 1) - 1] for k in range(1000 // 25)]
#     plt.plot(x, data, linestyle='-', linewidth=1.5, label=label_name[i+7])
plt.axhline(0.95,color='gray')
# # trans = transforms.blended_transform_factory(plt.get_yticklabels()[0].get_transform(), plt.transData)
plt.text(-10, 0.95, "0.95", ha="right", va="center")
plt.legend(loc=4)
plt.savefig('lstm_fedrs_miu.pdf')
plt.show()

# plt.subplot(1,3,1)
# # plt.ylim(0.96, 1)
# data_acc[7] = np.load(r'./checkpoints/added/adam45feature_lstm_an15_sn3_D5_al0.9_E5_B50_lr0.05_num_clients100_cf0.npy')
# data_acc[8] = np.load(r'./checkpoints/added/adam45feature_lstm_an15_sn3_D5_al0.9_E5_B50_lr0.05_num_clients100_cf0.1.npy')
# data_acc[9] = np.load(r'./checkpoints/added/adam45feature_lstm_an15_sn3_D5_al0.9_E5_B50_lr0.05_num_clients100_cf0.5.npy')
# data_acc[10] = np.load(r'./checkpoints/added/adam45feature_lstm_an15_sn3_D5_al0.9_E5_B50_lr0.05_num_clients100_cf1.npy')
# print('%.3f' % np.mean(data_acc[8][:,0][-50:]))
# for i in range(4):
#     data = [0] + [data_acc[i+7][:, 0][25 * (k + 1) - 1] for k in range(1000 // 25)]
#     plt.plot(x, data, linestyle='-', linewidth=1.5, label=label_name[i+7])
# plt.title('LSTM-FedRS')
# plt.legend()
# plt.subplot(1,3,2)
# # plt.ylim(0.96, 1)
# data_acc[7] = np.load(r'./checkpoints/added/adam45feature_lstm_an15_sn3_D5_al1_E5_B50_lr0.05_num_clients100_cf0.npy')
# data_acc[8] = np.load(r'./checkpoints/added/adam45feature_lstm_an15_sn3_D5_al1_E5_B50_lr0.05_num_clients100_cf0.1.npy')
# data_acc[9] = np.load(r'./checkpoints/added/adam45feature_lstm_an15_sn3_D5_al1_E5_B50_lr0.05_num_clients100_cf0.5.npy')
# data_acc[10] = np.load(r'./checkpoints/added/adam45feature_lstm_an15_sn3_D5_al1_E5_B50_lr0.05_num_clients100_cf1.npy')
# print('%.3f' % np.mean(data_acc[8][:,0][-50:]))
# for i in range(4):
#     data = [0] + [data_acc[i+7][:, 0][25 * (k + 1) - 1] for k in range(1000 // 25)]
#     plt.plot(x, data, linestyle='-', linewidth=1.5, label=label_name[i+7])
# plt.title('LSTM-FedAvg')
# plt.legend()
# plt.subplot(1,3,3)
# # plt.ylim(0.95, 1)
# data_acc[7] = np.load(r'./checkpoints/added/cfadam4515feature_mlp_an15_sn3_D125_al1_E5_B1250_lr0.05_num_clients100_cf0.0.npy')
# data_acc[8] = np.load(r'./checkpoints/added/cfadam4515feature_mlp_an15_sn3_D125_al1_E5_B1250_lr0.05_num_clients100_cf0.1.npy')
# data_acc[9] = np.load(r'./checkpoints/added/cfadam4515feature_mlp_an15_sn3_D125_al1_E5_B1250_lr0.05_num_clients100_cf0.5.npy')
# data_acc[10] = np.load(r'./checkpoints/added/cfadam4515feature_mlp_an15_sn3_D125_al1_E5_B1250_lr0.05_num_clients100_cf1.0.npy')
# print('%.3f' % np.mean(data_acc[8][:,0][-25:]))
# for i in range(4):
#     data = [0] + [data_acc[i+7][:, 0][25 * (k + 1) - 1] for k in range(1000 // 25)]
#     plt.plot(x, data, linestyle='-', linewidth=1.5, label=label_name[i+7])
# plt.title('MLP')
# plt.legend()
# plt.show()