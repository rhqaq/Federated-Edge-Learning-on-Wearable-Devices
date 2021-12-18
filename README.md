# Federated Edge Learning with LSTM and FedRS for Wearable Devices
**Copyright (C) <2021-> by Mobile Systems and Networking Group, School of Computer Science, Fudan University**

In reality, a large number of users' wearable devices are mainly smart watches which can only collect limited user information features.So federation learning with MLP model is unable to get good enough performance in classification task. We use the classification model based on **LSTM and attention mechanism** to learn the temporal relationship in limited information and achieve better **Human Activity Recognition (HAR)**.
The federated learning for wearable device will face the **label distribution non IID** problem. In order to solve this problem, we applies **FedRS** which is a new federated learning algorithm. We found that in the federated learning task for wearable devices, when the main parameters of FedRS α higher can improve the model training effect, but α Lower will make local training difficult to converge.
**Edge servers** are used to reduce the local training pressure of wearable devices and reduce the communication cost of Federated learning.



## Requirements

+ Python
+ Pytorch



## Data

The dataset is built from a Daily and Sports Activities Data Set ([DSA](https://archive.ics.uci.edu/ml/datasets/Daily+and+Sports+Activities)), and comprises of **left arm** motion sensor data of 15 daily sports activities.



## System

<img src="https://github.com/rhqaq/Federated-Edge-Learning-on-Wearable-Devices/blob/main/figures/wearableFLSys-alls.png"  />
Each group consists of people with similar activity habits, only having examples of three activities.
Data of each group is uploaded to a trusted edge server to do efficient local training.
The model is download from a cloud parameter server before local training and aggregated by the cloud parameter server after local training.



## Model

The selected model is a **LSTM with Attention**, composed of one LSTM layer and one output layer, and one attention layer 
To solve label distribution non IID problem, we adopt FedRS to limit the update of missing classes’ weights during the local procedure
The baseline is **MLP**,composed of one input and one output layer, and one hidden layer with 1000 units using ReLU activations.



## Results

<img src="https://github.com/rhqaq/Federated-Edge-Learning-on-Wearable-Devices/blob/main/figures/all.png" alt="alt text" style="zoom:50%;" />
When the number of features is small and there are more activity categories to be predicted, the convergence speed of LSTM model in federated learning is significantly faster than MLP, and the number of communications required to achieve the same prediction accuracy is less
FedRS fetch α= 0.9 has better federal learning performance than FedAvg, and α= 0.5 will slow down the convergence, but reduce the fluctuation of performance.



## References

[1] Billur Barshan and Murat Cihan Yüksek. 2013. Recognizing Daily and Sports Activities in Two Open Source Machine Learning Environments Using Body-Worn Sensor Units. The Computer Journal 57, 11 (2013), 1649–1667. 

[2] Xin-Chun Li and De-Chuan Zhan. 2021. FedRS: Federated Learning with Restricted Softmax for Label Distribution Non-IID Data. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining.


