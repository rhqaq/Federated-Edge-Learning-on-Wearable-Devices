# Federated Edge Learning with LSTM and FedRS for Wearable Devices
**Copyright (C) <2021-> by Mobile Systems and Networking Group, School of Computer Science, Fudan University**

Nowadays, a large number of users’ wearable devices are smart watches, which can only collect limited user information features. Under this limitation, federation learning with MLP model is unable to achieve good performance in classification tasks. We design a classification model based on **LSTM and attention mechanism** to learn the temporal relationship in limited information and achieve better **Human Activity Recognition (HAR)** performance. To deal with the **non-IID distribution** problem of labels **indifferent group**, we involve the federated learning method **FedRS**. We found that in the federated learning task for wearable devices, when the main parameters of FedRS α is set a higher value,  the model can be trained to reach a better performance, but a smaller α will make local training difficult to converge. **Edge servers** are used to reduce the local training load for wearable devices and reduce the communication cost of different groups in the federated learning framework.


## Requirements

+ Python
+ Pytorch



## Data

The dataset is built from a Daily and Sports Activities Data Set ([DSA](https://archive.ics.uci.edu/ml/datasets/Daily+and+Sports+Activities)),  mainly the **left arm** motion sensor data of 15 daily sports activities.



## System

<img src="https://github.com/rhqaq/Federated-Edge-Learning-on-Wearable-Devices/blob/main/figures/wearableFLSys-alls.png"  />
In the federate learning framework, each group consists of people with similar activity habits, i.e., each group only has data examples of three activities. Data of each group is uploaded to a trusted edge server to realize efficient local training. The model is downloaded from the cloud-based parameter aggregator before local training and will also be aggregated by the cloud-based parameter aggregator after local training.



## Model

The designed model is composed of one **LSTM** layer, one **attention** layer, and one output layer. We adopt **FedRS** to limit the update of missing class weights during the local training. 
The baseline model is implemented by **MLP**, composed of one input layer and one output layer, also one hidden layer with 1000 units using ReLU activations.



## Results

<img src="https://github.com/rhqaq/Federated-Edge-Learning-on-Wearable-Devices/blob/main/figures/all.png" />
When we only have a few features and have to predict more activity categories, the convergence speed of LSTM model in federated learning is significantly faster than MLP, and the number of communications required to achieve the same prediction performance is less. When we set α= 0.9, FedRS will show better federal learning performance than FedAvg. 
Setting α= 0.5 will slow down the convergence, but the fluctuation of performance will be reduced. 



## References

[1] Billur Barshan and Murat Cihan Yüksek. 2013. Recognizing Daily and Sports Activities in Two Open Source Machine Learning Environments Using Body-Worn Sensor Units. The Computer Journal 57, 11 (2013), 1649–1667. 

[2] Xin-Chun Li and De-Chuan Zhan. 2021. FedRS: Federated Learning with Restricted Softmax for Label Distribution Non-IID Data. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining.


