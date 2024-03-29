# CloudyFL: A Cloudlet-Based Federated Learning Framework for Sensing User Behavior Using Wearable Devices
**Copyright (C) <2021-> by Mobile Systems and Networking Group, School of Computer Science, Fudan University**

Nowadays, a large number of users are using wearable devices, such as smartwatches, which can only be used to collect limited user information features. Under this limitation, federation learning with MLP model is unable to achieve good performance in classification tasks. To tackle this problem, we design a classification model based on **LSTM and attention mechanism** to learn the temporal relationship in limited information and achieve a better **Human Activity Recognition (HAR)** performance. To deal with the **non-IID distribution** problem of labels **indifferent group**, we involve the federated learning method **FedRS**. We found that in the federated learning task for wearable devices, when the main parameters of FedRS α is configured as a higher value, the model can be trained to reach a better performance. Differently, a smaller α will make local training difficult to converge. **Cloudlets** are used to reduce the local training load for wearable devices, and thus save the battery life of them. The use of cloudlets could also reduce the communication cost of different groups in the federated learning framework. 



## Requirements

+ Python
+ PyTorch



## Data

The first dataset is Daily and Sports Activities Data Set ([DSA](https://archive.ics.uci.edu/ml/datasets/Daily+and+Sports+Activities)),  mainly the motion sensor data of 19 daily human activity traces.
The second dataset is Physical Activity Monitoring for Aging People ([PAMAP](http://www.pamap.org/index.html)), containing signals of 8 participants performing 14 activities.



## System

<img src="https://github.com/rhqaq/Federated-Edge-Learning-on-Wearable-Devices/blob/main/figures/system.png"  />
In the design of CloudyFL, we divide the participants into different trusted zones. Each trusted zone is an organization which is trusted by the participating users and their devices within a coverage area. For each trusted zone, a cloudlet is responsible for executing the human activity recognition training tasks for the FL model and is trustworthy for all wearable devices within the trusted zone. In other words, the isolated trusted zones work collaboratively to build a global FL model. This model benefits from the sensed data from different trusted zones, without the need for retrieving the data out of each trusted zone. To reduce the communication cost, the trained parameters in each trusted zone are collected by the cloudlet and are then aggregated in the aggregator to update the global model. This two-stage process will iterate until reaching a convergence. 
In particular, CloudyFL could leverage deep learning models designed to solve different application problems, such as predicting mobile phone user's activities, 
recognizing pedestrian behaviors for autopilot, and health management. 


## Model

<img src="https://github.com/rhqaq/Federated-Edge-Learning-on-Wearable-Devices/blob/main/figures/wearableFLSys-alls.png"  />

The designed model **Att-LSTM** is composed of one **LSTM** layer, one **attention** layer, and one output layer. We adopt **FedRS** to limit the update of missing class weights during the local training. 
The baseline models are implemented by **MLP**, composed of one input layer and one output layer, also one hidden layer with 1000 units using ReLU activations and linear SVM.



## Results

<img src="https://github.com/rhqaq/Federated-Edge-Learning-on-Wearable-Devices/blob/main/figures/result.jpg" />
In our experiment, we develop four experimentation setups for the evaluation of the CloudyFL architecture. Considering the architecture of federated learning that the wearable devices are distributed to different cloudlets, we compare the performance of our system design in four setups. Based on PAMAP dataset and the 19 types of human activities in DSA dataset, we define the following four setups. 
In Scenario-10, Scenario-15 and Scenario-19, the number of iteration is 1000, while in Pamap-14 it is 2000. 
The average performance of the trained model in the last 25 iterations are shown in Table 3. Notably, when using the (Att-LSTM)-FedRS design in the CloudyFL architecture, the human activity recognition reaches the best performance for all setups.

## Resources

  * **Paper:** ["CloudyFL: A Cloudlet-Based Federated Learning Framework for Sensing User Behavior Using Wearable Devices"](https://www.researchgate.net/profile/Yang-Chen-125/publication/361025351_CloudyFL_A_Cloudlet-Based_Federated_Learning_Framework_for_Sensing_User_Behavior_Using_Wearable_Devices_ACM_Reference_Format/links/6298927b416ec50bdb04c440/CloudyFL-A-Cloudlet-Based-Federated-Learning-Framework-for-Sensing-User-Behavior-Using-Wearable-Devices-ACM-Reference-Format.pdf)
  ### Reference
```
@article{gong2022cloudyfl,
  title={CloudyFL: A Cloudlet-Based Federated Learning Framework for Sensing User Behavior Using Wearable Devices},
  author={Gong, Qingyuan and Ruan, Hui and Chen, Yang and Su, Xiang},
  booktitle = {Proceedings of the 6th International Workshop on Embedded and Mobile Deep Learning},
  year={2022}
}
```
## References

[1] Billur Barshan and Murat Cihan Yüksek. 2013. Recognizing Daily and Sports Activities in Two Open Source Machine Learning Environments Using Body-Worn Sensor Units. The Computer Journal 57, 11 (2013), 1649–1667. 

[2] Xin-Chun Li and De-Chuan Zhan. 2021. FedRS: Federated Learning with Restricted Softmax for Label Distribution Non-IID Data. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining.

[3] Attila Reiss and Didier Stricker. 2011. Towards global aerobic activ- ity monitoring. In Proceedings of the 4th International Conference on PErvasive Technologies. ACM, 12. 
