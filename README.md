# WiFi Fingerprinting Indoor Navigation MultiLabel Classifier System
**Wifi based Indoor Localization system utilizing Wi-Fi Fingerprinting Technique - AI/ML approach**

**Introduction:** Indoor localization refers to determining the location of an individual or an object inside a building or any indoor environment. With the increasing demand for location-based services, indoor localization has gained significant attention in recent years. WiFi fingerprinting is one of the widely used techniques for indoor localization. Generally, It involves measuring the signal strength of Wi-Fi access points (APs) to determine the location of a device.

**Issues of indoor localization:** Indoor positioning focuses on predicting the user’s location in constrained environments (e.g. office buildings, hospitals, train stations, shopping malls, etc.). Such an environment often contains multiple rooms, corridors and floors, and is crowded with furniture, walls and people. Therefore, the electromagnetic signals are usually blocked, attenuated and reflected when travelling in an indoor environment.

**Steps of wifi fingerprinting:** The use of WiFi fingerprinting has become a popular approach also for indoor localization due to its wide availability, low cost, and high accuracy. The process of Wi-Fi fingerprinting involves the following steps: 
Establish a database containing WiFi signals collected at every reference point in the targeted indoor environment. 
Each location in the targeted area has its own distinguishing WiFi fingerprint i.e., distinct WiFi signal pattern because the propagation of the WiFi signal is affected by the complex indoor environment.
Positioning systems take advantage of this feature of the WiFi fingerprint to accurately perform location estimation of the user.
Match the real-time WiFi signals received by the user with those in the database, so that positioning estimation of the user’s current location could be generated based on their relevance.

**Dataset Description**
For building prediction algorithms and performing analysis in the initial stages of this project a publicly available dataset UJIIndoorLoc is utilised. UJIIndoorLoc is a multi-building and multi-floor WLAN localization database. The UJIIndoorLoc database covers three buildings of Universitat Jaume I, Madrid Spain with 4 or more floors and almost 110000 m2. The full raw information is collected by more than 20 users and by means of 25 devices. The dataset consists of 19937 calibration/training/reference records and 1111 positioning/validation/test records.  

**Classification Approach**
The work is started with developing a classification model for building id and floor id simultaneous prediction. Classification is a supervised machine learning method where the model tries to predict the correct label of a given input data. There are two types of learners in classification: eager and lazy learners. Eager learners are machine learning algorithms that first build a model from the training dataset before making any prediction on future datasets. They spend more time during the training process and require less time to make predictions. Lazy learners or instance-based learners, on the other hand, do not create any model immediately from the training data. They  memorise the training data, and each time there is a need to make a prediction, they search for the nearest neighbour from the whole training data, which makes them  slow during prediction. Multi-Label K-Nearest Neighbors (ML-KNN) approach, which is a type of instance-based learner, is used to approach the classification problem. The model simultaneously predicts Building ID and Floor ID which reduces the compute power, memory and processing time. The classical and adaptive approaches are implemented. </br>

**@ Ing. Amber Tiwari (PhD)**
