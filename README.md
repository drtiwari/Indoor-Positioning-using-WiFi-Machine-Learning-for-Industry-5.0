# WiFi Fingerprinting Indoor Navigation MultiLabel Classifier System
**Wifi based Indoor Localization system utilizing Wi-Fi Fingerprinting Technique - AI/ML approach**

**Introduction:** Indoor localization refers to determining the location of an individual or an object inside a building or any indoor environment. With the increasing demand for location-based services, indoor localization has gained significant attention in recent years. WiFi fingerprinting is one of the widely used techniques for indoor localization. Generally, It involves measuring the signal strength of Wi-Fi access points (APs) to determine the location of a device.

**Issues of indoor localization:** Indoor positioning focuses on predicting the user’s location in constrained environments (e.g. office buildings, hospitals, train stations, shopping malls, etc.). Such an environment often contains multiple rooms, corridors and floors, and is crowded with furniture, walls and people. Therefore, the electromagnetic signals are usually blocked, attenuated and reflected when travelling in an indoor environment.

**Steps of wifi fingerprinting:** The use of WiFi fingerprinting has become a popular approach also for indoor localization due to its wide availability, low cost, and high accuracy. The process of Wi-Fi fingerprinting involves the following steps: 
Offline phase - Establish a database containing WiFi signals collected at every reference point in the targeted indoor environment. 
Online phase - Match the real-time WiFi signals received by the user with those in the database, so that positioning estimation of the user’s current location could be generated based on their relevance.

**Dataset Description**
For building prediction algorithms and performing analysis in the initial stages of this project a publicly available dataset UJIIndoorLoc is utilised. UJIIndoorLoc is a multi-building and multi-floor WLAN localization database. The UJIIndoorLoc database covers three buildings of Universitat Jaume I, Madrid Spain with 4 or more floors and almost 110000 m2. The full raw information is collected by more than 20 users and by means of 25 devices. The dataset consists of 19937 calibration/training/reference records and 1111 positioning/validation/test records.  

**Classification Approach**
A multi-label classification model for predicting Building ID and Floor ID simultaneously was prepared. The multi-label approach reduces the compute power, memory and processing time. </br>

**@ Ing. Amber Tiwari (PhD)**
