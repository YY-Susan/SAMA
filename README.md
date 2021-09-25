# SAMA
Implementation of Paper：Self-Attentive Moving Average for Time Series Prediction
## Abstract
Time series prediction has been studied for decades due to its potential in a wide range of applications. As one of the most popular technical indicators, moving average summarizes the overall changing patterns over a past period and is frequently used to predict the future trend of time series. However, traditional moving average indicators are calculated by averaging the time series data with equal or predefined weights, and ignore the subtle difference in the importance of different time steps. Moreover, unchanged data weights will be applied across different time series, regardless of the differences in their inherent characteristics. In this paper, we propose a learning-based moving average indicator, called Self-Attentive Moving Average (SAMA). After encoding the input signals of time series based on recurrent neural networks, we introduce the self-attention mechanism to adaptively determine the data weights at different time steps for calculating the moving average. Furthermore, we use multiple self-attention heads to model SAMA indicators of different scales, and finally combine them for time series prediction. Extensive experiments on two real-world datasets demonstrate the effectiveness of our approach.
## Dataset
### stock
The stock dataset contains 1,026 stocks collected from the NASDAQ market, with trading records consisting of the opening price, closing price, highest price, lowest price and trading volume of each day between 1/2/2013 and 8/12/2017.
### air quality
The air quality dataset contains the statistics of PM<sub>2.5</sub>, PM<sub>10</sub>, SO<sub>2</sub>, CO, NO<sub>2</sub>, O<sub>3</sub>, and AQI per day from 12/2/2013 to 10/31/2018(https://download.csdn.net/download/godspeedch/10627195?utm\%20source=iteye\%20new).
## Models
/Net/Models_SAMA.py: end-to-end prediction framework;  
/training/train_SAMA5.py: the 5-step SAMA indicator;  
/training/train_SAMA20.py: the 20-step SAMA indicator;  
/training/train_SAMA60.py: the 60-step SAMA indicator;
## Requirements
Python >= 3.7  
numpy  
## Hyperparameter Settings
Epoch: 300  
BatchSize: 8  
Learning Rate: 1e-3  
Dropout: 0.1
## Contact
If you have any questions, please contact <suyaxi0301@163.com>.
