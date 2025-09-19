# NYC Trip Duraion
<p>
  <img src='https://github.com/ahmeddiab1234/NYC_Trip_Duration/blob/master/utils/digrams/readme_digrame.png'>
</p>

<h3>

  This NYC-trip duration project inspired by [Kaggle competition](https://www.kaggle.com/competitions/nyc-taxi-trip-duration) aims to
predict the duration for trips in New York by given 10 features and
1Millioon example for train and 200000 for val after doing feature
engineering and feature extraction and use model like linear
regression, ridge, neural network, xgboost we end up with best model
from xgboost regressor with f1-score: 76% for val & 72% for test

</h3>


### Features
***id*** - a unique identifier for each trip
  
***vendor_id*** - a code indicating the provider associated with the trip record  

***dropoff_datetime*** - date and time when the meter was disengaged  

***passenger_count*** - the number of passengers in the vehicle (driver entered value)  

***pickup_longitude*** - the longitude where the meter was engaged  

***pickup_latitude*** - the latitude where the meter was engaged  

***dropoff_longitude*** - the longitude where the meter was disengaged  

***dropoff_latitude*** - the latitude where the meter was disengaged  

***store_and_fwd_flag*** - This flag indicates whether the trip record was held in vehicle memory before sending to the vendor because the vehicle did not have a connection to the server - Y=store and forward; N=not a store and forward trip  

***trip_duration*** - duration of the trip in seconds  




### From EDA
- Original train_val contain 10 features
- drop `id` uselss feature 
- data are clean and just 6 dublicate examples 
- target feature
    - before remove outlier, very high skewness, upnormal plot
    - after remove outlier, the data in normal range, and almost guassian plot
- most of the passengers are 1 and 2 but there high outlier like 7 and 8
- we calculate haversine distance from longitude&latitude and got very good correlation with target feature 
- from datetime feature varation
    - Friday and Saturday is the most days that there are trips counts
    - most of the trips are after hour 18 (may be because this is after the working time)
    - other variations are not so important


- the top 8 features with target are [`haversine_distance`,`dropoff_longitude`,  `pickup_longitude`,`pickup_datetime`,`month`,`dropoff_latitude`,  `pickup_latitude`,`season_encoder`]

```
├── logs
│ ├── model_results_LinearRegression.txt
│ ├── model_results_Neural_Network.txt
│ ├── model_results_Ridge.txt
│ └── model_results_XGboost.txt
├── outputs
├── Processing
├── split
│ ├── test.csv
│ ├── train_val.csv
│ ├── train.csv
│ └── val.csv
├── split_sample
│ ├── test.csv
│ ├── train.csv
│ └── val.csv
├── utils
│ ├── pycache
│ ├── digrams
│ ├── init.py
│ └── helper_fun.py
├── val_pkl
├── venv
├── init.py
├── .gitignore
├── config.yaml
├── EDA.ipynb
├── Evaluation.py
├── fine_tune.py
├── report.docx
├── report.pdf
├── requirments.txt
├── split_sample-20250715165954Z-1-001.zip
├── split-20250715165953Z-1-001.zip
├── test.py
├── val.py
└── XGboost.pkl

```

## Initialization & Setup

### 1- Clone Repository
```bash 
git clone https://github.com/your_username/NYC_Trip_Duration.git
cd NYC_Trip_Duration
```
### 2- Create Virtual Environment
```
python -m venv venv
```

Activate it:
Windows: ```venv\Scripts\activate```

Linux/Mac: ```source venv/bin/activate```

### 3- Install Requirments
```pip install -r requirements.txt```

### 4- Usage 

using random forest:
run training-validation:  ```python model.py --config config.yaml```
run validation: ```python val.py --config config.yaml```
run testing: ```python test.py --config config.yaml```

### Configurationo file

```
RANDOME_STATE: 42

dataset:
  train: 'split/train.csv'
  val: 'split/val.csv'
  train_val: 'split/train_val.csv'


preprocessing:
  drop_outlier: True
  apply_log: True
  calculate_haversine: True
  best_features: False

  polynomial:
    degree: 2
    include_bias: True
  
  scaling:
    option: 2


Model:
  model_name: XGboost

  LinearRegression:
    fit_intercept: True

  Ridge:
    alpha: 1
    fit_intercept: True
    solver: 'auto'
    positive: False

  NeuralNetwork:
    hidden_layers: [32, 8]
    solver: 'adam'
    init_lr: 0.001
    max_iter: 500
    early_stopping: True
    alpha: 0.1

  XGBoost:
    n_estimators: 300
    learning_rate: 0.1
    max_depth: 9
    min_child_weight: 3
    gamma: 0
    reg_alpha: 1
    reg_lambda: 0

```
- if you want to change the model just change model name in config file


### Metrices:
  We focus on r2-score and mean square error (mse)  
  
### Results:
| Metric            | R2-score (val) | MSE (val) |
|-------------------|----------------|-----------|
| Linear regression | 0.6            | 0.226     |
| Ridge             | 0.59           | 0.232     |
| Neural Network    | 0.586          | 0.234     |
| Xgboost           | 0.76           | 0.134     |


The best model we choose for testing XGBoost that give in val: r2-
score: 0.76% and MSE: 0.134
And for test: r2score: 0.725 and MSE: 0.174


