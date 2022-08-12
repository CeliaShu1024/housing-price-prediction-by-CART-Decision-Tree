# GBDT Implementation
import numpy as np
import pandas as pd
from pkg_resources import get_build_platform
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import pipeline
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, scale
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# ignore system warnings
import warnings
warnings.filterwarnings('ignore')

# import dataset
# load dataset
cal = pd.read_csv(r'housing.csv', encoding='utf-8')
# # print information of dataset
# cal.info()

# data cleaning
# Separate numerical (continuous) type data
numerical_cols = list(cal.select_dtypes(include = 'float64').columns)
# Remove prediction column
numerical_cols.remove('median_house_value')
# Separate categorized (discrete) data 
cat_cols=list(cal.select_dtypes(include = 'object').columns)

# Split testing set and training set randomly and in 1:5 ratio
x_train, x_test, y_train, y_test = train_test_split(cal.drop('median_house_value', axis = 1)
, cal['median_house_value'], test_size=0.2, random_state=42)

# data pre-processing: substitude missing value with median value
num_process_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', MinMaxScaler())])
cat_process_pipe = Pipeline([('ohe', OneHotEncoder())])
preprocessing_pipe = ColumnTransformer([('num_cols2', num_process_pipe, numerical_cols), ('cat_cols', cat_process_pipe, cat_cols)])

# apply pre-processing
preprocessing_pipe.fit(x_train)

# CART modeling by scikit-learn
CART=Pipeline([('preprocessing', preprocessing_pipe), ('model', DecisionTreeRegressor(
    criterion='mse' # tree split strategy of regression
    # pruning parameters
    , max_depth = 13
    , min_samples_split=2   # minimum sample number of the regression tree when splitting
    , min_samples_leaf=1    # minimum sample number on the leaf node of the regression tree
    , max_features='log2' # maximum number of candidate features for selecting split features
    # in this case, max_features=log2(n_features), which decreases the squared error and increases deviation
    , random_state=None # No random seed is set
))])

# GBDT modeling by scikit-learn
GBDT=Pipeline([('preprocessing', preprocessing_pipe), ('model', GradientBoostingRegressor(
    loss='ls'   # least squared loss
    , learning_rate=0.1   # step length 
    , n_estimators=300  # iteration times
    , subsample= 0.8    # the smaller subsample, the stronger gradient boosting
    , criterion= 'mse'  # tree split strategy of regression
    
    # pruning parameters
    , min_samples_split=2   # minimum sample number of base learner tree when splitting
    , min_samples_leaf=1    # minimum sample number on the leaf node of the base learner tree
    , max_depth=13  # max depth of base learner

    , init=None # initialize loss estimator
    , random_state=None # No random seed is set
    , max_features='log2'   # maximum number of candidate features for selecting split features
    # in this case, max_features=log2(n_features), which decreases the squared error and increases deviation
    , verbose=1 # print learning process
    , warm_start=False))])

# set output target
tar = ['longitude','latitude','housing_median_age','total_rooms','total_bedrooms'
        ,'population','households','median_income','ocean_proximity','actual_median_house_value','predicted_median_house_value']

# apply CART model
CART.fit(x_train, y_train)
# make CART prediction
CART_y_train_prediction = CART.predict(x_train)
CART_y_test_prediction = CART.predict(x_test)
# convert prediction to attribute matrix
CART_train_prediction = np.column_stack((x_train, y_train, CART_y_train_prediction))
CART_test_prediction = np.column_stack((x_test, y_test, CART_y_test_prediction))
# transform the prediction results into dataframes
df_CART_p1 = pd.DataFrame(CART_train_prediction, columns=tar)
df_CART_p2 = pd.DataFrame(CART_test_prediction, columns=tar)
# fitting evaluation
CART_score1 = CART.score(x_train, y_train)
CART_score2 = CART.score(x_test, y_test)
# calculate mean squared error of training and testing process
CART_mse1 = np.sqrt(mean_squared_error(y_train, CART_y_train_prediction))
CART_mse2 = np.sqrt(mean_squared_error(y_test, CART_y_test_prediction))

# apply GBDT model
GBDT.fit(x_train, y_train)
# make GBDT prediction
GBDT_y_train_prediction = GBDT.predict(x_train)
GBDT_y_test_prediction = GBDT.predict(x_test)
# convert prediction results to attribute matrix
GBDT_train_prediction = np.column_stack((x_train, y_train, GBDT_y_train_prediction))
GBDT_test_prediction = np.column_stack((x_test, y_test, GBDT_y_test_prediction))
# transform the prediction results into dataframes
df_GBDT_p1 = pd.DataFrame(GBDT_train_prediction, columns=tar)
df_GBDT_p2 = pd.DataFrame(GBDT_test_prediction, columns=tar)
# fitting evaluation
GBDT_score1 = GBDT.score(x_train, y_train)
GBDT_score2 = GBDT.score(x_test, y_test)
# calculate mean squared error of training and testing process
GBDT_mse1 = np.sqrt(mean_squared_error(y_train, GBDT_y_train_prediction))
GBDT_mse2 = np.sqrt(mean_squared_error(y_test, GBDT_y_test_prediction))

# output CART prediction results
print("CART Training Prediction")
print(df_CART_p1)
print("CART Testing Prediction")
print(df_CART_p2)
# output GBDT prediction results
print("GBDT Training Prediction")
print(df_GBDT_p1)
print("GBDT Testing Prediction")
print(df_GBDT_p2)
# output results
print('CART Regression Evaluation:')
print('Fitting score of training set=%0.2f' % (CART_score1))
print('Fitting score of testingset=%0.2f' % (CART_score2))
print('Mean squared error of training=%0.2f' %(CART_mse1))
print('Mean squared error of testing=%0.2f' %(CART_mse2))
print('GBDT Regression Evaluation:')
print('Fitting score of training set=%0.2f' % (GBDT_score1))
print('Fitting score of testing set=%0.2f' % (GBDT_score2))
print('Mean squared error of training=%0.2f' %(GBDT_mse1))
print('Mean squared error of testing=%0.2f' %(GBDT_mse2))

# GBDT_hist_test = plt.hist(GBDT_y_test_prediction,bins=10)
# plt.show()