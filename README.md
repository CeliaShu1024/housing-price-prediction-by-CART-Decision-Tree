# housing-price-prediction-by-CART-Decision-Tree
This is my Principles of Artificial Intelligence final project. I applied `Python@3.9` + `LaTex` to create the project and report.</br>
The copyright of this repository belongs to me, whose github UID = `CeliaShu1024`.
# Content of Repository
`Report.pdf`: The project report which contains the theoretical demonstration of decision tree, algorithms and pre-processing progress.</br>
`cal_predict.py`: Project source code.</br>
`housing.csv`: Dataset.

# Configuration
Programming IDE: `Visual Studio Code`</br>
Programming Language: `Python@3.9.0` and `LaTex`</br>
Libraries: `pandas`, `numpy`, `scikit-learn`

# Dataset Info
`fetch_California_housing` dataset in `scikit-learn` (`sklearn` for short) library was used in this project.</br>
This dataset contains information about randomly selected used and new construction and agricultural equipment sold in the state of California in the United States (sales prices are in US dollars). It is a medium dataset with 20640 cases. </br>
This sample dataset contains 10 attributes, the first 8 and 10th of which are used as feature inputs to predict California housing prices, and the 9th attribute is used as the target to be predicted.</br>

# Data Pre-Processing
The process of this stage is demonstrated in `Section@2.6` of `Report.pdf`.</br>
In this stage, I applied measures of pre-processing. First, split the target column from dataset. Second, translate `object` type column into `float` type. Third, substitude missing values by filling median value of the attribute. Fourth, normalize these data by `Min-Max Scaler`. Last, split `train-set` and `test-set` with the ratio of 5:1.

# Parameter Tuning
In this stage, I selected 6 embedded parameters of decision tree in `sklearn` library. They are elaborately introduced in `Section@3.1` of `Report.pdf`.</br>
The priority of parameters to be tuned is: `max_depth` > `min_samples_leaf` > `min_samples_split` > `subsample` > `learning_rate` = `n_estimators`. The process of parameter tuning is also demonstrated in `Section@3.1` of `Report.pdf`.

# Sample Output
I added 5 groups of sample output in `Section@3.2` of `Report.pdf`.
