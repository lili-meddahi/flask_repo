#####################
## ALL THE IMPORTS ##
#####################

import re
import pandas as pd
import pickle as pk
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


###################
## READ THE DATA ##
###################

# read in data
ds = pd.read_csv('used_car_dataset.csv')
# copy dataset
features = ['Brand', 'model', 'Year', 'kmDriven', 'Transmission', 'FuelType']
ds_copy = ds.copy()

ds_copy = ds_copy.drop("Age", axis=1)
ds_copy = ds_copy.drop("Owner", axis=1)
ds_copy = ds_copy.drop("PostedDate", axis=1)
ds_copy = ds_copy.drop("AdditionInfo", axis=1)
ds_copy['kmDriven'] = ds_copy['kmDriven']


####################
## CLEAN THE DATA ##
####################

# label encoder
label_encoder_map = {}

#convert strings to lowercase
ds_copy = ds_copy.map(lambda s:s.lower() if type(s) == str else s)
encoded_labels = ds_copy.copy()

# remove units
ds_copy['kmDriven'] = ds_copy['kmDriven'].astype(str)
ds_copy['AskPrice'] = ds_copy['AskPrice'].replace("₹ ", "")

columns_to_clean = ['kmDriven', 'AskPrice'] 
pattern = r'[^\w\s]'
# Use regex to remove special chars
for col in columns_to_clean:
    ds_copy[col] = ds_copy[col].apply(lambda x: re.sub(pattern, '', x))
    ds_copy[col] = ds_copy[col].str.replace('km', '')
    print(ds_copy.head)
    ds_copy[col] = pd.to_numeric(ds_copy[col], errors='coerce')

# make kph --> mph
ds_copy['kmDriven'] = ds_copy['kmDriven']/1.609

# change everything to numerical values
for feat in features:
    categories = ds_copy[feat].astype('category').cat.categories
    feat_map = {category: code for code, category in enumerate(categories)}
    
    label_encoder_map[feat] = feat_map
    encoded_labels[feat] = ds_copy[feat].map(feat_map)

# fill in nans
ds_encoded = encoded_labels.dropna()

# PICKLE! 
pk.dump(label_encoder_map, open("label_encoder_map.pkl", "wb"))

####################
## GENERATE MODEL ##
####################

# test/train split
X = ds_encoded.loc[:, ds_encoded.columns != 'AskPrice']
y = ds_encoded['AskPrice'].str.slice(2)
y = y.str.replace(",", "")

print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

X_train_norm = preprocessing.normalize(X_train)
X_test_norm = preprocessing.normalize(X_test)

# decision tree regression
regr = DecisionTreeRegressor(random_state=5)
regr = regr.fit(X_train, y_train)

# # kmeans
# regr = KMeans(n_clusters = 3, random_state = 0, n_init='auto')
# regr = regr.fit(X_train_norm, y_train)

# # random forest
# regr = RandomForestRegressor(random_state=42, n_estimators=100)
# regr = regr.fit(X_train_norm, y_train)

y_pred = regr.predict(X_test_norm)

# error eval
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("R2:", r2)

# pickle!!
with open("dec_tree_regr.pkl", "wb") as file:
    pk.dump(regr, file) 