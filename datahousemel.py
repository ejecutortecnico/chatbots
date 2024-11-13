import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeRegressor

melbourne_file_path = 'melb_data.csv'
# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path) 

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]   #DataFrame with only features. 
y = melbourne_data.Price    # Prediction Taget using Dot notation
# Model - Decision Tree

melbourne_model = DecisionTreeRegressor(random_state = 1)  # Random state puts a number to a certain output. 

# Fit model 
melbourne_model.fit(X,y)

# print a summary of the data in Melbourne data
print("Making predictions for the following 5 houses: ")
print("\r")
print(X.head())
print("\r")
print("The predictions are: ")
print(melbourne_model.predict(X.head()))