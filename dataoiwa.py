import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

#Creating variable for the file path
iowa_file_path = "train.csv"

#Reading .csv file and assigning it to a dataframe
home_data = pd.read_csv(iowa_file_path)

# Create target and features
print(home_data.columns)
print("\n")
y = home_data.SalePrice   # Prediction Target

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = home_data[features]   # Feature Dataframe

## Now we split the dataset into training and validation data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)  # train is the training data and val is the validation data

# Specify the model 
iowa_model = DecisionTreeRegressor(random_state = 1)

# Fit the model 
iowa_model.fit(train_X, train_y)

# Make predictions and calculate mean absolute error. 
val_preds = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_preds, val_y)
print("Validations MAE: {:,.0f}".format(val_mae))

# We will write a custom mae function that takes in training and validation data and number of nodes in the decision tree model which is usually supplied
# in a list. 

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state = 0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae

# Now we will be given or we will take a list of nodes which we will use to calculate the mae and then compare the different results to select the best 
# node that gives the least possible mae. So for that we will be writing a for loop to run through all the values one by one. 

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]    # given value

for max_leaf_nodes in candidate_max_leaf_nodes:
    mean_abs_err = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d \t\t Mean Absolute Error: %d" %(max_leaf_nodes, mean_abs_err))