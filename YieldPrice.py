import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pickle as pickle
# Load the dataset again
data = pd.read_csv('C:/Users/jvr/Desktop/project/corrected_crop_data1.csv')

# Preparing the data for modeling
X = data[['Crop', 'Total Rainfall', 'Max. Temp', 'Min Temp','District']]
y = data['Total Yield']

# Preprocess the data by encoding categorical variables
categorical_features = ['Crop','District']
numerical_features = ['Total Rainfall', 'Max. Temp', 'Min Temp']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Defining the Gradient Boosting Regressor model
gbr_model = GradientBoostingRegressor(random_state=42)

# Creating a pipeline with preprocessing and the model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', gbr_model)
])

# Fitting the model to the training data
pipeline.fit(X_train, y_train)

# Predicting on the test set
y_pred = pipeline.predict(X_test)

# Calculating the Root Mean Square Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

rmse
pickle.dump(pipeline, open('YieldPrice.pkl','wb'))

model = pickle.load(open('YieldPrice.pkl','rb'))