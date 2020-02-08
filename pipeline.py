# Data wrangling
import pandas as pd 

# The deep learning class
from deep_model import DeepModelTS

# Reading the data 
d = pd.read_csv('input/DAYTON_hourly.csv')

# Initiating the class 
deep_learner = DeepModelTS(
    data=d, 
    Y_var='DAYTON_MW',
    lag=24,
    LSTM_layer_depth=100
)

# Fitting the model 
model = deep_learner.LSTModel()

# Making the prediction for the next 48 hours
yhat = deep_learner.predict(model, 48)