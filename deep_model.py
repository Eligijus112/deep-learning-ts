# Data wrangling
import pandas as pd
import numpy as np

# Deep learning: 
from keras.models import Sequential
from keras.layers import LSTM, Dense


class DeepModelTS():
    """
    A class to create a deep time series model
    """
    def __init__(
        self, 
        data: pd.DataFrame, 
        Y_var: str,
        lag: int,
        LSTM_layer_depth: int, 
        epochs=10, 
        batch_size=256
    ):

        self.data = data 
        self.Y_var = Y_var 
        self.lag = lag 
        self.LSTM_layer_depth = LSTM_layer_depth
        self.batch_size = batch_size
        self.epochs = epochs

    def create_data_for_NN(
        self,
        use_last_n=None
        ):
        """
        A method to create data for the neural network model
        """
        # Extracting the main variable we want to model/forecast
        y = self.data[self.Y_var].tolist()

        # Subseting the time series if needed
        if use_last_n is not None:
            y = y[-use_last_n:]

        # The X matrix will hold the lags of Y 
        X, Y = [], []

        if len(y) - self.lag <= 0:
            X.append(y)
        else:
            for i in range(len(y) - self.lag):
                Y.append(y[i + self.lag])
                X.append(y[i:(i + self.lag)])
        
        X = np.array(X)
        Y = np.array(Y)

        # Reshaping the X array to an LSTM input shape 
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        return X, Y

    def LSTModel(self):
        """
        A method to fit the LSTM model 
        """
        # Getting the data 
        X, Y = self.create_data_for_NN()

        # Defining the model
        model = Sequential()
        model.add(LSTM(self.LSTM_layer_depth, activation='relu', input_shape=(self.lag, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        # Fitting the model 
        model.fit(
            x=X,
            y=Y, 
            batch_size=self.batch_size,
            epochs=self.epochs
        )

        return model

    def predict(self, model, n_ahead):
        """
        A method to predict n steps ahead using the data which was used in 
        creating the DeepModelTS class
        """
        # Getting the last n time series 
        X, _ = self.create_data_for_NN(use_last_n=self.lag)        

        # Making the prediction list 
        yhat = []

        for _ in range(n_ahead):
            # Making the prediction
            fc = model.predict(X)
            yhat.append(fc)
            
            # Creating a new input matrix for forecasting
            X = np.append(X, fc)

            # Ommiting the first variable
            X = np.delete(X, 0)

            # Reshaping for the next iteration
            X = np.reshape(X, (1, len(X), 1))

        yhat = [y[0][0] for y in yhat]

        return yhat