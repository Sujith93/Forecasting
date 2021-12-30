import pandas as pd
import numpy as np

df = pd.read_excel('data_Test_patterns/V1_Test Data for ML Model Testing_3-Dec-19.xlsx',
                   sheet_name = 'Z-Axis Current WON')
df.dtypes
df.columns


#train and test
train = df[df['Date'] < '2019-11-01']
test = df[df['Date'] >= '2019-11-01']

#train_col = train[['Step-Up']]
#test_col = test[['Step-Up']]

# univariate cnn lstm example
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D


# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

    
def lstm_forecast(train, test, colname,sheet = 'Z-Axis Current WON'):
    train_col = train[[colname]]
    test_col = test[[colname]]
    # define input sequence
    raw_seq = train_col[colname].to_list()
    # choose a number of time steps
    n_step = 12
    # split into samples
    X, y = split_sequence(raw_seq, n_step)
    # reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
    n_features = 1
    n_seq = 4
    n_steps = 3
    X = X.reshape((X.shape[0], n_seq, n_steps, n_features))
    # define model
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, n_steps, n_features)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    model.fit(X, y, epochs=500, verbose=0)
    # demonstrate prediction
    x_input = array(raw_seq[-n_step:])
    x_input = x_input.reshape((1, n_seq, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    print(yhat)
    
    lstm_predictions = list()
    for i in range(len(test)):
        batch = array(raw_seq[-n_step:])
        current_batch = batch.reshape((1, n_seq, n_steps, n_features))
        lstm_pred = model.predict(current_batch)[0]
        lstm_predictions.append(lstm_pred[0])
        raw_seq = np.append(raw_seq, lstm_pred)
        
    
    forecast = pd.DataFrame({"forecast" : lstm_predictions})
    test_col['forecast'] = lstm_predictions
    #ploting
    fori = pd.concat([train_col,test_col],axis=1)
    fori.index = df['Date'].rename('dates')
    fori.columns = ['train','test','forecast']
    
    pd.plotting.register_matplotlib_converters()
    ax = fori.plot.line()
    fig = ax.get_figure()
    fig.savefig('data_Test_patterns/plots_convlstm/'+ sheet +'/ '+ sheet +'_'+colname+'.pdf',bbox_inches='tight')

    test_col = test_col.reset_index(drop=True)
    ## metrics
    from statsmodels.tools.eval_measures import rmse
    from sklearn.metrics import mean_squared_error,mean_absolute_error
    predictions = forecast.copy()
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    rms_error = rmse(test_col[[colname]],predictions)[0]
    mae = mean_absolute_error(test_col[colname],predictions)
    mse = mean_squared_error(test_col[colname],predictions)
    mape = mean_absolute_percentage_error(test_col[[colname]],predictions)
    return [colname,rms_error, mae, mse, mape]



lis = []
for i in df.columns.values[1:]:
    print(i)
    lis.append(lstm_forecast(train, test, colname = i,sheet = 'Z-Axis Current WON'))

sheet = 'Z-Axis Current WON'
df_matrics = pd.DataFrame(lis,columns = ['pattern','RMSE','MAE','MSE','MAPE'])
df_matrics.to_csv('data_Test_patterns/plots_convlstm/Z-Axis Current WON/Z-Axis_Current_WON.csv',index = False)

