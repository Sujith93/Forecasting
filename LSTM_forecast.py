#https://medium.com/@cdabakoglu/time-series-forecasting-arima-lstm-prophet-with-python-e73a750a9887
#https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/

import pandas as pd
import numpy as np

df = pd.read_excel('data_Test_patterns/V1_Test Data for ML Model Testing_3-Dec-19.xlsx',
                   sheet_name = 'Z-Axis Current WON')
df.dtypes
df.columns


#train and test
train = df[df['Date'] < '2019-11-01']
test = df[df['Date'] >= '2019-11-01']


def lstm_forecast(train, test, colname,sheet = 'Z-Axis Current WON'):
    colname = 'Step-Up'
    
    #    train_col = train[['Complex-1']]
#    test_col = test[['Complex-1']]
    train_col = train[[colname]]
    test_col = test[[colname]]
    
    
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(train_col)
    scaled_train_data = scaler.transform(train_col)
    scaled_test_data = scaler.transform(test_col)
    
    
    from keras.preprocessing.sequence import TimeseriesGenerator
    n_input = 12
    n_features= 1
    generator = TimeseriesGenerator(scaled_train_data, scaled_train_data, length=n_input, batch_size=1)
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Bidirectional
    from keras.layers import LSTM
    
    lstm_model = Sequential()
    lstm_model.add(Bidirectional(LSTM(500, activation='relu'), input_shape=(n_input, n_features)))
#    ,return_sequences=True))
#    lstm_model.add(Dropout(0.2))
#    lstm_model.add(LSTM(200, activation='relu'))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer='adam', loss='mape')
    
    
#    lstm_model.summary()
    lstm_model.fit_generator(generator,epochs=30,steps_per_epoch=1)
    
    import matplotlib.pyplot as plt
    losses_lstm = lstm_model.history.history['loss']
    plt.figure(figsize=(12,4))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xticks(np.arange(0,30,1))
    plt.plot(range(len(losses_lstm)),losses_lstm);
    
    
    lstm_predictions_scaled = list()
    
    batch = scaled_train_data[-n_input:]
    current_batch = batch.reshape((1, n_input, n_features))
    
    for i in range(len(test_col)):   
        lstm_pred = lstm_model.predict(current_batch)[0]
        lstm_predictions_scaled.append(lstm_pred) 
        current_batch = np.append(current_batch[:,1:,:],[[lstm_pred]],axis=1)
        
    lstm_predictions_scaled
    lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)
    lstm_predictions
    forecast = pd.DataFrame({"forecast" :lstm_predictions.reshape(29,)})
    
    test_col['forecast'] = lstm_predictions
    #ploting
    fori = pd.concat([train_col,test_col],axis=1)
    fori.index = df['Date'].rename('dates')
    fori.columns = ['train','test','forecast']
    
    pd.plotting.register_matplotlib_converters()
#    fori.fillna(0,inplace = True)
    ax = fori.plot.line()
    fig = ax.get_figure()
    fig.savefig('data_Test_patterns/plots_lstm/'+ sheet +'/ '+ sheet +'_'+colname+'.pdf',bbox_inches='tight')

    test_col = test_col.reset_index(drop=True)
    ## metrics
    from statsmodels.tools.eval_measures import rmse
    from sklearn.metrics import mean_squared_error,mean_absolute_error
    predictions = forecast.copy()
#    rms_error = rmse(test_col[['Complex-1']],predictions)[0]
#    mae = mean_absolute_error(test_col['Complex-1'],predictions)
#    mse = mean_squared_error(test_col['Complex-1'],predictions)
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#    mape = mean_absolute_percentage_error(test_col[['Complex-1']],predictions)
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
df_matrics.to_csv('data_Test_patterns/plots_lstm/Z-Axis Current WON/Z-Axis_Current_WON.csv',index = False)



