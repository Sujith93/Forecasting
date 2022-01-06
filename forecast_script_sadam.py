#### import required libraries

"""
Python script for Schuler machine which has six different motors(X-axis, Z-axis,
LP1-axis, LP2-axis,C1-axis,C2-axis) to forecast for 1 month for different parameters viz
current, power, energy consumption.
"""

import os
import logging
import warnings
import base64

from datetime import datetime, timedelta
import pandas as pd

import psycopg2

from fbprophet import Prophet
from pmdarima import auto_arima

from dateutil.relativedelta import relativedelta

import boto3

warnings.filterwarnings("ignore")
logging.getLogger('fbprophet').setLevel(logging.WARNING)
logging.getLogger().setLevel(logging.ERROR)

os.environ['no_proxy'] = '*'

def db_connect(database, user, password, host, port):
    """Return connection
    Connecting to Postgre SQL with database, user, password, host, port
    """
    conn = psycopg2.connect(
        database=database,
        user=user,
        password=password,
        host=host,
        port=port
        )
    print("Opened database successfully")
    return conn


def key_decrypt():
    """Return decrypted string
    Decryption of the Encrypted string with Amazon KMS
    """
    session = boto3.session.Session()
    access_key = 'XXXX'
    secret_key = 'XXXX'
    kms = session.client('kms', region_name='us-east-2',
                         aws_access_key_id=access_key,
                         aws_secret_access_key=secret_key
    )
    encrypted_password = 'AQICAHjZWXubbPDmTbqB05OVmyA1TM+OEML1HX3c+xgbRQDUHgGDZoJY1cCBbeJbEuCtrzqqAAAAuTCBtgYJKoZIhvcNAQcGoIGoMIGlAgEAMIGfBgkqhkiG9w0BBwEwHgYJYIZIAWUDBAEuMBEEDOdy1BzpkFKu0IXxPAIBEIBywFO19XbUv/PbItVT/myNSC0Ej8iV9NOmFpllhymbmZen6wleidHsG+I5tjdobES1nqVgIL3Lhcx/7QJoJQXe2aS0K186siM4fEAqyxeL9Lm9mWJrqk2R4IiS/KXCXIAFypScFGqITFi4yGnlBojYkp2z'
    binary_data = base64.b64decode(encrypted_password)
    meta = kms.decrypt(CiphertextBlob=binary_data)
    plaintext = meta[u'Plaintext']
    return plaintext.decode()


def key_values(key):
    """Return string values of database, user, password, host, port
    Spliting the decrypted string to get the database credentials
    """
    key_split = key.split(':')
    user = key_split[0]
    password = key_split[2].split('@')[0]
    host = key_split[2].split('@')[1]
    port = key_split[3].split('/')[0]
    database = key_split[3].split('/')[1]
    return database, user, password, host, port


def data_fetch(partid):
    """Return dataframe which is one year from current date for that particular partid
    Connecting to DB, with date reference fetch the data and convert into dataframe
    """
    curent_day = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_date = datetime.strptime(str(curent_day), "%Y-%m-%d %H:%M:%S")
    twelve_months = current_date + relativedelta(months=-12)
    twelve_months_date = twelve_months.strftime("%Y-%m-%d %H:%M:%S")
    database, user, password, host, port = key_values(key_decrypt())
    conn = db_connect(database, user, password, host, port)
    cursor = conn.cursor()
    postgresql_select_query = '''SELECT * FROM parametersummarydaily WHERE \
                                    StartDateTime BETWEEN %s AND %s ;'''
    cursor.execute(postgresql_select_query, (twelve_months_date, curent_day, ))
    mobile_records = cursor.fetchall()
    data_frame = pd.DataFrame(columns=['Id', 'StartDateTime', 'EndDateTime', 'ParameterId',
                                       'ModuleId', 'PartId', 'SummaryTypeId',
                                       'FrequencyTypeId', 'SummaryValue', 'HighWarning',
                                       'HighCritical', 'MotorStatus', 'ActualPosition',
                                       'Torque', 'InsertedTimeStamp'])
    for i, row in enumerate(mobile_records):
        data_frame.at[i, 'Id'] = row[0]
        data_frame.at[i, 'StartDateTime'] = row[1]
        data_frame.at[i, 'EndDateTime'] = row[2]
        data_frame.at[i, 'ParameterId'] = row[3]
        data_frame.at[i, 'ModuleId'] = row[4]
        data_frame.at[i, 'PartId'] = row[5]
        data_frame.at[i, 'SummaryTypeId'] = row[6]
        data_frame.at[i, 'FrequencyTypeId'] = row[7]
        data_frame.at[i, 'SummaryValue'] = row[8]
        data_frame.at[i, 'HighWarning'] = row[9]
        data_frame.at[i, 'HighCritical'] = row[10]
        data_frame.at[i, 'MotorStatus'] = row[11]
        data_frame.at[i, 'ActualPosition'] = row[12]
        data_frame.at[i, 'Torque'] = row[13]
        data_frame.at[i, 'InsertedTimeStamp'] = row[14]
    data_frame['StartDateTime'] = pd.to_datetime(data_frame['StartDateTime'],
                                                 format='%Y-%m-%d %H:%M:%S')
    data_frame = data_frame[data_frame['PartId'] == partid]
    data_frame = data_frame.sort_values(by='StartDateTime')
    return data_frame


def forecast_model(df_part, model_n, forecast_for=31):
    """Return dataframe with 30 days of forecasted values from fbprophet and Arima models
    Building the models with fbprophet and arima and return the next 1 month of
    forecasted values
    """
    print('Model : ', model_n)
    ## Prophet model
    if model_n == 'prophet':
        df_part_pro = df_part.copy()
        df_part_pro.dropna(inplace=True)
        df_part_pro.columns = ['ds', 'y']
        df_part_pro['ds'] = df_part_pro['ds'].astype(str)
        model_prophet = Prophet(daily_seasonality=True)
        model_prophet.fit(df_part_pro)
        future = model_prophet.make_future_dataframe(periods=forecast_for)
        forecast = model_prophet.predict(future)
        predictions = forecast.iloc[len(df_part_pro):][['ds', 'yhat']]
        predictions = predictions.reset_index(drop=True)
        predictions.columns = ['StartDateTime', 'SummaryValue']
        return predictions
    elif model_n == 'arima':
    ## Arima model
        df_part_arm = df_part.copy()
        df_part_arm.dropna(inplace=True)
        df_part_arm.set_index('StartDateTime', inplace=True)
        df_part_arm.index = pd.to_datetime(df_part_arm.index)
        model_auto = auto_arima(df_part_arm, error_action='ignore',
                                suppress_warnings=True)
        predictions = model_auto.predict(n_periods=forecast_for)
        test_day = forecast_for
        li = df_part_arm.index.to_list()
        for i in range(1, test_day+1):
            li.append(df_part_arm.index[-1] + timedelta(days=i))
        # Obtain predicted values
        start = len(df_part_arm)
        predictions = pd.Series(predictions, index=li[start:])
        predictions = predictions.reset_index()
        predictions.columns = ['StartDateTime', 'SummaryValue']
        return predictions


def forecast_table(part_data, m_id, p_id, model_name):
    """Return forecast dataframe for that particular partid & parameterid
    Creating a dataframe with the forecasted values.
    """
    fore_table = pd.DataFrame()
    part_data = part_data[part_data['PartId'] == m_id]
    part_data = part_data[part_data['ParameterId'] == p_id]
    part_data = part_data[['StartDateTime', 'SummaryValue']]
    forecast = forecast_model(part_data, model_n=model_name)
    forecast['ParameterId'] = p_id
    forecast['PartId'] = m_id
    now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    forecast['InsertedTimeStamp'] = now
    forecast['EndDateime'] = forecast[['StartDateTime']] + pd.Timedelta(hours=23)
    fore_table = pd.concat([fore_table, forecast])
    return fore_table


def db_del():
    """Return None
    Connecting to DB and deleting the values which are after the current date
    from ParameterForecastData table
    """
    database, user, password, host, port = key_values(key_decrypt())
    conn = db_connect(database, user, password, host, port)
    cur_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur = conn.cursor()
    delete_sql = '''DELETE FROM ParameterForecastData WHERE StartDateTime > %s ;'''
    cur.execute(delete_sql, (cur_date,))
    conn.commit()


def db_idmax():
    """Return Maxvalue of the Id
    Connecting to DB and get the max id from ParameterForecastData table
    """
    try:
        database, user, password, host, port = key_values(key_decrypt())
        conn = db_connect(database, user, password, host, port)
        cur = conn.cursor()
        cur.execute("SELECT MAX(Id) FROM parameterforecastdata;")
        max_id = cur.fetchall()
        max_id = int(max_id[0][0])
    except:
        max_id = 0
    return max_id


def db_update(ft):
    """Return None
    Connecting to DB and inserting the forecasted values in the
    ParameterForecastData table
    """
    ft = ft[['Id', 'StartDateTime', 'EndDateime', 'ParameterId', 'PartId',
             'SummaryValue', 'InsertedTimeStamp']]
    ft['Id'] = ft['Id'].astype(str)
    ft['ParameterId'] = ft['ParameterId'].astype(str)
    ft['PartId'] = ft['PartId'].astype(str)
    ft['SummaryValue'] = ft['SummaryValue'].astype(str)
    ft['InsertedTimeStamp'] = pd.to_datetime(ft['InsertedTimeStamp'])
    database, user, password, host, port = key_values(key_decrypt())
    conn = db_connect(database, user, password, host, port)
    cursor = conn.cursor()
    postgres_insert_query = ''' INSERT INTO ParameterForecastData(Id, \
    StartDateTime, EndDateime,ParameterId,PartId,SummaryValue,InsertedTimeStamp) \
                                                VALUES (%s,%s,%s,%s,%s,%s,%s)'''
    for i in range(ft.shape[0]):
        cursor.execute(postgres_insert_query, (ft.iloc[i].to_list()[0],
                                               ft.iloc[i].to_list()[1],
                                               ft.iloc[i].to_list()[2],
                                               ft.iloc[i].to_list()[3],
                                               ft.iloc[i].to_list()[4],
                                               ft.iloc[i].to_list()[5],
                                               ft.iloc[i].to_list()[6]))
    conn.commit()


# variables declaration
# Motor
X_AXIS = 1
Z_AXIS = 2
C1_AXIS = 3
C2_AXIS = 4
LP1_AXIS = 5
LP2_AXIS = 6
MOTOR_ID = [X_AXIS, Z_AXIS, C1_AXIS, C2_AXIS, LP1_AXIS, LP2_AXIS]
MOTOR_NAME = ['X_AXIS', 'Z_AXIS', 'C1_AXIS', 'C2_AXIS', 'LP1_AXIS', 'LP2_AXIS']
# Parameters
CURRENT = 3
POWER = 4
ENERGY = 6
PARAMETER_ID = [CURRENT, POWER, ENERGY]
PARAMETER_NAME = ['Status', 'ActualPosition', 'MotorCurrent', 'MotorPower',
                  'MotorTorque', 'MotorLoadingPercentage', 'EnergyCOUNTER',
                  'DriveTemperature']

FORECAST_DAYS = 31

if __name__ == "__main__":
#Process flow:
#    1. Delete the data from ParameterForecastData table
#    2. Get the max id from ParameterForecastData table
#    3. Loop for all the partid and parameterid
#    4. Update the ParameterForecastData table with new forecasted values
    db_del()
    COUNTER = db_idmax() + 1
    COUNTER1 = db_idmax() + FORECAST_DAYS  + 1
    for mi in MOTOR_ID:
        print('Motor name : ' + MOTOR_NAME[mi-1])
        data_part = data_fetch(partid=mi)
        for pi in PARAMETER_ID:
            print('Parameter name : ' + PARAMETER_NAME[pi-1])
            forecasted_data = forecast_table(data_part, mi, pi, model_name='prophet')
            forecasted_data['Id'] = range(COUNTER, COUNTER1)
            db_update(forecasted_data)
            print(" Wrote forecasted values to DB ")
            COUNTER += FORECAST_DAYS
            COUNTER1 += FORECAST_DAYS
