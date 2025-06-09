#!/usr/bin/env python
# coding: utf-8


import pickle
import pandas as pd
import sys

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df


def ride_duration_prediction(taxi_type='yellow', year=2023, month=3):
    df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    print(y_pred.std())
    print('predicted mean duration:', y_pred.mean())
    output_file = f'output/{taxi_type}/{year:04d}-{month:02d}.parquet'




    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')





    df['predicted_duration'] = y_pred

    apply_model(df, y_pred, output_file)


def apply_model(df, y_pred, output_file):

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred
    #df_result['tpep_pickup_datetime'] = df['tpep_pickup_datetime']
    #df_result['PULocationID'] = df['PULocationID']
    #df_result['DOLocationID'] = df['DOLocationID']
    #df_result['actual_duration'] = df['duration']

    #df_result['diff'] = df_result['actual_duration'] - df_result['predicted_duration']

    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )


def run():
    #taxi_type = sys.argv[1] # 'green'
    #year = int(sys.argv[2]) # 2021
    #month = int(sys.argv[3]) # 3
    year = 2023
    month = 5
    taxi_type = 'yellow'
    ride_duration_prediction(
        taxi_type=taxi_type,
        year=year, 
        month=month)


if __name__ == '__main__':
    run()




