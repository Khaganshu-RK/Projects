from airflow import DAG
from airflow.decorators import task
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime, timedelta
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 10, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'data_pipeline',
    default_args=default_args,
    description='A simple data pipeline',
    schedule_interval='@daily',
    catchup=False,
) as dag:
    @task
    def create_table():
        hook = PostgresHook(postgres_conn_id='postgres_conn')
        conn = hook.get_conn()
        cursor = conn.cursor()
        with open('/usr/local/airflow/include/SQL_Query.sql', 'r') as file:
            sql_query = file.read()
        cursor.execute(sql_query)
        conn.commit()
        cursor.close()
        conn.close()
        return "load"        

    @task
    def load_data(varmm):
        if varmm == 'load':
            hook = PostgresHook(postgres_conn_id='postgres_conn')
            engine = hook.get_sqlalchemy_engine()
            df = pd.read_csv('/usr/local/airflow/data/data.csv')
            df.to_sql('data_table', engine, if_exists='replace', index=False)
            return "ingest"
    @task
    def ingest_data(varmm):
        if varmm == 'ingest':
            hook = PostgresHook(postgres_conn_id='postgres_conn')
            engine = hook.get_sqlalchemy_engine()
            df = pd.read_sql_table('data_table', engine)
            return df
    
    @task
    def validate_data(df):
        if df.isnull().sum().sum() > 0:
            df = df.dropna()
        if df.duplicated().sum() > 0:
            df = df.drop_duplicates()
        if (df['y'] < 0).any():
            df = df[df['y'] >= 0]
        if df.empty:
            raise ValueError("Data is empty")
        return df
    
    @task(multiple_outputs=True)
    def transform_data(df):
        X = df.drop('x', axis=1)
        y = df['y']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        y_train = y_train.to_list()
        y_test = y_test.to_list()
        return {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}
    
    @task
    def train_model(X_train, y_train):
        model = LinearRegression()
        y_train = pd.Series(y_train)
        model.fit(X_train, y_train)
        with open('/usr/local/airflow/models/model.pkl', 'wb') as f:
            pickle.dump(model, f)
        return "/usr/local/airflow/models/model.pkl"       
    
    @task
    def save_model(path, y_test, X_test):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        y_test = pd.Series(y_test)
        y_pred = model.predict(X_test)
        print(f"y_pred': {y_pred[0:5]}")
        print(f"y_test': {y_test[0:5]}")
        print("Model loaded and prediction made")

    ############## DAG Execution ##############
    value=create_table()
    ingest=load_data(value)
    load=ingest_data(ingest)
    data=validate_data(load)
    transform=transform_data(data)
    path=train_model(transform['X_train'], transform['y_train'])
    save_model(path, transform['y_test'], transform['X_test'])
    