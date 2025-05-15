import pandas as pd
import pymongo
from sqlalchemy import create_engine
import psycopg2
from psycopg2.extras import execute_batch
import dotenv
import os, math

dotenv.load_dotenv()

MONGODB_USER = os.getenv("MONGODB_USER")
MONGODB_PASSWORD = os.getenv("MONGODB_PASS")

POSTGRESQL_USER = ""
POSTGRESQL_PASSWORD = ""
POSTGRESQL_PORT = 5433
POSTGRESQL_HOST = ""
POSTGRESQL_DB = ""

def upload_data_to_mongodb(df):
    client = pymongo.MongoClient(f"mongodb+srv://{MONGODB_USER}:{MONGODB_PASSWORD}@cluster0.pygyg.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
    db = client[""]
    collection = db[""]
    data = df.to_dict(orient='records')
    collection.insert_many(data)
    print("Data uploaded to MongoDB")

def create_table(cursor, table_name):
    create_table_sql = f"""

    CREATE SCHEMA IF NOT EXISTS my_schema AUTHORIZATION krk_largeright;
    DROP TABLE IF EXISTS my_schema.{table_name};

    CREATE TABLE my_schema.{table_name} (
        order_id TEXT,
        payment_sequential INTEGER,
        payment_type TEXT,
        payment_installments INTEGER,
        payment_value FLOAT
    );
    """
    cursor.execute(create_table_sql)
    print(f"Table '{table_name}' created")

def upload_data_to_postgresql(df, batch_size=1000, table_name='order_payments'):
    # engine = create_engine(f"postgresql://{POSTGRESQL_USER}:{POSTGRESQL_PASSWORD}@{POSTGRESQL_HOST}:{POSTGRESQL_PORT}/{POSTGRESQL_DB}")
    # total_rows = len(df)
    # total_batches = math.ceil(total_rows / batch_size)
    # print(f"Total rows: {total_rows} — Uploading in {total_batches} batches of {batch_size}")
    # df.to_sql('order_payments', con=engine, if_exists='replace', index=False, chunksize=batch_size, method='multi')

    conn = psycopg2.connect(
            dbname=POSTGRESQL_DB,
            user=POSTGRESQL_USER,
            password=POSTGRESQL_PASSWORD,
            host=POSTGRESQL_HOST,
            port=POSTGRESQL_PORT
        )
    cursor = conn.cursor()

    # Create table if it doesn't exist
    create_table(cursor, table_name)

    # Prepare columns and insert query
    columns = ','.join(df.columns)
    values_template = ','.join(['%s'] * len(df.columns))
    insert_query = f"INSERT INTO my_schema.{table_name} ({columns}) VALUES ({values_template})"
    # Convert DataFrame to list of tuples
    data = [tuple(x) for x in df.to_numpy()]
    print(f"Uploading {len(data)} rows in batches of {batch_size}...")
    # Execute in batches
    execute_batch(cursor, insert_query, data, page_size=batch_size)
    conn.commit()
    cursor.close()
    conn.close()


    print("Data uploaded to PostgreSQL")

if __name__ == "__main__":
    df_postgresql = pd.read_csv('Data/order_payments.csv')
    
    print(df_postgresql.head())
    upload_data_to_postgresql(df_postgresql)