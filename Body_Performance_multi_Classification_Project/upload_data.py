import pandas as pd
import pymongo
from sqlalchemy import create_engine
import dotenv
import os

dotenv.load_dotenv()

MONGODB_USER = os.getenv("MONGODB_USER")
MONGODB_PASSWORD = os.getenv("MONGODB_PASS")

POSTGRESQL_USER = os.getenv("POSTGRES_USER")
POSTGRESQL_PASSWORD = os.getenv("POSTGRES_PASS")
POSTGRESQL_PORT = os.getenv("POSTGRES_PORT")
POSTGRESQL_HOST = os.getenv("POSTGRES_HOST")
POSTGRESQL_DB = os.getenv("POSTGRES_DB")

def load_data_from_mongodb():
    client = pymongo.MongoClient(f"mongodb+srv://{MONGODB_USER}:{MONGODB_PASSWORD}@cluster0.pygyg.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
    db = client["Body_Performance"]
    collection = db["Body_Performance"]
    data = collection.find()
    df = pd.DataFrame(list(data))
    return df    

def load_data_from_postgresql():
    engine = create_engine(f"postgresql://{POSTGRESQL_USER}:{POSTGRESQL_PASSWORD}@{POSTGRESQL_HOST}:{POSTGRESQL_PORT}/{POSTGRESQL_DB}") 
    df = pd.read_sql_query('select * from "Body_Performance"',con=engine)
    return df

def upload_data_to_mongodb(df):
    client = pymongo.MongoClient(f"mongodb+srv://{MONGODB_USER}:{MONGODB_PASSWORD}@cluster0.pygyg.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
    db = client["Body_Performance"]
    collection = db["Body_Performance"]
    data = df.to_dict(orient='records')
    collection.insert_many(data)
    print("Data uploaded to MongoDB")

def upload_data_to_postgresql(df):
    engine = create_engine(f"postgresql://{POSTGRESQL_USER}:{POSTGRESQL_PASSWORD}@{POSTGRESQL_HOST}:{POSTGRESQL_PORT}/{POSTGRESQL_DB}")
    df.to_sql('Body_Performance', con=engine, if_exists='replace', index=False)
    print("Data uploaded to PostgreSQL")

if __name__ == "__main__":
    df = pd.read_csv('data/cleaned_Data.csv')
    df_mongo = df.iloc[:len(df)//2]
    df_postgresql = df.iloc[len(df)//2:]
    # upload_data_to_mongodb(df_mongo)
    # upload_data_to_postgresql(df_postgresql)