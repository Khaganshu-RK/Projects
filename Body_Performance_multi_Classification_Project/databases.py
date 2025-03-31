import pymongo
import psycopg2
import dotenv
import os
from logger import logging

dotenv.load_dotenv()

MONGODB_USER = os.getenv("MONGODB_USER")
MONGODB_PASSWORD = os.getenv("MONGODB_PASS")

POSTGRESQL_USER = os.getenv("POSTGRES_USER")
POSTGRESQL_PASSWORD = os.getenv("POSTGRES_PASS")
POSTGRESQL_PORT = int(os.getenv("POSTGRES_PORT"))
POSTGRESQL_HOST = os.getenv("POSTGRES_HOST")
POSTGRESQL_DB = os.getenv("POSTGRES_DB")


class Databases:
    def __init__(self, db_type):
        if db_type == "mongodb":
            self.client = pymongo.MongoClient(f"mongodb+srv://{MONGODB_USER}:{MONGODB_PASSWORD}@cluster0.pygyg.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
            self.db = self.client["Body_Performance"]
            self.collection = self.db["Body_Performance"]
            logging.info(f"Connected to MongoDB DB: {self.db} and collection: {self.collection}")
        elif db_type == "postgresql":
            self.conn = psycopg2.connect(
                host=POSTGRESQL_HOST,
                port=POSTGRESQL_PORT,
                database=POSTGRESQL_DB,
                user=POSTGRESQL_USER,
                password=POSTGRESQL_PASSWORD
            )
            self.cursor = self.conn.cursor()
        else:
            raise Exception("Invalid database type")
        
    def read_data(self, db_type):
        if db_type == "mongodb":
            data = self.collection.find({}, {"_id": 0})
            return data
        elif db_type == "postgresql":
            self.cursor.execute('SELECT * FROM "Body_Performance"')
            data = self.cursor.fetchall()
            column_names = [desc[0] for desc in self.cursor.description]
            return data, column_names
        else:
            raise Exception("Database not found")