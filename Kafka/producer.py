from confluent_kafka import Producer
import json
import pandas as pd
import dotenv
import os
import logging

LOG_FILE = 'producer.log'
os.mkdir(os.path.join(os.path.dirname(__file__), "Logs")) if not os.path.exists(os.path.join(os.path.dirname(__file__), "Logs")) else None
LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), "Logs", LOG_FILE)
# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH)
    ]
)

# Load environment variables from .env file
dotenv.load_dotenv()

# Access environment variables
TOPIC_NAME = os.getenv("KAFKA_TOPIC")
BOOTSTRAP_SERVERS = os.getenv("CONFLUENT_BOOTSTRAP_SERVERS")
SASL_USERNAME = os.getenv("CONFLUENT_API_KEY")
SASL_PASSWORD = os.getenv("CONFLUENT_API_SECRET")
CLIENT_ID = os.getenv("CONFLUENT_CLIENT_ID")

config = {
# Required connection configs for Kafka producer, consumer, and admin
"bootstrap.servers":BOOTSTRAP_SERVERS,
"security.protocol":"SASL_SSL",
"sasl.mechanisms":"PLAIN",
"sasl.username":SASL_USERNAME,
"sasl.password":SASL_PASSWORD,
# Best practice for higher availability in librdkafka clients prior to 1.7
"session.timeout.ms":"45000",
"client.id":CLIENT_ID,
}

file_csv = pd.read_csv('/Users/khaganshu/Projects/Kafka/data/all-scripts.csv')
logging.info(f"CSV file read successfully: {file_csv.shape[0]} rows and {file_csv.shape[1]} columns")

file_csv = file_csv.iloc[100:200]
logging.info(f"CSV file truncated to: {file_csv.shape[0]} rows and {file_csv.shape[1]} columns")

json_data = file_csv.to_json(orient='records')
data = json.loads(json_data)

producer = Producer(config)

def delivery_report(err, msg): 
    if err is not None:
        print('Message delivery failed: {}'.format(err))
        logging.error('Message delivery failed: {}'.format(err))
    else:
        print('Message delivered to {} [{}] at offset {}'.format(msg.topic(), msg.partition(), msg.offset()))
        logging.info('Message delivered to {} [{}] at offset {}'.format(msg.topic(), msg.partition(), msg.offset()))

if __name__ == "__main__":

    for record in data:
        try:
            record_key = str(record['idx']).encode('utf-8')
            record_value = json.dumps(record)
            producer.produce(TOPIC_NAME, key=record_key, value=record_value, callback=delivery_report)
            producer.poll(0)
        except KeyError as e:
            print(f"KeyError: {e}")
            logging.error(f"KeyError: {e}")
            continue
        except Exception as e:
            print(f"Exception: {e}")
            logging.error(f"Exception: {e}")
    producer.flush()