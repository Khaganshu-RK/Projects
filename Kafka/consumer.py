from confluent_kafka import Consumer, KafkaError, KafkaException
import dotenv
import os
import logging

LOG_FILE = 'consumer.log'
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
# Additional configs for Kafka consumer
"auto.offset.reset":"earliest",
"group.id":"my_group",
}

consumer = Consumer(config)
consumer.subscribe([TOPIC_NAME])

def process_message(msg):
    try:
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                print('End of partition event')
                logging.info('End of partition event')
            elif msg.error():
                raise KafkaException(msg.error())
        else:
            record_key = msg.key().decode('utf-8') if msg.key() else None
            record_value = msg.value().decode('utf-8')
            print(f"Received message: key={record_key}, value={record_value}")
            logging.info(f"Received message: key={record_key}, value={record_value}")
    except Exception as e:
        print(f"Error processing message: {e}")
        logging.error(f"Error processing message: {e}")


if __name__ == "__main__":
    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg:
                process_message(msg)
    except KeyboardInterrupt:
        print("Consumer interrupted by user")
        logging.info("Consumer interrupted by user")
    except Exception as e:
        print(f"Consumer error: {e}")
        logging.error(f"Consumer error: {e}")
    finally:
        consumer.close()
        print("Consumer closed")
        logging.info("Consumer closed")
            