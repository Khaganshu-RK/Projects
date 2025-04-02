# Kafka Streaming Project Using confluent_kafka

This project demonstrates a simple real-time streaming setup using the `confluent_kafka` library. It provides both a producer and a consumer to showcase how to publish and consume messages via a Kafka broker.

## Project Structure

```
consumer.py       # Kafka consumer that subscribes to a topic and processes incoming messages, logging output to Logs/consumer.log
producer.py       # Kafka producer that reads messages from data/all-scripts.csv and publishes them to Kafka, logging status to Logs/producer.log
README.md         # This readme file
data/
    all-scripts.csv   # Sample CSV file containing messages to be published
Logs/
    consumer.log      # Log file capturing consumer events
    producer.log      # Log file capturing producer events
```

## Prerequisites

- **Python 3.12+**
- A running Kafka broker (local or remote). You can start Kafka using Docker or install it directly.
- Install the `confluent_kafka` library:
  ```bash
  pip install confluent-kafka
  ```

## Configuration

Both `producer.py` and `consumer.py` contain configuration settings for:

- **Bootstrap Servers:** Set the Kafka broker address (e.g., `"localhost:9092"`).
- **Topic Names:** Define the Kafka topic to which messages are published and from which they are consumed.

Review and adjust these settings in the scripts to match your Kafka environment.

## Running the Project

### Start the Kafka Broker

Ensure your Kafka broker is running. For example, using Docker:

```bash
docker run -d --name zookeeper -p 2181:2181 zookeeper:latest
docker run -d --name kafka -p 9092:9092 --link zookeeper:zookeeper wurstmeister/kafka
```

### Run the Producer

Publish messages from the CSV file to Kafka:

```bash
python Kafka/producer.py
```

The producer reads `data/all-scripts.csv` line by line and sends each message to the configured Kafka topic. Check `Logs/producer.log` for publish status.

### Run the Consumer

Consume messages from Kafka:

```bash
python Kafka/consumer.py
```

The consumer subscribes to the specified Kafka topic, processes incoming messages, and logs the events to `Logs/consumer.log`.

## How It Works

- **Producer:**  
  Utilizes `confluent_kafka.Producer` to read messages from a CSV file and publish them to a Kafka topic. Delivery reports are logged to help monitor successful message transmissions.

- **Consumer:**  
  Uses `confluent_kafka.Consumer` to subscribe to the Kafka topic, continuously polling for new messages. Received messages are processed and logged accordingly.

## Additional Notes

- **Logging:**  
  Both scripts output logs into the `Logs/` directory for easier debugging and monitoring.
- **Customization:**  
  Modify the configuration parameters in the Python scripts as per your Kafka environment and topic requirements.
- **Scalability:**  
  Extend this setup by adding additional producers or consumers to meet more demanding real-time processing needs.

Happy Streaming!
