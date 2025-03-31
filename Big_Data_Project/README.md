# Big Data Project

The **Big Data Project** is a multi-component system that leverages Apache Airflow for orchestrating data pipelines and Apache Kafka for streaming data processes. This project was generated with Astronomer for Airflow deployments and includes sample Kafka producer/consumer scripts. It is designed to help you build, test, and deploy scalable data workflows.

---

## Project Structure

```
Big_Data_Project/
├── Airflow/
│   ├── .dockerignore
│   ├── .env
│   ├── .gitignore
│   ├── airflow_settings.yaml              # Configure local Airflow connections, variables, and pools
│   ├── docker-com.yml                     # Docker Compose configuration for Airflow
│   ├── Dockerfile                         # Astro Runtime Docker image definition
│   ├── packages.txt                       # OS-level package requirements
│   ├── README.md                          # Airflow project documentation (see below)
│   ├── requirements.txt                   # Python package requirements for Airflow
│   ├── .astro/                            # Astronomer configuration files
│   │   ├── config.yaml
│   │   ├── dag_integrity_exceptions.txt
│   │   └── test_dag_integrity_default.py
│   ├── dags/                              # Airflow DAG definitions
│   │   ├── .airflowignore
│   │   ├── data_pipeline.py
│   │   └── __pycache__/
│   │         └── data_pipeline.cpython-312.pyc
│   ├── data/                              # Sample CSV file for data ingestion
│   │   └── data.csv
│   ├── include/                           # Additional SQL queries or files to include
│   │   └── SQL_Query.sql
│   ├── models/                            # ML or data models (if applicable)
│   ├── plugins/                           # Custom or third-party Airflow plugins
│   └── tests/                             # Unit tests for your DAGs
│         └── dags/
│                 └── test_dag_example.py
└── Kafka/
    ├── consumer.py                        # Kafka consumer script
    ├── producer.py                        # Kafka producer script
    └── data/
          └── all-scripts.csv              # Sample data used by Kafka processes
```

---

## Getting Started

### Prerequisites

- **Docker** and **Docker Compose** – Required for running Airflow via Astronomer.
- **Python 3.8+** – To run Airflow and utility scripts.
- (Optional) **Kafka** – To run and test streaming workflows. You can install Kafka locally or use a cloud provider.

### Installation

1. **Clone the Repository**

   ```sh
   git clone https://github.com/yourusername/Big_Data_Project.git
   cd Big_Data_Project
   ```

2. **Set Up Airflow Environment**

   Navigate to the `Airflow` directory:

   ```sh
   cd Airflow
   ```

   Install the required Python packages:

   ```sh
   pip install -r requirements.txt
   ```

3. **Set Up Kafka Environment**

   Ensure you have a running Kafka instance. If you need Kafka on your local machine, consider using Docker:

   ```sh
   docker run -d --name zookeeper -p 2181:2181 zookeeper:3.7
   docker run -d --name kafka -p 9092:9092 --link zookeeper:zookeeper wurstmeister/kafka
   ```

---

## Airflow Usage

### Running Airflow Locally (Astronomer)

This project uses the Astronomer CLI to simplify the deployment and management of your Airflow environment.

1. **Start Airflow**

   In the `Airflow` directory, launch Airflow using:

   ```sh
   astro dev start
   ```

   This will spin up 4 Docker containers for:

   - **Postgres:** Airflow's Metadata Database (port 5432)
   - **Webserver:** The Airflow UI (port 8080)
   - **Scheduler:** Triggers task execution
   - **Triggerer:** Handles deferred tasks

2. **Verify Containers**

   Confirm that the containers are running:

   ```sh
   docker ps
   ```

3. **Access the Airflow UI**

   Open your browser and navigate to [http://localhost:8080](http://localhost:8080). Log in with:

   - **Username:** admin
   - **Password:** admin

4. **Local Data Pipeline**

   The DAG defined in `dags/data_pipeline.py` demonstrates a sample ETL pipeline. Modify or add new DAGs in this directory as needed.

### Additional Airflow Configuration

- **airflow_settings.yaml:**  
  Use this file to pre-configure connections, variables, and pool settings for local development.

- **include Directory:**  
  Place any supplementary SQL files or data needed across your DAGs here.

- **Tests:**  
  Execute tests from the `tests/dags` directory to ensure your DAGs behave as expected.

---

## Kafka Usage

The Kafka component provides simple scripts to produce and consume messages, enabling you to test streaming data workflows.

### Running the Producer

From the root of the project or within the `Kafka` directory, run:

```sh
python Kafka/producer.py
```

This will read from the sample data (`Kafka/data/all-scripts.csv`) and send messages to your Kafka broker.

### Running the Consumer

Similarly, run the consumer with:

```sh
python Kafka/consumer.py
```

The consumer will listen for messages from the Kafka broker and process them accordingly.

---

## Deployment

### Local Deployment

- **Airflow:**  
  Deploy your Airflow project locally as described above using `astro dev start`.

- **Kafka:**  
  Ensure your Kafka instance is running (as detailed in the Kafka installation) or use a managed Kafka service.

### Deployment to Astronomer

If you have an Astronomer account, you can deploy your Airflow project to Astronomer. For detailed deployment instructions, refer to the [Astronomer documentation](https://www.astronomer.io/docs/astro/deploy-code/).

---

## Contributing

Contributions are welcome! Follow these steps to contribute:

1. Fork the repository.
2. Create a branch for your feature or bug fix.
3. Commit your changes with meaningful commit messages.
4. Open a pull request with a brief description of your changes.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- **Astronomer & Apache Airflow:** For orchestrating complex workflows.
- **Apache Kafka:** For robust real-time data streaming.
- Thanks to the open-source community for their contributions.

Happy Coding!
