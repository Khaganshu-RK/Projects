# Repository Projects Overview

This repository contains multiple self-contained projects demonstrating different aspects of machine learning, deep learning, data engineering, and web deployment. Below is a detailed overview of each project, outlining its purpose, key components, and overall structure.

---

## 1. Cars Sales ML Regression Project

**Purpose:**  
Predict car prices based on various features using a complete machine learning pipeline.

**Key Components:**

- **Data Pipeline:**  
  Reads raw CSV data, cleans it, and transforms it for modeling.
- **Model Training:**  
  Employs regression techniques (with hyperparameter tuning using GridSearchCV, for example) to train predictive models.
- **Evaluation:**  
  Evaluates regression performance metrics and generates artifacts (e.g., trained models, intermediate datasets).
- **Logging & Exception Handling:**  
  Uses custom modules (`logger.py` and `exceptions.py`) to manage logs and errors.
- **Web Interface:**  
  Includes a Dockerized web environment (separate backend and frontend) for deploying the model and making predictions.

**Structure Highlights:**

- **src/** – Core pipeline scripts and utilities.
- **Data/** – Contains raw and processed datasets.
- **Logs/** – Stores execution logs.
- **Web/** – Configuration for Docker Compose and web deployment.

---

## 2. Airline Satisfaction Classification Project

**Purpose:**  
Predict airline customer satisfaction using classification techniques on preprocessed airline data.

**Key Components:**

- **Data Processing Pipeline:**  
  Ingests raw CSV data, validates using a YAML schema (`Schema/data_schema.yaml`), and transforms data ready for training.
- **Model Training & Evaluation:**  
  Trains a classification model, evaluating it with appropriate metrics. Artifacts and model outputs are organized in dedicated directories.
- **Visualization & Web Deployment:**  
  Offers a web interface (with a Next.js frontend and API backend) to deploy the model and visualize predictions.
- **Logging & Error Management:**  
  Implements comprehensive logging and exception handling.

**Structure Highlights:**

- **Data/** – Hosts raw and cleaned data files.
- **Artifacts/** – Saves outputs from different stages of the pipeline.
- **Notebooks/** – Contains Jupyter notebooks for exploratory data analysis (EDA).
- **src/** – Houses the core code for data ingestion, transformation, and modeling.
- **Web/** – Provides the Docker Compose setup and code for the deployment of a web interface.

---

## 3. UTKFace Classification Project

**Purpose:**  
Utilize a Convolutional Neural Network (CNN) to classify face images by age, gender, and ethnicity based on the UTKFace dataset.

**Key Components:**

- **Deep Learning Pipeline:**  
  Implements a CNN architecture in the main script (`project.py`) to process images and perform classification.
- **Data Management:**  
  Organizes the UTKFace dataset and associated label data (stored as numpy arrays).
- **Model Artifacts:**  
  Saves the trained model and logs model performance, allowing for inference demonstrations.
- **Web Deployment:**  
  Supports a containerized web interface to execute model inference and showcase output.

**Structure Highlights:**

- **Data/** – Contains the UTKFace images and preprocessed data.
- **Main Script:**  
  `project.py` that coordinates model training, evaluation, and inference.
- **Web/** – Provides the necessary Docker configurations and deployment code.

---

## 4. Body Performance Multi-Classification Project

**Purpose:**  
Classify body performance data into multiple categories by applying a full machine learning workflow.

**Key Components:**

- **Data Workflow:**  
  Ingests raw performance data from CSV files, performs cleaning, and prepares data for multi-class classification.
- **Model Training:**  
  Trains classification models and evaluates them with relevant metrics.
- **Visualization & Web UI:**  
  Includes a web interface to visualize the data outcomes and predictions interactively.
- **Utilities:**  
  Implements logging, exception handling, and helper modules for robust operation.

**Structure Highlights:**

- **Data/** – Raw and processed data files (e.g., `bodyPerformance.csv` and `cleaned_Data.csv`).
- **Notebooks/** – Contains EDA notebooks.
- **src/** – Includes all core functionality for the pipeline.
- **Web/** – Dockerized deployment of the web-based interface for user interaction.

---

## 5. Big Data Project

**Azure Big Data Pipeline**

A scalable **big data pipeline** using **Azure Data Factory, ADLS Gen2, Databricks, Synapse, MongoDB, and PostgreSQL**. The project follows the **Medallion Architecture (Bronze → Silver → Gold)** to ingest data from **GitHub APIs & SQL**, process it with **PySpark in Databricks**, enrich it with **MongoDB**, and store analytics-ready data in **Azure Synapse as CETAS Parquet tables**. Ideal for **data science and analytics** workflows.

## 6. Kafka Project

**Purpose:**  
Demonstrate the usage of Kafka for real-time streaming data processing.

**Key Components:**

- **Producer:**
  A Python script that reads data from a CSV file and sends it to a Kafka topic.
- **Consumer:**
  A Python script that consumes messages from a Kafka topic and processes them.

**Structure Highlights:**

- **Kafka/** – Contains the Kafka producer/consumer scripts and sample data used for simulation.

## 7. Airflow Project

**Purpose:**  
Demonstrate the usage of Apache Airflow for orchestrating data pipelines using Astronomer. I have created a LinearRegression DAG for a sample ETL and Data Science pipeline.

**Key Components:**

- **DAGs:**
  Contains Airflow DAGs for ETL pipelines and Data Science workflows located in the `Airflow/dags/` folder. Uses Astronomer configurations to simplify local development.

**Structure Highlights:**

- **Airflow/** – All configurations, DAGs, plugins, and settings for managing data pipelines.

---

## Final Notes

Each project in this repository is developed with best practices in mind. They feature modular designs, robust error handling via custom logging and exception modules, and dockerized environments (where applicable) for consistent deployments. Whether you are exploring machine learning regression, deep learning classification, or big data orchestration, this repository offers a wealth of examples to learn from and extend.
