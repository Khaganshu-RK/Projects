# Airline Satisfaction Classification Project

The Airline Satisfaction Classification Project is designed to predict customer satisfaction by analyzing airline data. This project incorporates a full machine learning pipeline including data ingestion, validation, transformation, model training, evaluation, and web-based deployment.

## Overview

This project:

- **Processes Airline Data:** Reads raw data from CSV files and validates it against a defined YAML schema ([data_schema.yaml](Airline_Satisfaction_Classification_Project/Schema/data_schema.yaml)).
- **Transforms Data:** Applies preprocessing to convert raw features into a format suitable for modeling using a pipeline defined in [`src/pipelines/pipeline.py`](Airline_Satisfaction_Classification_Project/src/pipelines/pipeline.py).
- **Trains Models:** Supports both regression and classification models with hyperparameter tuning ([model_trainer.py](Airline_Satisfaction_Classification_Project/src/components/model_trainer.py)).
- **Evaluates Models:** Generates performance reports based on various metrics using custom evaluation functions ([model_evaluation.py](Airline_Satisfaction_Classification_Project/src/components/model_evaluation.py)).
- **Deploys a Web Interface:** Provides a Dockerized web frontend and a FastAPI backend to handle predictions ([Web/compose.yaml](Airline_Satisfaction_Classification_Project/Web/compose.yaml)).

## Project Structure

```
Airline_Satisfaction_Classification_Project/
│
├── Artifacts/                  # Output artifacts (models, reports, etc.)
├── Data/                       # Raw and cleaned data files (e.g., Airline_Data.csv)
├── Logs/                       # Execution logs
├── Notebooks/                  # Jupyter notebooks for EDA
├── Schema/                     # Schema definitions for data validation ([data_schema.yaml](Airline_Satisfaction_Classification_Project/Schema/data_schema.yaml))
├── src/                        # Core source code:
│   ├── components/             # Modules for data ingestion, transformation, validation, model training, etc.
│   │   ├── data_ingestion.py   # Handles reading and splitting the raw data ([data_ingestion.py](Airline_Satisfaction_Classification_Project/src/components/data_ingestion.py))
│   │   ├── data_validation.py  # Validates input datasets ([data_validation.py](Airline_Satisfaction_Classification_Project/src/components/data_validation.py))
│   │   ├── data_transformation.py  # Preprocesses data and encodes target variable ([data_transformation.py](Airline_Satisfaction_Classification_Project/src/components/data_transformation.py))
│   │   ├── model_trainer.py    # Trains various models and tunes hyperparameters ([model_trainer.py](Airline_Satisfaction_Classification_Project/src/components/model_trainer.py))
│   │   └── model_evaluation.py # Evaluates the trained models ([model_evaluation.py](Airline_Satisfaction_Classification_Project/src/components/model_evaluation.py))
│   ├── extra/                  # Configuration and artifact entity definitions ([config_entity.py](Airline_Satisfaction_Classification_Project/src/extra/config_entity.py))
│   ├── pipelines/              # End-to-end pipeline integration ([pipeline.py](Airline_Satisfaction_Classification_Project/src/pipelines/pipeline.py))
│   └── utils/                  # Utility functions for file I/O, cloud operations, etc. ([main_utils.py](Airline_Satisfaction_Classification_Project/src/utils/main_utils.py))
│
├── Web/                        # Dockerized web deployment:
│   ├── backend/                # FastAPI based backend for prediction ([server.py](Airline_Satisfaction_Classification_Project/Web/backend/app/server.py))
│   └── frontend/               # Next.js based frontend for user interactions
│       └── app/
│           └── form/           # A sample form for input to the prediction API ([page.tsx](Airline_Satisfaction_Classification_Project/Web/frontend/app/form/page.tsx))
│
├── exceptions.py               # Custom exception classes ([exceptions.py](Airline_Satisfaction_Classification_Project/exceptions.py))
├── logger.py                   # Logger configuration for runtime logging ([logger.py](Airline_Satisfaction_Classification_Project/logger.py))
└── main.py                     # Entry point which orchestrates the pipeline process ([main.py](Airline_Satisfaction_Classification_Project/main.py))
```

## Artifacts Structure

```
Artifacts/
├── run_datetime/
│   ├── 01_Data_Ingestion/
│   │   ├── feature_store/
│   │   │   └── raw_Data.csv
│   │   └── ingested/
│   │       ├── test.csv
│   │       └── train.csv
│   ├── 02_Data_Validation/
│   │   ├── drift_report/
│   │   │   └── drift_report.yaml
│   │   └── validated/
│   │       ├── train.csv
│   |       └── test.csv
│   ├── 03_Data_Transformation/
│   │   |── transformed/
│   │   |    ├── test.npy
│   │   |    └── train.npy
│   |   └── transformed_object/
│   |        |── input_preprocessor.pkl
│   |        └── target_encoder.pkl
│   ├── 04_Model_Training/
│   │   ├── final_model/
│   │   │   └── model.pkl
│   │   └── models_report/
│   │       └── models_report.yaml
```

## Prerequisites

- **Python 3.12+**
- **Node.js** and **npm** (for the frontend)
- **Docker** and **Docker Compose** (for the web service)

## Installation

1. **Clone the Repository**

   ```sh
   git clone https://github.com/Khaganshu-RK/Projects.git
   cd Projects
   ```

2. **Set Up Python Environment**

   Create a virtual environment and activate it:

   ```sh
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

3. **Install Frontend Dependencies (Optional)**

   If you need to run or modify the web frontend:

   ```sh
   cd Airline_Satisfaction_Classification_Project/Web/frontend
   npm install
   ```

## Usage

### Running the Pipeline

Run the complete training and evaluation pipeline by executing:

```sh
python main.py
```

This script:

- Performs data ingestion via [`src/components/data_ingestion.py`](Airline_Satisfaction_Classification_Project/src/components/data_ingestion.py).
- Validates and transforms data using modules in [`src/components/data_validation.py`](Airline_Satisfaction_Classification_Project/src/components/data_validation.py) and [`src/components/data_transformation.py`](Airline_Satisfaction_Classification_Project/src/components/data_transformation.py).
- Trains and evaluates models via [`src/components/model_trainer.py`](Airline_Satisfaction_Classification_Project/src/components/model_trainer.py) and [`src/components/model_evaluation.py`](Airline_Satisfaction_Classification_Project/src/components/model_evaluation.py).
- Uploads/downloads required files from an S3 bucket via [`src/utils/cloud_utils.py`](Airline_Satisfaction_Classification_Project/src/utils/cloud_utils.py).

### Web Deployment

To deploy the web interface:

1. **Navigate to the Web Directory**

   ```sh
   cd Airline_Satisfaction_Classification_Project/Web
   ```

2. **Start Docker Containers**

   ```sh
   docker-compose up --build
   ```

   The frontend (Next.js) and backend (FastAPI) services will start, with the frontend accessible at [http://localhost:3000](http://localhost:3000) and the backend at [http://localhost:8000](http://localhost:8000).

## Acknowledgements

- Thanks to the developers behind Python, FastAPI, Next.js, Docker, and the open-source community for providing the tools that make this project possible.

Happy Coding!
