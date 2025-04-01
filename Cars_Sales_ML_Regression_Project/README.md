# Cars Sales ML Regression Project

This project is a machine learning pipeline for predicting car prices based on their features. It leverages several data processing stages including data ingestion, validation, transformation, model training, and evaluation. In addition, a web interface is provided for model deployment and predictions.

## Project Structure

- **exceptions.py**  
  Custom exceptions used throughout the project.

- **logger.py**  
  Logger configuration for recording runtime logs. Refer to [logger.py](Cars_Sales_ML_Regression_Project/logger.py).

- **main.py**  
  Main entry point which orchestrates the pipeline process. Refer to [main.py](Cars_Sales_ML_Regression_Project/main.py).

- **Artifacts/**  
  Contains output artifacts generated during pipeline execution (e.g., intermediate data, model files).

- **Data/**  
  Contains raw and processed data files (e.g., `car_prices.csv`, `cleaned_Data.csv`).

- **Logs/**  
  Logs produced during pipeline execution.

- **Notebooks/**  
  Jupyter notebooks for exploratory data analysis (e.g., `Vehicle_Data_EDA.ipynb`).

- **Schema/**  
  Defines the data schema in YAML format (e.g., `data_schema.yaml`).

- **src/**  
  Contains the core source code for the pipeline:

  - **components/**  
    Implements core functionalities like data ingestion ([data_ingestion.py](Cars_Sales_ML_Regression_Project/src/components/data_ingestion.py)), data validation ([data_validation.py](Cars_Sales_ML_Regression_Project/src/components/data_validation.py)), and model evaluation ([model_evaluation.py](Cars_Sales_ML_Regression_Project/src/components/model_evaluation.py)).
  - **constants/**  
    Global constants used across the project. See [constants/**init**.py](Cars_Sales_ML_Regression_Project/src/constants/__init__.py).
  - **extra/**  
    Contains configuration entities ([config_entity.py](Cars_Sales_ML_Regression_Project/src/extra/config_entity.py)) and artifact definitions ([artifact_entity.py](Cars_Sales_ML_Regression_Project/src/extra/artifact_entity.py)).
  - **pipelines/**  
    Integration of different components for pipeline execution.
  - **utils/**  
    Utility functions to read and write files (e.g., CSV, numpy arrays, pickle) in [main_utils.py](Cars_Sales_ML_Regression_Project/src/utils/main_utils.py).

- **Web/**  
  Contains deployment resources:
  - **compose.yaml** for Docker Compose configuration.
  - **backend/** and **frontend/** directories for API and UI respectively.
  - Frontend API routes are demonstrated in [Web/frontend/app/api/getColumns/route.ts](Cars_Sales_ML_Regression_Project/Web/frontend/app/api/getColumns/route.ts).

### Prerequisites

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
   Use a virtual environment and install dependencies.

   ```sh
   python -m venv venv
   source venv/bin/activate  # For Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Additional Setup**
   - Ensure required Python packages (e.g., pandas, numpy, scikit-learn, PyYAML) are installed.
   - Install Docker and Docker Compose if you plan to run the web service.

## Usage

### Pipeline Execution

Run the main script to execute the entire pipeline:

```sh
python main.py
```

This script will:

- Ingest data from [Data](Cars_Sales_ML_Regression_Project/Data).
- Validate data using components in [data_validation.py](Cars_Sales_ML_Regression_Project/src/components/data_validation.py).
- Transform data as configured in [config_entity.py](Cars_Sales_ML_Regression_Project/src/extra/config_entity.py).
- Train and evaluate the model, with evaluation logic detailed in [model_evaluation.py](Cars_Sales_ML_Regression_Project/src/components/model_evaluation.py).
- Upload/download artifacts to/from AWS S3 as specified in the pipeline.

### Logging

Logs are saved in the [Logs](Cars_Sales_ML_Regression_Project/Logs) directory. For real-time monitoring, you can execute:

```sh
tail -f Logs/log_<timestamp>.log
```

### Web Deployment

To deploy the web interface for model serving:

1. Navigate to the **Web** directory:

   ```sh
   cd Cars_Sales_ML_Regression_Project/Web
   ```

2. Start the Docker containers:

   ```sh
   docker-compose up --build
   ```

   The frontend (Next.js) and backend (FastAPI) services will start, with the frontend accessible at [http://localhost:3000](http://localhost:3000) and the backend at [http://localhost:8000](http://localhost:8000).

   The backend API and frontend UI will be built and served. Refer to [Cars_Sales_ML_Regression_Project/Web/compose.yaml](Cars_Sales_ML_Regression_Project/Web/compose.yaml) for configuration details.

## Pipeline Components Details

- **Data Ingestion**  
  Implemented in [data_ingestion.py](Cars_Sales_ML_Regression_Project/src/components/data_ingestion.py), this module copies data from the source file and creates a feature store.

- **Data Validation**  
  The [DataValidation](Cars_Sales_ML_Regression_Project/src/components/data_validation.py) class uses a YAML schema ([Schema/data_schema.yaml](Cars_Sales_ML_Regression_Project/Schema/data_schema.yaml)) to validate the ingested data.

- **Data Transformation & Model Training**  
  Configurations for data transformation and model training are defined in [config_entity.py](Cars_Sales_ML_Regression_Project/src/extra/config_entity.py). Artifacts generated during this phase are captured in [artifact_entity.py](Cars_Sales_ML_Regression_Project/Artifacts/).

- **Model Evaluation**  
  The [ModelEvaluation](Cars_Sales_ML_Regression_Project/src/components/model_evaluation.py) class evaluates the trained model using regression metrics like R2 score and, for classification, metrics such as accuracy, precision, recall, and f1 score depending on the pipeline stage.

- **Utilities**  
  File I/O operations such as reading CSVs, numpy arrays, YAML files, and pickling objects are defined in [main_utils.py](Cars_Sales_ML_Regression_Project/src/utils/main_utils.py).

## Acknowledgements

- Thank you to all contributors and open-source projects that made this work possible.

Happy Coding!
