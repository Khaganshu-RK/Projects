# Body Performance Multi-Classification Project

The **Body Performance Multi-Classification Project** is a machine learning pipeline that processes and classifies body performance data. It includes data ingestion, cleaning, transformation, model training, evaluation, and a web-based interface for visualization and predictions.

## Project Structure

```
databases.py
exceptions.py
logger.py
main.py
upload_data.py
__pycache__/
    Data compiled Python files
Data/
    bodyPerformance.csv         # Raw performance data
    cleaned_Data.csv            # Processed data ready for analysis
Notebooks/
    EDA.ipynb                   # Exploratory data analysis notebook
Schema/
    data_schema.yaml            # Schema for data validation
src/
    __init__.py
    components/
        data_ingestion.py         # Module for data ingestion
        data_transformation.py    # Module for data transformation
        data_validation.py        # Module for data validation
        ...                     # Other modules (model training, evaluation, etc.)
    constants/
        ...                     # Global constants
    extra/
        ...                     # Additional configuration and helpers
    pipelines/
        ...                     # Integration of pipeline components
    utils/
        ...                     # Utility functions (I/O operations, etc.)
Web/
    compose.yaml                # Docker Compose config for web services
    backend/                    # Backend API and model serving code
    frontend/                   # Frontend code (Next.js application)
```

## Getting Started

### Prerequisites

- **Python 3.8+**
- **Node.js** and **npm** (for the frontend)
- **Docker** and **Docker Compose** (for the web service)

### Installation

1. **Clone the Repository**

   ```sh
   git clone https://github.com/yourusername/Body_Performance_multi_Classification_Project.git
   cd Body_Performance_multi_Classification_Project
   ```

2. **Set Up the Python Environment**

   Create and activate a virtual environment:

   ```sh
   python -m venv venv
   source venv/bin/activate  # For Windows: venv\Scripts\activate
   ```

   Install required Python packages:

   ```sh
   pip install -r requirements.txt
   ```

3. **Install Frontend Dependencies (Optional)**

   If you plan to modify or run the frontend locally:

   ```sh
   cd Web/frontend
   npm install
   cd ../../
   ```

## Usage

### Running the Pipeline

The main pipeline, which handles data processing, model training, and evaluation, is executed via `main.py`.

To run the complete pipeline:

```sh
python main.py
```

If you need to upload new data into the pipeline:

```sh
python upload_data.py
```

### Working with Data

- **Raw Data:** Located at `Data/bodyPerformance.csv`
- **Cleaned Data:** Generated and saved as `Data/cleaned_Data.csv`
- **Data Validation:** Uses the schema defined in `Schema/data_schema.yaml`

### Exploring Data

For exploratory data analysis, open the Jupyter Notebook:

```sh
jupyter notebook Notebooks/EDA.ipynb
```

### Running the Web Service

The project includes a web interface for visualizing predictions and interacting with the model.

1. **Start the Web Service**

   Navigate to the `Web` directory and start Docker Compose:

   ```sh
   cd Web
   docker-compose up --build
   ```

   This will launch the backend API and the Next.js frontend. By default, the frontend is accessible at [http://localhost:3000](http://localhost:3000) (or another port if configured differently).

2. **Frontend Details**

   The Next.js frontend, found in `Web/frontend`, provides an interactive interface. For more information on Next.js setup and deployment, refer to the documentation included in [`Web/frontend/README.md`](./Web/frontend/README.md).

## Components Overview

- **Data Ingestion:**  
  The `data_ingestion.py` module reads raw CSV data and feeds it into the pipeline.

- **Data Transformation:**  
  Implemented in `data_transformation.py`, this module cleans and transforms raw data into a format suitable for modeling.

- **Data Validation:**  
  The `data_validation.py` module checks the data against the schema defined in `Schema/data_schema.yaml` to ensure quality and consistency.

- **Logging & Exception Handling:**  
  Logging is configured in `logger.py`, while custom exceptions are managed in `exceptions.py`.

- **Utilities & Constants:**  
  Common functions and constants are maintained in the `src/utils/` and `src/constants/` directories respectively.

## Contributing

Contributions are welcome! If you wish to help improve the project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes with descriptive messages.
4. Open a pull request describing your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

- Thanks to the developers behind Python, Next.js, Docker, and other open-source tools that made this project possible.
- Special recognition to the open-source community for ongoing support and contributions.

Happy Coding!
