# Airflow Project with Astronomer

This project is an Apache Airflow setup configured to run in a Dockerized environment using Astronomer's best practices. It provides the structure for developing, testing, and deploying Airflow DAGs and includes sample configurations, DAGs, and tests.

## Project Structure

```
.dockerignore         # Specifies files to ignore during Docker builds
.env                  # Environment variables for local development
.gitignore            # Git ignore settings
airflow_settings.yaml # Pre-configured Airflow settings (e.g., connections, variables, pools)
docker-compose.yml   # Docker Compose configuration for running Airflow services locally (Override by renaming to docker-compose.override.yml)
Dockerfile           # Dockerfile for building the Airflow image with Astronomer
packages.txt         # OS-level package requirements for the Airflow image
README.md            # This documentation file
requirements.txt     # Python package dependencies
.astro/              # Astronomer configuration files (e.g., config.yaml)
dags/                # Airflow DAG definitions (e.g., data_pipeline.py)
data/                # Sample data files (e.g., data.csv) for testing and demos
include/             # Supplementary files such as SQL queries
models/              # (Optional) Directory for machine learning models or related files
plugins/             # Custom or third-party Airflow plugins
tests/               # Unit tests for DAGs and other components
```

## Prerequisites

- **Docker & Docker Compose:** Ensure both are installed to run the containerized Airflow environment.
- **Astronomer CLI (astro):** Recommended tool for building and deploying your Airflow project.

## Usage

### Starting Airflow Locally

1. **Start the Environment with Astronomer CLI:**

   From the project root (`Airflow` folder), run:

   ```bash
   astro dev start
   ```

   This builds the Docker images and starts the required Airflow components (webserver, scheduler, triggerer, and Postgres for the metadata database).

2. **Access the Airflow Web UI:**

   Open your browser and navigate to:
   [http://localhost:8080](http://localhost:8080)

   Login using the default credentials (check your `.env` file or Astronomer documentation for details).

### Running Tests

- Execute tests located in the `tests/dags/` folder to ensure your DAGs are correctly configured:
  ```bash
  python -m unittest discover tests/dags/
  ```

### Deploying to Astronomer

- When ready to deploy to a managed Astronomer environment, follow the Astronomer deployment guidelines using the `astro` CLI:
  ```bash
  astro deploy
  ```

## Additional Notes

- **Configurations:**  
  Adjust `airflow_settings.yaml` and `.env` to configure your Airflow instance for local or production environments.
- **Extensibility:**  
  Add new DAGs in the `dags/` folder, extend your plugins in the `plugins/` folder, or include additional resources in the `include/` folder.
- **Logging:**  
  Monitor Airflow logs via the web UI and container log outputs for debugging and performance analysis.

Happy Coding with Astronomer and Airflow!
