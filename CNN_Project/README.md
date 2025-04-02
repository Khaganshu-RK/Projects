# CNN Project: UTKFace Classification

This project implements a Convolutional Neural Network (CNN) for classifying faces using the UTKFace dataset. The model is trained to predict attributes such as age, gender, and ethnicity based on facial images. In addition, the project includes a web interface for deployment and visualization.

Note: The Model accuracy is not very high. We need more epochs to train the model and more hyperparameters to improve the accuracy.

## Project Structure

- **project.py**  
  Main entry point for training, evaluating, or inferring with the CNN model.

- **project.log**  
  Log file capturing runtime logs, training progress, and errors.

- **data/**  
  Contains all data-related files.

  - **files/**  
    Contains preprocessed data arrays:
    - `ages.npy` — Numpy array with ages.
    - `ethnicities.npy` — Numpy array with ethnicity labels.
    - `genders.npy` — Numpy array with gender labels.
    - `X.npy` — Numpy array with image data or features.
  - **model/**  
    Contains the trained CNN model:
    - `cnn_model.keras` — The saved Keras model used for prediction.
  - **UTKFace/**  
    Contains a subset of the original UTKFace dataset images for testing or sample inference.
    - Example images: `1_0_0_20161219140623097.jpg.chip.jpg`, `1_0_0_20161219140627985.jpg.chip.jpg`, etc.

- **Web/**  
  Contains the web deployment components.
  - **compose.yaml**  
    Docker Compose configuration for orchestrating the web services.
  - **backend/**  
    Backend service source code and Dockerfile for API and model serving.
  - **frontend/**  
    Frontend code for the user interface.

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
   It is recommended to use a virtual environment.

   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   pip install -r requirements.txt  # Create a requirements.txt if not already present
   ```

3. **Install Additional Dependencies**  
   Ensure you have the necessary libraries installed (e.g., TensorFlow/Keras, numpy). If using Docker for the web service, ensure Docker and Docker Compose are installed.

## Usage

### Running the Model

Execute the model script to perform training, evaluation, or inference:

```sh
python CNN_Project/project.py
```

### Viewing Logs

Check the `project.log` file for detailed logs:

```sh
tail -f CNN_Project/project.log
```

### Web Interface

If you wish to deploy the web service:

1. Navigate to the `Web` directory:
   ```sh
   cd CNN_Project/Web
   ```
2. Start the web service using Docker Compose:
   ```sh
   docker-compose up --build
   ```
   The frontend (Next.js) and backend (FastAPI) services will start, with the frontend accessible at [http://localhost:3000](http://localhost:3000) and the backend at [http://localhost:8000](http://localhost:8000).

## Data Preparation

- Preprocessed numpy arrays (`ages.npy`, `ethnicities.npy`, `genders.npy`, `X.npy`) are expected to be located in the `data/files` directory.
- For training and inference, the CNN uses images from the `data/UTKFace` folder as samples.
- If you have additional data, place it in the appropriate subfolder within `data/`.

## Model Details

- The CNN model architecture and training routine are implemented in [`project.py`](./project.py).
- The trained model is saved in the `data/model/cnn_model.keras` file.
- Modify or extend the model architecture according to your requirements by editing the corresponding sections in the `project.py` script.

## Configuration

- Environment variables: For any configuration (e.g., file paths, hyperparameters), use environment variables or configuration files as necessary.
- Logging: Logging is configured in `project.py` and outputs to `project.log`.

## Acknowledgements

- [UTKFace Dataset](https://susanqq.github.io/UTKFace/) for providing the dataset used in this project.
- Docker and Docker Compose for containerizing the web services.

Happy Coding!
