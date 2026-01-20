# Housing Price Prediction - MLOps Project

This project incorporates Machine Learning Operations (MLOps) practices to build a robust pipeline for predicting housing prices. 

## Project Status

**Current implementation status:**
The pipeline is currently completed up to the **Model Training** stage.

- [x] Data Ingestion
- [x] Data Validation
- [x] Data Transformation
- [x] Model Training
- [ ] Model Evaluation (To be implemented)
- [ ] Model Pusher (To be implemented)

## Prerequisites

Before running the project, ensure you have the following installed:
- Python 3.8+
- MongoDB

## Environment Variables

You need to set up the following environment variables. Create a `.env` file in the root directory or export them in your shell.

```bash
MONGODB_URL="mongodb+srv://<username>:<password>@cluster0.example.mongodb.net/?retryWrites=true&w=majority"
```

> **Note:** The `MONGODB_URL` is required for the Data Ingestion component to fetch data from the database.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd housing-price-mlops
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

To start the training pipeline, run the `app.py` script:

```bash
python app.py
```

This will trigger the following steps:
1. Ingest data from the MongoDB database.
2. Validate the data against the identified schema.
3. Transform the data (preprocessing).
4. Train the model.

## Project Structure

```
├── artifact/               # Stores generated artifacts (data, models, reports)
├── config/                 # Configuration files (model.yaml, schema.yaml)
├── notebook/               # Jupyter notebooks for experimentation
├── src/                    # Source code
│   ├── components/         # Pipeline components (Ingestion, Validation, etc.)
│   ├── configuration/      # Database and Cloud configurations
│   ├── constants/          # Project constants
│   ├── entity/             # Data classes for config and artifacts
│   ├── pipline/            # Training and Prediction pipeline orchestration
│   └── ...
├── app.py                  # Entry point to run the pipeline
├── requirements.txt        # Project dependencies
└── ...
```
