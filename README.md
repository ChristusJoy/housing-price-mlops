# Housing Price Prediction — An MLOps Tutorial for Beginners

> **Learn MLOps by building a complete, production-style machine learning pipeline from scratch.**
>
> This project predicts housing prices using the Ames Housing dataset. Along the way you will learn how to structure an ML project, build reusable components, orchestrate a training pipeline, and prepare for deployment — all following MLOps best practices.

---

## Table of Contents

1. [What is MLOps and Why Should You Care?](#1-what-is-mlops-and-why-should-you-care)
2. [What This Project Does (High-Level Overview)](#2-what-this-project-does-high-level-overview)
3. [Project Status](#3-project-status)
4. [Architecture & How the Pieces Fit Together](#4-architecture--how-the-pieces-fit-together)
5. [Project Structure — File by File](#5-project-structure--file-by-file)
6. [Prerequisites](#6-prerequisites)
7. [Installation (Step-by-Step)](#7-installation-step-by-step)
8. [Setting Up MongoDB (Your Data Source)](#8-setting-up-mongodb-your-data-source)
9. [Running the Training Pipeline](#9-running-the-training-pipeline)
10. [Deep Dive — Every Pipeline Stage Explained](#10-deep-dive--every-pipeline-stage-explained)
    - [Stage 1: Data Ingestion](#stage-1-data-ingestion)
    - [Stage 2: Data Validation](#stage-2-data-validation)
    - [Stage 3: Data Transformation](#stage-3-data-transformation)
    - [Stage 4: Model Training](#stage-4-model-training)
    - [Stage 5: Model Evaluation (Planned)](#stage-5-model-evaluation-planned)
    - [Stage 6: Model Pusher (Planned)](#stage-6-model-pusher-planned)
11. [Understanding the Supporting Modules](#11-understanding-the-supporting-modules)
    - [Constants](#constants)
    - [Custom Exception Handling](#custom-exception-handling)
    - [Logging](#logging)
    - [Utility Functions](#utility-functions)
    - [Entity Classes (Config & Artifact)](#entity-classes-config--artifact)
12. [Configuration Files Explained](#12-configuration-files-explained)
13. [What Happens When You Run `python app.py`?](#13-what-happens-when-you-run-python-apppy)
14. [How Artifacts Are Organized](#14-how-artifacts-are-organized)
15. [Key MLOps Concepts You Have Learned](#15-key-mlops-concepts-you-have-learned)
16. [Next Steps & Ideas for Practice](#16-next-steps--ideas-for-practice)
17. [Troubleshooting / FAQ](#17-troubleshooting--faq)

---

## 1. What is MLOps and Why Should You Care?

**MLOps** (Machine Learning Operations) is a set of practices that combines Machine Learning, DevOps, and Data Engineering to deploy and maintain ML systems in production **reliably and efficiently**.

In a typical data-science notebook you might:
- Load data → clean it → train a model → look at metrics → done.

That works for learning, but in the real world you need to:

| Concern | What MLOps gives you |
|---|---|
| **Reproducibility** | Every run produces timestamped artifacts so you can go back to any version. |
| **Modularity** | Each step (ingestion, validation, …) is an independent, testable component. |
| **Automation** | A single command triggers the whole pipeline end-to-end. |
| **Validation** | Data is automatically checked against a schema before training. |
| **Logging & Error Handling** | Every action is logged; errors contain file names and line numbers. |
| **Deployment readiness** | The trained model is wrapped with its preprocessor so it can serve predictions immediately. |

This project teaches you all of the above by example.

---

## 2. What This Project Does (High-Level Overview)

```
MongoDB (raw data)
       │
       ▼
┌──────────────────┐
│  Data Ingestion  │  ── Fetches data, saves CSV, splits train/test
└───────┬──────────┘
        ▼
┌──────────────────┐
│ Data Validation  │  ── Checks columns, types, and schema match
└───────┬──────────┘
        ▼
┌──────────────────────┐
│ Data Transformation  │  ── Feature engineering, imputing, scaling, encoding
└───────┬──────────────┘
        ▼
┌──────────────────┐
│  Model Training  │  ── Trains XGBRegressor, evaluates R², saves model
└───────┬──────────┘
        ▼
┌──────────────────┐
│ Model Evaluation │  ── (Planned) Compare with existing model in S3
└───────┬──────────┘
        ▼
┌──────────────────┐
│  Model Pusher    │  ── (Planned) Push accepted model to AWS S3
└──────────────────┘
```

**Target variable:** `SalePrice` (the price a house sold for).

**Algorithm:** XGBoost Regressor with tuned hyperparameters.

---

## 3. Project Status

| Stage | Status |
|---|---|
| Data Ingestion | ✅ Complete |
| Data Validation | ✅ Complete |
| Data Transformation | ✅ Complete |
| Model Training | ✅ Complete |
| Model Evaluation | 🔲 Placeholder (to be implemented) |
| Model Pusher | 🔲 Placeholder (to be implemented) |
| Prediction Pipeline / API | 🔲 Placeholder (to be implemented) |

---

## 4. Architecture & How the Pieces Fit Together

```
housing-price-mlops/
│
├── app.py                          ← Entry point: "Run the pipeline"
│
├── src/
│   ├── constants/__init__.py       ← All magic numbers & paths live here
│   ├── exception/__init__.py       ← Custom exception with file + line info
│   ├── logger/__init__.py          ← Rotating file + console logger
│   ├── utils/main_utils.py         ← YAML, pickle (dill), numpy I/O helpers
│   │
│   ├── configuration/
│   │   └── mongo_db_connection.py  ← Singleton MongoDB client
│   │
│   ├── data_access/
│   │   └── project_data.py         ← Exports a MongoDB collection → DataFrame
│   │
│   ├── entity/
│   │   ├── config_entity.py        ← @dataclass configs for every stage
│   │   ├── artifact_entity.py      ← @dataclass outputs for every stage
│   │   └── estimator.py            ← MyModel wraps preprocessor + model
│   │
│   ├── components/                 ← One component per pipeline stage
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   │
│   └── pipline/
│       └── training_pipeline.py    ← Orchestrates components in order
│
├── config/
│   └── schema.yaml                 ← Ground truth: expected columns & types
│
└── artifact/                       ← Auto-generated per run (timestamped)
```

**Key design patterns:**
- **Config → Component → Artifact**: Each pipeline stage receives a *config* dataclass, does its work, and returns an *artifact* dataclass. The artifact of one stage becomes the input for the next.
- **Singleton MongoDB client**: Only one connection is opened and shared across the application.
- **Timestamped artifact directories**: Every pipeline run creates a new folder like `artifact/01_20_2026_03_13_39/` so nothing is ever overwritten.

---

## 5. Project Structure — File by File

```
housing-price-mlops/
│
│── app.py                      # Entry point — creates TrainPipeline and calls run_pipeline()
│── demo.py                     # Scratch file used during development for testing logging/exceptions
│── template.py                 # One-time script that generates the initial directory tree
│── setup.py                    # Makes the `src` package installable with pip (pip install -e .)
│── pyproject.toml              # Modern Python project metadata
│── requirements.txt            # All pip dependencies
│── Dockerfile                  # (Empty) Placeholder for containerized deployment
│
├── config/
│   ├── model.yaml              # (Empty) Reserved for model hyperparameter overrides
│   └── schema.yaml             # Defines every column, its dtype, and which are numerical/categorical
│
├── notebook/
│   ├── data.csv                # Raw dataset for exploratory analysis
│   ├── model.ipynb             # Jupyter notebook — EDA and model experiments
│   └── mongodb_text.ipynb      # Notebook for testing MongoDB connectivity
│
├── src/                        # ← All production source code lives here
│   ├── __init__.py
│   │
│   ├── constants/__init__.py           # Central place for every constant (paths, params, thresholds)
│   ├── exception/__init__.py           # MyException — rich error messages with file + line number
│   ├── logger/__init__.py              # Rotating log files + console output
│   │
│   ├── configuration/
│   │   ├── mongo_db_connection.py      # MongoDBClient (singleton pattern)
│   │   └── aws_connection.py           # (Placeholder) AWS credentials
│   │
│   ├── cloud_storage/
│   │   └── aws_storage.py              # (Placeholder) S3 upload/download helpers
│   │
│   ├── data_access/
│   │   └── project_data.py             # ProjectData — MongoDB collection → pandas DataFrame
│   │
│   ├── entity/
│   │   ├── config_entity.py            # @dataclass for each pipeline stage's configuration
│   │   ├── artifact_entity.py          # @dataclass for each pipeline stage's output
│   │   ├── estimator.py                # MyModel — wraps preprocessing + model for inference
│   │   └── s3_estimator.py             # (Placeholder) For loading models from S3
│   │
│   ├── components/
│   │   ├── data_ingestion.py           # Fetch data from MongoDB, save CSV, train/test split
│   │   ├── data_validation.py          # Check columns, dtypes against schema.yaml
│   │   ├── data_transformation.py      # Feature eng., imputing, scaling, encoding
│   │   ├── model_trainer.py            # Train XGBRegressor, compute R², save model
│   │   ├── model_evaluation.py         # (Placeholder) Compare new vs. production model
│   │   └── model_pusher.py             # (Placeholder) Push model to AWS S3
│   │
│   ├── pipline/
│   │   ├── training_pipeline.py        # TrainPipeline — runs all stages in order
│   │   └── prediction_pipeline.py      # (Placeholder) Serve predictions via API
│   │
│   └── utils/
│       └── main_utils.py               # read/write YAML, save/load objects (dill), save/load numpy
│
├── artifact/                           # Generated at runtime — one timestamped folder per run
│   └── MM_DD_YYYY_HH_MM_SS/
│       ├── data_ingestion/
│       │   ├── feature_store/data.csv
│       │   └── ingested/
│       │       ├── train.csv
│       │       └── test.csv
│       ├── data_validation/
│       │   └── report.yaml
│       ├── data_transformation/
│       │   ├── transformed/
│       │   │   ├── train.npy
│       │   │   └── test.npy
│       │   └── transformed_object/
│       │       └── preprocessing.pkl
│       └── model_trainer/
│           └── trained_model/
│               └── model.pkl
│
└── logs/                               # Rotating log files (5 MB max, 3 backups)
    └── MM_DD_YYYY_HH_MM_SS.log
```

---

## 6. Prerequisites

| Requirement | Why |
|---|---|
| **Python 3.8+** | Language runtime (3.10+ recommended) |
| **pip** | Package installer |
| **MongoDB Atlas account** (free tier is fine) | The raw housing data is stored in a MongoDB collection. The pipeline pulls it from there. |
| **Git** | To clone the repository |
| **(Optional) AWS account** | Needed only for the Model Evaluation / Pusher stages, which are not yet implemented. |

> **Don't have MongoDB set up yet?** See [Section 8](#8-setting-up-mongodb-your-data-source) for a full walkthrough.

---

## 7. Installation (Step-by-Step)

### 7.1 Clone the repository

```bash
git clone <repository-url>
cd housing-price-mlops
```

### 7.2 Create a virtual environment

```bash
python -m venv venv
```

Activate it:

```bash
# Linux / macOS
source venv/bin/activate

# Windows (Command Prompt)
venv\Scripts\activate

# Windows (PowerShell)
venv\Scripts\Activate.ps1
```

> **What is a virtual environment?** It is an isolated Python installation. Packages you install inside it won't affect your system Python.

### 7.3 Install dependencies

```bash
pip install -r requirements.txt
```

This installs everything the project needs, including:

| Package | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `scikit-learn` | Preprocessing, metrics, train/test split |
| `xgboost` | Gradient-boosted tree model (the algorithm we train) |
| `pymongo`, `certifi` | Connect to MongoDB |
| `dill` | Serialize (save) Python objects to disk |
| `PyYAML` | Read/write YAML config files |
| `from_root` | Resolve the project root directory reliably |
| `python-dotenv` (`dotenv`) | Load environment variables from a `.env` file |
| `fastapi`, `uvicorn`, `jinja2` | (For future prediction API) |
| `boto3`, `mypy-boto3-s3` | (For future AWS S3 integration) |
| `-e .` | Installs the local `src` package in editable/development mode |

> **What does `-e .` mean?** It runs `pip install --editable .` which reads `setup.py` and makes the `src` package importable from anywhere in the project without needing `sys.path` hacks.

### 7.4 Create the `.env` file

Create a file named `.env` in the project root:

```bash
# .env
MONGODB_URL="mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority"
```

Replace `<username>`, `<password>`, and the cluster address with your actual MongoDB Atlas credentials.

> **Security tip:** Never commit `.env` to Git. Add it to your `.gitignore`.

---

## 8. Setting Up MongoDB (Your Data Source)

In a real MLOps workflow, your data lives in a database — not a local CSV. This project uses **MongoDB Atlas** (a free cloud-hosted MongoDB service).

### 8.1 Create a free Atlas cluster

1. Go to [https://www.mongodb.com/cloud/atlas](https://www.mongodb.com/cloud/atlas) and sign up.
2. Create a **free shared cluster** (M0 tier).
3. Under **Database Access**, create a database user with a username and password.
4. Under **Network Access**, add your IP address (or `0.0.0.0/0` for development).
5. Click **Connect → Drivers → Python** and copy the connection string.

### 8.2 Upload the housing dataset

The raw data is in `notebook/data.csv`. You can import it into MongoDB using the `mongodb_text.ipynb` notebook, or manually:

```python
import pandas as pd
import pymongo
import json

# Connect
client = pymongo.MongoClient("mongodb+srv://<your-connection-string>")
db = client["housing_price"]
collection = db["housing_price_data"]

# Load CSV and insert
df = pd.read_csv("notebook/data.csv")
records = json.loads(df.to_json(orient="records"))
collection.insert_many(records)

print(f"Inserted {len(records)} documents.")
```

### 8.3 Verify the `.env` file

Make sure `.env` contains your MongoDB URL (see [Section 7.4](#74-create-the-env-file)).

The `src/constants/__init__.py` file uses `python-dotenv` to load it:

```python
from dotenv import load_dotenv
load_dotenv()
MONGODB_URL_KEY = "MONGODB_URL"
```

And `src/configuration/mongo_db_connection.py` reads it with:
```python
mongo_db_url = os.getenv(MONGODB_URL_KEY)
```

---

## 9. Running the Training Pipeline

Once setup is complete, just run:

```bash
python app.py
```

That's it! The script does the following in sequence:

```
Data Ingestion → Data Validation → Data Transformation → Model Training
```

You'll see logs in the terminal and in the `logs/` folder. The trained model and all intermediate data will be saved in a new timestamped folder inside `artifact/`.

---

## 10. Deep Dive — Every Pipeline Stage Explained

### Stage 1: Data Ingestion

**File:** `src/components/data_ingestion.py`

**What it does:**
1. Connects to MongoDB and fetches the entire `housing_price_data` collection.
2. Converts it to a pandas DataFrame.
3. Removes the MongoDB `_id` column and replaces `"na"` strings with `NaN`.
4. Saves the full dataset as `data.csv` in the **feature store**.
5. Splits the data into **train** (75%) and **test** (25%) sets using `sklearn.model_selection.train_test_split`.
6. Saves `train.csv` and `test.csv`.

**Config used:** `DataIngestionConfig` (from `src/entity/config_entity.py`)
```python
@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str          # artifact/<timestamp>/data_ingestion
    feature_store_file_path: str     # .../feature_store/data.csv
    training_file_path: str          # .../ingested/train.csv
    testing_file_path: str           # .../ingested/test.csv
    train_test_split_ratio: float    # 0.25 (25% test)
    collection_name: str             # "housing_price_data"
```

**Artifact produced:** `DataIngestionArtifact`
```python
@dataclass
class DataIngestionArtifact:
    trained_file_path: str   # path to train.csv
    test_file_path: str      # path to test.csv
```

**Key takeaway for beginners:** In MLOps, data ingestion is automated and versioned. You don't manually download CSVs — the pipeline pulls from the source of truth (database) every time.

---

### Stage 2: Data Validation

**File:** `src/components/data_validation.py`

**What it does:**
1. Reads the train and test CSV files produced by Data Ingestion.
2. Loads the expected schema from `config/schema.yaml`.
3. **Validates column count** — does the dataset have the right number of columns?
4. **Validates column existence** — are all expected numerical and categorical columns present?
5. Generates a validation report (JSON) saved to `report.yaml`.
6. If validation fails, the error message is propagated so downstream stages can halt.

**Why this matters:**
Imagine your MongoDB data changes (someone adds/removes a column). Without validation, the model would either crash during training or silently produce garbage predictions. Data validation is your safety net.

**Config used:** `DataValidationConfig`
```python
@dataclass
class DataValidationConfig:
    data_validation_dir: str              # artifact/<timestamp>/data_validation
    validation_report_file_path: str      # .../report.yaml
```

**Artifact produced:** `DataValidationArtifact`
```python
@dataclass
class DataValidationArtifact:
    validation_status: bool      # True if all checks pass
    message: str                 # Empty string if valid, error details otherwise
    validation_report_file_path: str
```

---

### Stage 3: Data Transformation

**File:** `src/components/data_transformation.py`

**What it does:**
1. Reads train/test CSVs.
2. Separates features from the target (`SalePrice`).
3. **Feature engineering** — creates new features:
   - `HouseAge` = `YrSold` - `YearBuilt`
   - `RemodAge` = `YrSold` - `YearRemodAdd`
   - `TotalBathrooms` = `FullBath` + 0.5 × `HalfBath` + `BsmtFullBath` + 0.5 × `BsmtHalfBath`
   - `TotalSF` = `GrLivArea` + `TotalBsmtSF`
   - `HasGarage` = 1 if `GarageArea` > 0
   - `HasBasement` = 1 if `TotalBsmtSF` > 0
4. Builds a **preprocessing pipeline** using scikit-learn's `ColumnTransformer`:
   - **Numerical columns:** Median imputation → Standard scaling
   - **Categorical columns:** Constant imputation (`"missing"`) → One-hot encoding
5. Fits the preprocessor on training data, transforms both train and test.
6. Saves transformed data as `.npy` arrays and the preprocessor object as `preprocessing.pkl`.

**Why a preprocessing pipeline?** If you apply transformations manually you'll inevitably introduce a *train-serve skew* — the preprocessing at prediction time won't match training. By saving the fitted `ColumnTransformer` as a pickle, you guarantee the exact same transformations are applied later.

**Config used:** `DataTransformationConfig`
```python
@dataclass
class DataTransformationConfig:
    transformed_train_file_path: str     # .../transformed/train.npy
    transformed_test_file_path: str      # .../transformed/test.npy
    transformed_object_file_path: str    # .../transformed_object/preprocessing.pkl
```

**Artifact produced:** `DataTransformationArtifact`
```python
@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str
```

---

### Stage 4: Model Training

**File:** `src/components/model_trainer.py`

**What it does:**
1. Loads the transformed `.npy` arrays.
2. Splits into `X_train, y_train, X_test, y_test` (target is the last column).
3. Trains an **XGBRegressor** with these hyperparameters:

| Parameter | Value | Meaning |
|---|---|---|
| `n_estimators` | 200 | Number of boosting rounds |
| `max_depth` | 6 | Maximum tree depth |
| `learning_rate` | 0.05 | Step size shrinkage |
| `subsample` | 0.8 | Fraction of samples per tree |
| `colsample_bytree` | 0.8 | Fraction of features per tree |
| `min_child_weight` | 1 | Minimum sum of weights in a child |
| `gamma` | 0 | Minimum loss reduction for a split |
| `reg_alpha` | 0 | L1 regularization |
| `reg_lambda` | 1 | L2 regularization |
| `random_state` | 101 | For reproducibility |

4. Evaluates using **R² score** on the test set.
5. Wraps the fitted **preprocessor + model** into a `MyModel` object (see `src/entity/estimator.py`), so a single `.predict(raw_dataframe)` call handles everything.
6. Saves as `model.pkl`.

**Why wrap the preprocessor and model together?** This is a critical MLOps pattern. During inference you receive *raw* data (unscaled, with missing values). The `MyModel` class applies the exact same preprocessing before calling `model.predict()`:

```python
class MyModel:
    def predict(self, dataframe):
        transformed = self.preprocessing_object.transform(dataframe)
        return self.trained_model_object.predict(transformed)
```

**Config used:** `ModelTrainerConfig`
```python
@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str    # .../trained_model/model.pkl
    expected_accuracy: float        # 0.5 (minimum R² to accept the model)
    # ... plus all XGBoost hyperparameters
```

**Artifact produced:** `ModelTrainerArtifact`
```python
@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    metric_artifact: RegressionMetricArtifact  # contains r2_score
```

---

### Stage 5: Model Evaluation (Planned)

**File:** `src/components/model_evaluation.py` (currently a placeholder)

**What it will do:**
- Load the **currently deployed model** from AWS S3.
- Compare its R² against the newly trained model.
- Only accept the new model if it improves by at least `MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE` (0.02).
- Return a `ModelEvaluationArtifact` with `is_model_accepted: bool`.

---

### Stage 6: Model Pusher (Planned)

**File:** `src/components/model_pusher.py` (currently a placeholder)

**What it will do:**
- If the new model was accepted, upload `model.pkl` to an AWS S3 bucket (`my-model-mlopsproj`).
- This makes the model available for a production prediction API.

---

## 11. Understanding the Supporting Modules

### Constants

**File:** `src/constants/__init__.py`

This is the **single source of truth** for every configurable value in the project: database names, file names, directory names, hyperparameters, and thresholds.

Why centralize constants?
- Changing a value in one place updates the entire pipeline.
- No magic strings scattered across files.

Key constants:

```python
DATABASE_NAME = "housing_price"              # MongoDB database
COLLECTION_NAME = "housing_price_data"       # MongoDB collection
TARGET_COLUMN = "SalePrice"                  # What we're predicting
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO = 0.25 # 75/25 split
MODEL_TRAINER_N_ESTIMATORS = 200             # XGBoost trees
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE = 0.02
```

---

### Custom Exception Handling

**File:** `src/exception/__init__.py`

The `MyException` class wraps Python's built-in `Exception` to include:
- The **file name** where the error occurred.
- The **line number**.
- The original error message.

Example output:
```
Error occurred in python script: [src/components/data_ingestion.py] at line number [42]: connection refused
```

This makes debugging in production much faster than a generic traceback.

**Usage pattern (you'll see this everywhere):**
```python
try:
    # some code
except Exception as e:
    raise MyException(e, sys) from e
```

---

### Logging

**File:** `src/logger/__init__.py`

Configures Python's `logging` module with:
- **Rotating file handler** — log files are capped at 5 MB and rotated (3 backups kept).
- **Console handler** — prints INFO+ messages to your terminal.
- **Timestamp format:** `[ 2026-01-20 03:12:06,123 ] root - INFO - message`

Log files are saved in `logs/` with names like `01_20_2026_03_12_06.log`.

**Why rotating logs?** Without rotation, a long-running pipeline could fill your disk. Rotation ensures only the most recent ~20 MB of logs are kept.

---

### Utility Functions

**File:** `src/utils/main_utils.py`

Reusable I/O helpers used by multiple components:

| Function | What it does |
|---|---|
| `read_yaml_file(path)` | Reads a YAML file and returns a Python dict |
| `write_yaml_file(path, content)` | Writes a dict to a YAML file |
| `save_object(path, obj)` | Serializes any Python object to disk using `dill` |
| `load_object(path)` | Deserializes (loads) an object from disk |
| `save_numpy_array_data(path, array)` | Saves a numpy array as `.npy` |
| `load_numpy_array_data(path)` | Loads a `.npy` file |

**Why `dill` instead of `pickle`?** `dill` can serialize a wider range of Python objects (lambdas, nested functions, etc.) which makes it more robust for ML pipelines.

---

### Entity Classes (Config & Artifact)

**Files:** `src/entity/config_entity.py` and `src/entity/artifact_entity.py`

These use Python's `@dataclass` decorator to define clean, typed data structures.

**Config entities** = *inputs* to a pipeline stage (where to find/save things, hyperparameters).
**Artifact entities** = *outputs* of a pipeline stage (paths to generated files, metrics).

This separation makes the pipeline:
- **Testable** — you can mock configs easily.
- **Readable** — looking at a config class tells you exactly what a component needs.
- **Type-safe** — your IDE can catch typos.

---

## 12. Configuration Files Explained

### `config/schema.yaml`

Defines the **expected shape** of the dataset. It has four sections:

| Section | Purpose |
|---|---|
| `columns` | Full list of every column with its expected dtype (`int`, `float`, `category`) |
| `numerical_columns` | Subset of columns treated as numerical features |
| `categorical_columns` | Subset of columns treated as categorical features |
| `drop_columns` | Columns to exclude (e.g., `Id`) |
| `num_features` | Numerical features used in the transformation pipeline |

The Data Validation component reads this file to check whether the ingested data matches expectations.

### `config/model.yaml`

Currently empty — reserved for future use (e.g., hyperparameter overrides, model selection config).

---

## 13. What Happens When You Run `python app.py`?

Here is the exact sequence of events:

```
1.  app.py imports TrainPipeline and calls run_pipeline()
2.  TrainPipeline.__init__() creates config objects for all stages
3.  A timestamp is generated (e.g., "01_20_2026_03_13_39")
4.  artifact/01_20_2026_03_13_39/ directory is created
│
├─ 5.  start_data_ingestion()
│      ├── Connects to MongoDB (reads MONGODB_URL from .env)
│      ├── Fetches housing_price_data collection → DataFrame
│      ├── Drops _id column, replaces "na" → NaN
│      ├── Saves data.csv → artifact/.../feature_store/
│      ├── train_test_split(75/25)
│      ├── Saves train.csv, test.csv → artifact/.../ingested/
│      └── Returns DataIngestionArtifact
│
├─ 6.  start_data_validation(data_ingestion_artifact)
│      ├── Reads train.csv and test.csv
│      ├── Loads config/schema.yaml
│      ├── Checks: column count matches? All columns present?
│      ├── Writes report.yaml → artifact/.../data_validation/
│      └── Returns DataValidationArtifact (validation_status=True/False)
│
├─ 7.  start_data_transformation(data_ingestion_artifact, data_validation_artifact)
│      ├── Aborts if validation_status is False
│      ├── Reads train.csv and test.csv
│      ├── Creates engineered features (HouseAge, TotalSF, etc.)
│      ├── Builds ColumnTransformer (impute + scale numerics, impute + one-hot categoricals)
│      ├── fit_transform on train, transform on test
│      ├── Saves train.npy, test.npy, preprocessing.pkl
│      └── Returns DataTransformationArtifact
│
├─ 8.  start_model_trainer(data_transformation_artifact)
│      ├── Loads train.npy, test.npy
│      ├── Trains XGBRegressor (200 estimators, depth 6, lr 0.05, ...)
│      ├── Computes R² score on test set
│      ├── Wraps preprocessor + model → MyModel object
│      ├── Saves model.pkl → artifact/.../model_trainer/trained_model/
│      └── Returns ModelTrainerArtifact
│
└─ 9.  Pipeline complete! All artifacts saved.
```

---

## 14. How Artifacts Are Organized

Every run gets its own timestamped directory. Nothing is ever overwritten.

```
artifact/
└── 01_20_2026_03_13_39/          ← one complete pipeline run
    ├── data_ingestion/
    │   ├── feature_store/
    │   │   └── data.csv          ← full dataset from MongoDB
    │   └── ingested/
    │       ├── train.csv         ← 75% of the data
    │       └── test.csv          ← 25% of the data
    │
    ├── data_validation/
    │   └── report.yaml           ← {"validation_status": true, "message": ""}
    │
    ├── data_transformation/
    │   ├── transformed/
    │   │   ├── train.npy         ← preprocessed training features + target
    │   │   └── test.npy          ← preprocessed test features + target
    │   └── transformed_object/
    │       └── preprocessing.pkl ← fitted ColumnTransformer
    │
    └── model_trainer/
        └── trained_model/
            └── model.pkl         ← MyModel (preprocessor + XGBRegressor)
```

**Why timestamp each run?**
- You can compare models from different runs side by side.
- If a new model is worse, you can roll back to a previous artifact.
- It's a simple form of **experiment tracking** (more advanced tools include MLflow, Weights & Biases, etc.).

---

## 15. Key MLOps Concepts You Have Learned

By going through this project, you've been exposed to these MLOps fundamentals:

| Concept | Where you saw it |
|---|---|
| **Pipeline orchestration** | `training_pipeline.py` chains stages together |
| **Component-based architecture** | Each stage is an independent class in `components/` |
| **Config-driven design** | All parameters come from `constants/` and `config/schema.yaml` |
| **Artifact management** | Timestamped `artifact/` directories |
| **Data validation** | Schema checks before training |
| **Feature store** | `feature_store/data.csv` captures the raw ingested snapshot |
| **Preprocessing persistence** | `preprocessing.pkl` ensures train-serve consistency |
| **Model wrapping** | `MyModel` bundles preprocessing + model for easy deployment |
| **Structured logging** | Rotating file logs with consistent format |
| **Custom exceptions** | Rich error messages with file + line info |
| **Environment variables** | Secrets (DB URL) kept out of source code |
| **Editable installs** | `pip install -e .` for clean imports |

---

## 16. Next Steps & Ideas for Practice

Here are things you can try to deepen your MLOps knowledge:

1. **Implement Model Evaluation** — Load a previous model, compare R² scores, only accept improvements.
2. **Implement Model Pusher** — Upload the accepted model to AWS S3 (use the placeholder in `model_pusher.py`).
3. **Build the Prediction API** — Use FastAPI (already in requirements) to serve predictions:
   ```python
   @app.post("/predict")
   def predict(features: dict):
       model = load_object("artifact/.../model.pkl")
       df = pd.DataFrame([features])
       prediction = model.predict(df)
       return {"predicted_price": prediction[0]}
   ```
4. **Containerize with Docker** — Fill in the empty `Dockerfile` to make the project portable.
5. **Add CI/CD** — Use GitHub Actions to run the pipeline on every push.
6. **Add experiment tracking** — Integrate MLflow to log hyperparameters, metrics, and models.
7. **Add data drift detection** — Extend `data_validation.py` to detect statistical drift between training and new data.
8. **Hyperparameter tuning** — Use `config/model.yaml` to define a search space and add grid/random search.

---

## 17. Troubleshooting / FAQ

### `ModuleNotFoundError: No module named 'src'`
You forgot to install the package in editable mode. Run:
```bash
pip install -e .
```

### `Environment variable 'MONGODB_URL' is not set`
Create a `.env` file in the project root with your MongoDB connection string. See [Section 7.4](#74-create-the-env-file).

### `pymongo.errors.ServerSelectionTimeoutError`
- Check that your MongoDB Atlas cluster is running.
- Verify your IP is whitelisted under **Network Access** in Atlas.
- Make sure the connection string in `.env` is correct.

### `ModuleNotFoundError: No module named 'dotenv'`
The package name is `python-dotenv`. Install it with:
```bash
pip install python-dotenv
```
(It's already listed in `requirements.txt` as `dotenv` — if this causes issues, edit it to `python-dotenv`.)

### The pipeline says validation failed
Check `artifact/<timestamp>/data_validation/report.yaml` for the error message. Common causes:
- Columns were renamed or removed in the database.
- The schema in `config/schema.yaml` doesn't match the actual data.

### Out of memory during transformation
The one-hot encoding of many categorical columns can produce a very wide matrix. Consider:
- Using `sparse_output=True` in `OneHotEncoder`.
- Reducing cardinality by grouping rare categories.

### Where is the trained model saved?
Inside the latest timestamped artifact folder:
```
artifact/<latest_timestamp>/model_trainer/trained_model/model.pkl
```

### How do I use the trained model for predictions?
```python
from src.utils.main_utils import load_object
import pandas as pd

model = load_object("artifact/<timestamp>/model_trainer/trained_model/model.pkl")
sample = pd.read_csv("notebook/data.csv").drop(columns=["SalePrice", "Id"]).head(1)
prediction = model.predict(sample)
print(f"Predicted price: ${prediction[0]:,.2f}")
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

> *Built with ❤️ as a learning resource for the MLOps community.*
