# Project Midterm Report: QUIC Traffic Classification

This document outlines the progress made on the QUIC traffic classification project. It follows the journey from initial setup and baseline modeling to advanced feature engineering, GPU acceleration, and the challenges encountered when scaling the process to handle large, week-long datasets.

## Phase 1: Initial Setup and Baseline Modeling

**Objective:** Establish a baseline for classifying QUIC traffic by training a simple machine learning model on a single day of data.

### Step 1: Data Loading and Initial Scripting
The project began with the goal of training a `DecisionTreeClassifier`. The first step was to create a script, `scripts/load_day.py`, to handle data ingestion from the `cesnet-quic22` dataset. The initial version focused on:
- Loading a single day's data from a `.csv.gz` file using `pandas`.
- Selecting a small, initial set of raw numeric features and the `APP` column as the target label.
- Implementing a basic train/validation split.

### Step 2: Baseline Decision Tree
A corresponding training script, `scripts/train_decision_tree.py`, was created to consume the data from `load_day.py`. It trained a standard `sklearn.tree.DecisionTreeClassifier` and evaluated its accuracy. This provided the first performance benchmark and confirmed the viability of the dataset for application classification.

## Phase 2: Performance and Efficiency Enhancements

**Objective:** Address the bottleneck of slow data loading and preprocessing to enable faster iteration and experimentation.

### Step 1: Caching with Joblib
Reading and preparing features from the large CSV files was time-consuming on every run. To solve this, we implemented a caching layer in `load_day.py` using `joblib`.
- After the feature matrix (X) and label vector (y) were prepared, they were saved to a `datasets/cache/` directory.
- On subsequent runs, the script would check for these cached files and load them directly, bypassing the expensive CSV parsing and feature preparation steps.
- The cache key was designed to be dependent on the input file and the list of features, ensuring that changes to either would correctly invalidate the cache.

### Step 2: Transition to Parquet
While caching helped, the initial data loading from gzipped CSVs remained slow. We identified the Parquet file format as a more efficient, column-oriented storage solution.
- A new utility, `scripts/convert_to_parquet.py`, was developed to convert the raw `.csv.gz` files into `.parquet` files.
- `load_day.py` was then updated to automatically prefer loading a `.parquet` file if one existed, falling back to the CSV otherwise. This dramatically reduced data loading times.
- The conversion script was later enhanced to support parallel processing using `concurrent.futures.ProcessPoolExecutor`, allowing multiple large CSVs to be converted simultaneously, which was crucial for scaling up to weekly data.

## Phase 3: Advanced Modeling and Feature Engineering

**Objective:** Improve model accuracy by using a more powerful ensemble model and enriching the feature set.

### Step 1: From Decision Tree to Random Forest
A single decision tree is prone to overfitting. The logical next step was to implement a Random Forest, which is an ensemble of decision trees.
- We created `scripts/train_random_forest.py` to train a `sklearn.ensemble.RandomForestClassifier`. This provided a more robust CPU-based baseline.

### Step 2: GPU Acceleration with RAPIDS cuML
Training a Random Forest on millions of data points is computationally intensive. To accelerate this, we integrated RAPIDS cuML.
- A new script, `scripts/train_random_forest_cuml.py`, was created.
- This required adding helper functions to convert `pandas` DataFrames into `cudf` DataFrames, as cuML operates on GPU memory.
- The initial results were dramatic, with training times dropping from minutes to seconds.

### Step 3: Comprehensive Feature Engineering
Model performance is heavily dependent on the quality of features. We significantly expanded the feature set in `load_day.py`:
- **Histogram Expansion:** The `PHIST_*` columns, which contained lists of numbers, were parsed into separate features for each bin (e.g., `PHIST_SRC_SIZES_BIN_0`, `PHIST_SRC_SIZES_BIN_1`, etc.).
- **Derived Ratios and Aggregates:** Features like `BYTES_RATIO`, `PACKETS_BALANCE`, `MEAN_PACKET_SIZE_FWD`, and `BYTES_PER_SECOND` were created to capture the dynamics of the flows.
- **Time-Based Features:** The `TIME_FIRST` and `TIME_LAST` timestamps were used to derive features like `START_HOUR`, `START_DAY_OF_WEEK`, and `IS_WEEKEND`.
- **Log Transformations:** To handle skewed distributions, features like `LOG_TOTAL_BYTES` were added.

This expansion made the `engineer_features` function a central part of the preprocessing pipeline.

## Phase 4: Scaling to Week-Long Datasets and Overcoming Challenges

**Objective:** Train the model on an entire week of data (W-2022-47) and test it on another (W-2022-46) to assess generalization and robustness.

### Step 1: Adapting Scripts for Weekly Data
The existing scripts were designed for single-day files. We refactored them to handle collections of files.
- `train_random_forest_cuml.py` was updated to automatically discover all daily Parquet files within a given week's directory.
- `load_day.py` was extended with a `load_cached_week_features_with_labels` function. This function loads the features for each day in a week (reusing the per-day caches) and concatenates them into a single, massive feature matrix, which is then cached itself.

### Step 2: The Memory Wall
When we attempted to train the cuML Random Forest on the full 44 million rows of week W-2022-47, we hit a `MemoryError: std::bad_alloc`. The GPU ran out of memory.
- Our analysis showed that the feature matrix alone required ~11 GB of GPU memory, leaving little room for the model's internal data structures and workspace.
- We experimented with hyperparameters, reducing `n_estimators`, `max_depth`, and `n_bins`. We also introduced `max_batch_size` to control the number of nodes processed in a batch.
- Despite aggressive parameter reduction, the out-of-memory error persisted. The fundamental issue was the sheer size of the input data being copied to the GPU, not the model's complexity itself. This learning was crucial: for datasets that exceed GPU memory, a different training strategy is required.

## Next Steps and Future Vision

The project has successfully built a robust pipeline for feature engineering and model training. However, the memory limitations of the current approach prevent us from training on the full dataset with a single GPU.

My next step is to pivot to a model that supports **online learning**. This approach allows the model to be updated incrementally with new data, rather than requiring the entire dataset to be loaded into memory at once. This aligns perfectly with the ultimate goal of the project: a real-time traffic classification demo.

Potential online learning models to investigate:
1.  **River:** A Python library for online machine learning, offering models like Hoeffding Adaptive Tree which can handle streaming data and concept drift.
2.  **scikit-learn's `partial_fit`:** Models like `SGDClassifier` can be trained incrementally on batches of data.
3.  **XGBoost (Out-of-Core):** While not truly online, XGBoost can train on datasets larger than memory by streaming data from disk.

Once a suitable online model is identified and implemented, the project will move towards building the infrastructure for a real-time demonstration. This will involve capturing live network packets, preprocessing them using the established feature engineering pipeline, and feeding them to the trained model for classification on the fly.
