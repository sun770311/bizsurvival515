# Pipeline Overview

The pipeline performs the following steps:

1. Data ingestion
2. Preprocessing and feature construction
3. Predictive modeling
4. Survival modeling
5. Model inspection and evaluation
6. GeoJSON generation for visualization

---

# Data Ingestion

Two ingestion jobs fetch data from **NYC Open Data APIs** using the Socrata API and store them in **Delta Lake tables**. These jobs run in a **Databricks environment using PySpark**.

## 311 Service Requests Job

This job fetches new NYC 311 complaints and merges them into a Delta Lake table.

Key steps:

1. Determine the most recent `created_date` in the existing Delta table
2. Query the Socrata API for complaints newer than that timestamp
3. Normalize schema and convert timestamp fields
4. Merge new rows into the Delta table

Dataset ID:

erm2-nwe9

Table destination:

data_515.default.311_service_requests

Primary merge key:

unique_key

The job performs **incremental ingestion**, ensuring only new complaint records are retrieved and merged.

---

## Issued Business Licenses Job

This job fetches new NYC business licenses and merges them into a Delta Lake table.

Key steps:

1. Determine the most recent `license_creation_date`
2. Query the Socrata API for newer licenses
3. Normalize schema and timestamp columns
4. Merge new rows into the Delta table

Dataset ID:

w7w3-xahh

Table destination:

data_515.default.issued_licenses

Primary merge key:

license_nbr

---

# Exporting Datasets

The ingestion jobs populate Delta Lake tables in Databricks. These tables are exported to CSV format for use in the modeling pipeline.

The full datasets (CSV format) used in the project are available here:

**Full Business License Dataset**
[https://drive.google.com/file/d/1l9bIxhXUNT4h9UXLxL6V9WMWgONch0W5/view?usp=sharing](https://drive.google.com/file/d/1l9bIxhXUNT4h9UXLxL6V9WMWgONch0W5/view?usp=sharing)

**Full 311 Service Request Dataset**
[https://drive.google.com/file/d/1cDY8SsRB9DRBTSehNyLAlemB6Ch2mR6l/view?usp=sharing](https://drive.google.com/file/d/1cDY8SsRB9DRBTSehNyLAlemB6Ch2mR6l/view?usp=sharing)

These files serve as inputs to the preprocessing stage of the pipeline and correspond to dataset snapshots from March 8, 2026. The pipeline was run in Google Colab connected to an Nvidia G4 GPU runtime.

---

# Pipeline Components

The modeling pipeline is implemented in the `pipeline/` directory.

```markdown
pipeline/
├── preprocess.py
├── logistic.py
├── cox.py
├── mapbox.py
├── inspect_logistic.py
├── inspect_cox.py
├── run_pipeline.py
└── utils.py
```

Each module performs a specific role in the pipeline.

---

# Preprocessing

The preprocessing module converts the raw business license and 311 complaint datasets into the joined monthly panel used by the downstream models.

It performs the following steps:

* cleans and standardizes the raw license and 311 records
* parses dates, coordinates, identifiers, and category labels
* expands each business license into monthly observations across its active period
* builds business-level category indicators
* spatially joins 311 complaints to nearby businesses within a fixed radius
* aggregates monthly complaint counts and complaint-type features
* computes representative business coordinates and assigns location clusters
* merges business activity, complaint activity, and location features into one final dataset

The output is a monthly business panel in which each row represents a **business-month** observation and includes business, complaint, and location context.

---

# Logistic Modeling

The logistic modeling module estimates whether a business survives a fixed time horizon, defined in this project as **three-year survival**.

Its workflow is:

* build a **business-level training dataset** from the joined monthly panel
* summarize each eligible business using features aggregated from its first observed year
* assign a binary target indicating whether the business survived the defined time horizon
* remove redundant and leakage-prone columns
* filter near-constant predictors
* address class imbalance through oversampling
* split the balanced dataset into training and testing sets
* fit a regularized logistic regression pipeline with imputation and scaling
* evaluate predictive performance on a held-out test split

This stage treats the problem as:

* given a business’s early observed characteristics,
* estimate the **probability that it remains open after a fixed period**

Outputs include:

* trained logistic regression pipeline
* retained and dropped feature lists
* coefficient summary table
* balanced modeling dataset
* train/test split exports
* evaluation metrics such as accuracy, ROC AUC, confusion matrix, and classification summaries

---

# Cox Survival Modeling

The Cox modeling module estimates **business closure risk over time** using survival analysis.

It produces two complementary survival datasets and models:

### Time-Varying Survival Setting

In the time-varying setting:

* each business contributes multiple monthly rows
* predictors are allowed to change over time
* the model relates changing business and neighborhood conditions to the hazard of closure

This representation is useful when complaint exposure and other signals evolve over the life of the business.

### Standard Survival Setting

In the standard survival setting:

* each business contributes one row
* each row contains a fixed covariate profile and an observed duration
* the model estimates how baseline characteristics relate to closure risk

### Shared modeling steps

For both settings, the module:

* constructs the appropriate survival dataset
* removes redundant and near-constant predictors
* standardizes retained predictors
* fits penalized Cox models
* exports interpretable coefficient summaries and artifacts

Outputs include:

* trained time-varying Cox artifacts
* trained standard Cox artifacts
* scalers
* retained and dropped feature lists
* coefficient summary CSVs

Together, these models estimate the **risk of closure over time** from both changing monthly signals and one-row-per-business survival profiles.

---

# Logistic Inspection

The logistic inspection module helps interpret the fixed-horizon survival model.

Its role is to:

* summarize the fitted classifier
* highlight the most influential predictors
* examine coefficient direction and magnitude
* connect model behavior back to the project’s business survival questions

This inspection layer helps explain which early business and neighborhood characteristics are most associated with higher or lower estimated survival probability.

---

# Cox Inspection

The Cox inspection module helps interpret the survival models.

It focuses on:

* coefficient summaries
* relative effect sizes
* direction of association with closure risk
* hazard-oriented interpretation of predictors

Because the project includes both standard and time-varying survival models, this inspection layer helps compare how:

* static business characteristics
* evolving neighborhood conditions

relate to business closure risk over time.

---

# GeoJSON Generation

The GeoJSON module prepares the map-ready dataset used in the Streamlit application.

It performs the following tasks:

* loads the joined business-month panel and license metadata
* validates required fields
* cleans business identifiers, borough values, coordinates, and date fields
* filters licenses to valid NYC boroughs and NYC geographic bounds
* aggregates each business’s complaint activity and most recent observed month
* determines whether each business is active based on a cutoff date
* builds structured license metadata records for each business
* combines summary information with business coordinates
* exports the result as a GeoJSON FeatureCollection

Each GeoJSON feature represents a business and contains properties such as:

* business identifier
* active/inactive status
* last observed month
* complaint totals
* license count
* license metadata records

This output supports the Mapbox-based visualization in the Streamlit app.

---

# Pipeline Orchestration

The full pipeline runner coordinates the end-to-end workflow.

Its sequence is:

1. build the joined monthly business panel
2. run the logistic survival modeling pipeline
3. run the Cox survival modeling pipelines
4. generate the GeoJSON output for the map interface

At the end of execution, it outputs a summary containing:

* input file paths
* output artifact locations
* modeling run metadata

This orchestration layer is what turns the raw license and 311 datasets into the final artifacts used by the Streamlit application.

---

# Shared Utilities

The utilities module provides common functionality used across the pipeline.

It includes shared support for:

* study window constants
* variance-threshold defaults
* dataset loading and validation
* artifact serialization and saving
* reusable argument handling
* feature-selection metadata containers

This shared layer helps keep preprocessing, classification, survival modeling, and artifact generation consistent and reproducible.

