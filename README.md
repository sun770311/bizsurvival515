# Open or Closed? NYC Business Survival

[![Build and
Tests](https://github.com/sun770311/bizsurvival515/actions/workflows/build_tests.yml/badge.svg)](https://github.com/sun770311/bizsurvival515/actions/workflows/build_tests.yml)\
[![Coverage
Status](https://coveralls.io/repos/github/sun770311/bizsurvival515/badge.svg)](https://coveralls.io/github/sun770311/bizsurvival515)

## Overview

New York City generates millions of public service requests each year.
These complaints reflect real-world conditions such as sanitation
issues, noise disturbances, infrastructure problems, and other
neighborhood signals.

At the same time, thousands of businesses open and close across the
city.

This project explores an important urban question:

**Do neighborhood signals captured in city service complaints relate to
business survival?**

We built a data platform that integrates NYC Open Data sources,
processes complaint datasets, links them with business licensing
records, and exposes insights through an interactive analytical
application.

The system combines:

-   data pipelines
-   predictive modeling
-   survival analysis
-   geospatial visualization

to help explore **patterns associated with business survival in New York
City.**

------------------------------------------------------------------------

# Team Members

-   Aaron Lee
-   Hannah Sun
-   Juan Pablo Reyes Martinez
-   Pavankumar Suresh
-   Sreeraj Parakkat

Project Type: **Tool Project**

------------------------------------------------------------------------

# Research Questions

This project investigates how urban environmental signals relate to
business survival.

Key questions include:

-   Can patterns in **311 complaints and local activity** help predict
    whether a business remains open?
-   What **neighborhood and environmental factors** correlate with
    business survival versus closure?
-   How do survival patterns vary across **boroughs, business
    categories, and time periods**?

------------------------------------------------------------------------

# Product Overview

The final product is an **interactive analytical tool** that allows
users to explore relationships between city complaints and business
outcomes.

The application enables users to:

-   visualize businesses across NYC
-   analyze complaint patterns around businesses
-   explore survival probabilities
-   inspect model outputs and coefficients
-   investigate geographic patterns in complaints

The system combines **machine learning models, survival analysis, and
interactive geospatial visualization**.

------------------------------------------------------------------------

# System Architecture

The platform integrates multiple components to ingest, process, model,
and visualize data.

    NYC Open Data APIs
            │
            ▼
    Data Ingestion Pipelines (Python + Notebooks)
            │
            ▼
    Preprocessing and Feature Engineering
            │
            ▼
    Machine Learning & Survival Models
            │
            ▼
    Model Artifacts and Processed Datasets
            │
            ▼
    Streamlit Analytical Application
            │
            ▼
    Mapbox Interactive Visualization
![Architecture diagram](./images/Architecture_Diagram.png)

------------------------------------------------------------------------

# Data Sources

### NYC Issued Business Licenses

https://data.cityofnewyork.us/Business/Issued-Licenses/w7w3-xahh/about_data

Contains information about licensed businesses in New York City
including:

-   license identifiers
-   business type
-   location information
-   issuance and expiration data

### NYC 311 Service Requests

https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2020-to-Present/erm2-nwe9/about_data

Contains public service complaint records including:

-   complaint type
-   complaint location
-   timestamps
-   agency responses

These complaints provide signals about **neighborhood conditions**.

------------------------------------------------------------------------

# Data Pipeline

The project contains a modular data pipeline for preparing datasets used
by models and visualizations.

### Data Ingestion

311 complaint data and business license datasets are collected and
processed using notebooks and pipeline scripts.

Files:

-   notebooks/NYC 311 Service Request Job.ipynb
-   notebooks/nyc_311_service_request_job.py
-   notebooks/nyc_issued_licenses_job.py

### Data Preprocessing

Preprocessing prepares a combined dataset linking complaints to
businesses.

Key tasks include:

-   cleaning datasets
-   joining business and complaint data
-   constructing modeling features
-   generating survival analysis inputs

Implemented in:

-   pipeline/preprocess.py
-   pipeline/utils.py

### Feature Engineering

Feature builders generate variables used in predictive models.

Implemented in:

-   app/utils/feature_builder.py
-   app/utils/cox_feature_builder.py

These features capture signals such as:

-   complaint counts
-   complaint categories
-   geographic attributes
-   business characteristics

------------------------------------------------------------------------

# Modeling

## Logistic Regression Model

Predicts the probability that a business remains open.

Pipeline implementation:

-   pipeline/logistic.py

Artifacts include:

-   logistic_pipeline.pkl
-   logistic_coefficient_summary.csv
-   logistic_evaluation_metrics.json

## Cox Proportional Hazards Model

Used to estimate survival probabilities over time.

Implemented in:

-   pipeline/cox.py

Artifacts include:

-   coxph_model.pkl
-   coxph_scaler.pkl
-   coxph_summary.csv

## Time‑Varying Cox Model

Captures how features change over time.

Artifacts stored in:

-   outputs/cox/time_varying/

These models enable deeper survival analysis beyond static prediction.

------------------------------------------------------------------------

# Model Artifacts

Model artifacts are stored in the repository and loaded by the
application.

Locations include:

-   outputs/
-   app/artifacts/

Artifacts include:

-   trained model pipelines
-   feature scalers
-   kept/dropped feature lists
-   evaluation metrics
-   model summaries

------------------------------------------------------------------------

# Interactive Streamlit Application

The project includes a multi‑page analytical application built with
**Streamlit**.

Application directory:

-   app/

Main pages include:

### Home Page

app/home.py

### Map Visualization

app/pages/01_map.py

Displays businesses on an interactive Mapbox map.

### Logistic Regression Analysis

app/pages/02_logistic_regression.py

Allows exploration of logistic regression model outputs.

### Cox Survival Models

app/pages/03_cox_models.py

Displays survival model results.

### Key Findings

app/pages/04_findings.py

Summarizes insights derived from the models.

------------------------------------------------------------------------

# Map Visualization

The map visualization uses **Mapbox** to display business locations
across NYC.

Map files:

-   app/map_templates/index.html
-   app/map_templates/app.js

Data source:

-   geojson/businesses.geojson

Each point on the map represents a business and provides information
including:

-   business attributes
-   geographic location
-   complaint signals

------------------------------------------------------------------------

# Running the Application

## Install dependencies

    conda env create -f environment.yml
    conda activate bizsurvival515

## Run the Streamlit app

    streamlit run app/home.py

------------------------------------------------------------------------

# Project Structure

    bizsurvival515
    │
    ├── app
    │   ├── home.py
    │   ├── pages
    │   ├── utils
    │   ├── artifacts
    │   └── map_templates
    │
    ├── pipeline
    │   ├── preprocess.py
    │   ├── logistic.py
    │   ├── cox.py
    │   ├── mapbox.py
    │   └── run_pipeline.py
    │
    ├── notebooks
    │   └── data preparation notebooks
    │
    ├── outputs
    │   ├── logistic
    │   ├── cox
    │   └── geojson
    │
    ├── tests
    │
    ├── docs
    │
    ├── environment.yml
    └── pyproject.toml

------------------------------------------------------------------------

# Testing

Run the test suite:

    pytest

Tests validate pipeline modules, utilities, and modeling components.

------------------------------------------------------------------------

# Continuous Integration

GitHub Actions runs automated CI including:

-   dependency installation
-   running unit tests
-   computing coverage
-   reporting to Coveralls

Workflow:

`.github/workflows/build_tests.yml`

------------------------------------------------------------------------

# Future Improvements

Potential extensions include:

-   additional NYC datasets
-   improved predictive models
-   real‑time complaint data integration
-   deployment as a public web dashboard
-   enhanced geospatial analytics
