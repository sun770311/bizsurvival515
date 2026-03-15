# Open or Closed? NYC Business Survival

[![Build and Tests](https://github.com/sun770311/bizsurvival515/actions/workflows/build_tests.yml/badge.svg)](https://github.com/sun770311/bizsurvival515/actions/workflows/build_tests.yml)
[![Coverage Status](https://coveralls.io/repos/github/sun770311/bizsurvival515/badge.svg)](https://coveralls.io/github/sun770311/bizsurvival515)


## 🚀 Explore the Application! https://bizsurvival515.streamlit.app

# Overview

Small businesses in New York City face significant uncertainty, with many opening and closing each year due to a range of economic, geographic, and environmental factors. Understanding what influences **business survival** can help entrepreneurs, investors, and policymakers make better decisions about where and how businesses operate.

This project investigates whether **neighborhood conditions reflected in NYC 311 service complaints relate to business survival**. These complaints capture real-world signals about urban environments, including sanitation issues, infrastructure problems, noise disturbances, and other local conditions that may affect businesses.

We built a data pipeline and deployed a Streamlit application to analyze business outcomes across New York City. The system includes predictive modeling, survival analysis, and geospatial visualization tools that enable interactive exploration of the data. Together, these components support the investigation of patterns associated with business survival using public urban datasets.

## Project Type
**Tool Project**

# Team Members

Hannah Sun, Juan Pablo Reyes Martinez, Sreeraj Parakkat, Pavankumar Suresh, Aaron Lee

# Questions of Interest

* Can patterns in **311 complaints and neighborhood activity** help estimate the **probability that a business remains open for a certain period of time**?
* Can we estimate the **changing risk of closure over time** based on neighborhood and business characteristics?
* What **environmental and geographic factors** correlate with business survival versus closure?

# Our Goal

Our goal is to build an **interactive analytical tool** that enables users to explore relationships between neighborhood conditions and business outcomes.

The system allows users to:

* visualize businesses across NYC on an interactive map
* explore complaint patterns surrounding businesses
* examine model evaluation results and interpret coefficients
* create hypothetical business scenarios and examine resulting survival curves

# Data Sources

The project uses publicly available datasets from **NYC Open Data**, which are regularly updated.

### NYC Issued Business Licenses [[Link]](https://data.cityofnewyork.us/Business/Issued-Licenses/w7w3-xahh/about_data)

This dataset contains information about licensed businesses in New York City, including:

- business identifiers  
- industry classifications  
- geographic location  
- issuance and expiration dates  

### NYC 311 Service Requests [[Link]](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2020-to-Present/erm2-nwe9/about_data)

This dataset records public service complaints submitted by residents, including:

- complaint types  
- locations  
- timestamps  

These complaints provide signals about **neighborhood conditions surrounding businesses**.

# Software Dependencies

Software dependencies are defined in `pyproject.toml` and the Conda environment used to install them is specified in `environment.yml`.

# How to Run the Project

## Clone the Repository

```bash
git clone https://github.com/sun770311/bizsurvival515.git
cd bizsurvival515
```

## Create the project environment and install all dependencies:

```bash
conda env create -f environment.yml
conda activate bizsurvival
```

## Run the Data and Modeling Pipeline

```bash
python -m bizsurvival515.pipeline.run_pipeline \
  --data-dir bizsurvival515/tests/data \
  --output-dir bizsurvival515/outputs \
  --licenses-file licenses_sample.csv \
  --service-reqs-file service_reqs_sample.csv \
  --joined-file joined_dataset.csv \
  --write-joined-to-data-dir
```

See `docs/pipeline.md` for full documentation on the pipeline.

## Run the Interactive Streamlit Application

The Streamlit application visualizes model outputs generated from the **full joined dataset**, which is over **2 GB in size**. These artifacts are produced by running the modeling pipeline on the complete dataset rather than the small sample datasets provided in `bizsurvival515/tests/data/`.

Because the full dataset is large and the modeling steps can be computationally intensive, we recommend running the pipeline for the full dataset in a **high-resource environment**, such as:

- Google Colab (GPU runtime)
- a cloud compute environment

Launch the interactive application locally:

```bash
streamlit run bizsurvival515/app/home.py
```

## Run Unit Tests

Run all tests using:

```bash
python -m unittest discover -s bizsurvival515/tests -v
```

# Future Work

* Add scripting to automatically generate and export pipeline outputs as model artifacts used by the Streamlit application.

---

#### A University of Washington MSDS DATA 515 project.
