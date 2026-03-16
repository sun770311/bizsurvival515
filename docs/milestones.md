# Milestones

## Team Responsibilities and Achievements
* Hannah Sun
   * Streamlit application and Mapbox visualization
   * Local pipeline scripting and model implementation
   * Refactoring modeling utilities and project structure
   * Unit testing
   * Documentation and videos
* Juan Pablo Reyes Martinez
   * CI integration
   * Pylint
   * Project management and coordination
* Sreeraj Parakkat
   * Data pipeline in Databricks to continuously refresh license and 311 complaint datasets
   * Delta lake tables and table preparation
   * (Future reach goal) Automate dataset joining to replace `pipeline/preprocess.py`
* Pavankumar Suresh
   * Documentation updates
   * Architecture diagrams
   * (Future reach goal) MLflow experiment tracking integration
* Aaron Lee
   * Logistic regression modeling exploration
   * Model interpretation and coefficient analysis

* Weekly stand-ups: conducted throughout project to coordinate progress and resolve integration issues

---

## Milestone 1 — Design Documents 
### Due: Feb 17

### Goal
Complete all required design documentation describing system purpose, architecture, and implementation plan.

### Tasks
1. (DONE) First draft of Functional Specification  
2. (DONE) First draft of Component Specification  
3. (DONE) First draft of Milestones plan  
4. (DONE) Review documents for consistency across components and use cases  
5. (DONE) Add design documents to repo under `docs/`

---

## Milestone 2 — Technology Review & First Demo
### Due: Feb 24

### Goal
Select an appropriate Python library to support a key technology requirement (e.g., interactive map visualization) and demonstrate its feasibility.

### Tasks
1. (DONE) Define the technology need and relevant use case: predictive modeling
2. (DONE) Identify 2–3 candidate Python libraries: Logistic Regression (`sklearn`), XGBoost (`xgboost`)
3. (DONE) Install and test each candidate library  
4. (DONE) Evaluate libraries based on:
   - ability to meet project requirements  
   - compatibility with Python 3 and project stack  
   - ease of use and documentation quality  
   - computational efficiency for dataset size  
   - stability / absence of blocking bugs  
5. (DONE) Produce side-by-side comparison of the libraries  
6. (DONE) Select final library and justify the choice: Logistic Regression (see `technology_review/`)
7. (DONE) Document drawbacks and risks of the chosen library  
8. (DONE) Prepare demo (screen recording or live demonstration)  
9. (DONE) Write the Technology Review Markdown document  
10. (DONE) Upload writeup and demo to `docs/technology_review/` in GitHub  

---

## Milestone 3 — Pipeline Implementation
### Feb 24 – March 13

### Goal
Develop the core data and modeling pipelines required for  the analysis.

### Completed Tasks
#### Data Pipeline
* Implemented Databricks jobs to ingest NYC Open Data
* Automated updates to licenses and service requests datasets
* Enable periodic downloading of updated CSVs

#### Feature Engineering
* Joined complaint data with business license records
* Engineered temporal and categorical features representing neighborhood signals
* Created a structured dataset suitable for survival modeling

#### Modeling Pipelines
* Implemented modular pipelines including:
   * Logistic Regression pipeline
   * Cox Proportional Hazards survival pipeline

* Each pipeline includes:
   * Dataset preparation
   * Feature selection
   * Model training
   * Evaluation and interpretation

#### Pipeline Refactoring
* Refactored pipeline modules for improved maintainability
* Organized code into structured pipeline components

---

## Milestone 4 — Testing and Code Quality
### Feb 24 - March 13

### Goal
Ensure correctness, reliability, and maintainability of the pipeline codebase.

### Completed Tasks
* Implemented extensive unit tests for pipeline modules, including:
   * Data validation utilities
   * Modeling functions
   * Feature engineering logic
   * GeoJSON generation utilities
* Achieved broad test coverage across core pipeline functionality
* Integrated automated testing using GitHub Actions CI
* Enforced code quality standards using pylint
* Refactored code to resolve linting issues and improve readability
* Validated end-to-end pipeline execution

---

## Milestone 5 — Application Implementation in Streamlit
### Feb 24 - March 13

### Goal
Build an interactive web application that allows users to explore results.

### Completed Tasks
* Implemented a Streamlit application with a multi-page structure
* Built an interactive Mapbox visualization showing NYC businesses
* Developed a pipeline to generate GeoJSON outputs for map rendering
* Enabled hypothetical business creation and output analysis
* Created pages explaining:
   * Datasets used
   * Methodology
   * Model findings
* Extensively tested application utilities

---

## Milestone 6 — Final Demo Preparation
### Presentation day: March 16

### Completed Tasks
* Prepare a video walkthrough demonstrating:
   * Data pipeline
   * Modeling approach
   * Interactive application
* Finalize documentation across the repository
* Improve README and project explanations
* Perform final testing and integration checks
* Final presentation slides and script
