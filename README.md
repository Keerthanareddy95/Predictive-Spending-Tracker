# Predictive-Spending-Tracker
Predicting how much a person spends on various activities, by integrating with tools like ZenML, MLflow for deployment, tracking and more.

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/zenml)](https://pypi.org/project/zenml/)

The purpose of this repository is to demonstrate how [ZenML](https://github.com/zenml-io/zenml) empowers us to build and deploy machine learning pipelines in multiple ways like:

- By integrating with tools like [MLflow](https://mlflow.org/) for deployment, tracking and more
- By allowing you to build and deploy your machine learning pipelines easily
  

## :snake: Python Requirements

Let's jump into the Python packages needed. Within the Python environment of your choice, run:

```bash
pip install -r requirements.txt
```
Starting with ZenML 0.20.0, ZenML comes bundled with a React-based dashboard. This dashboard allows us to observe the stacks, stack components and pipeline DAGs in a dashboard interface. To access this, you need to [launch the ZenML Server and Dashboard locally](https://docs.zenml.io/user-guide/starter-guide#explore-the-dashboard), and you must install the optional dependencies for the ZenML server:

```bash
pip install "zenml["server"]"
zenml up
```
Installing mlflow integrations using ZenML:

```bash
zenml integration install mlflow -y
```

## Pipeline Development Process :

### 1. Creation of a Blueprint of the classes -
   
   Steps > ingest_data.py , clean_data.py , model_train.py , evaluation.py
   
### 2. Data Cleaning -
   
   data_cleaning.py > -DataPreprocess,  -DataDivision
### 3. Model Development -
   
   Building the model on Train & Test datasets.
   
### 4. Defining Evaluation metrics -
   
   src > evaluation.py - defining MSE , RMSE < R2 Score
   
### 5. Training pipeline -
   - `ingest_data`: This step will ingest the data and create a `DataFrame`.
   - `clean_data`: This step will clean the data and remove the unwanted columns.
   - `train_model`: This step will train the model and save the model using [MLflow autologging](https://www.mlflow.org/docs/latest/tracking.html).
   - `evaluation`: This step will evaluate the model and save the metrics -- using MLflow autologging -- into the artifact store.

<img width="272" alt="Screenshot 2024-02-17 224626" src="https://github.com/Keerthanareddy95/Predictive-Spending-Tracker/assets/123613605/99bb02cd-551b-4e39-9aae-1793595f7e1d">




The project can only be executed with a ZenML stack that has an MLflow experiment tracker and model deployer as a component. Configuring a new stack with the two components are as follows:

```bash
zenml integration install mlflow -y
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow --flavor=mlflow
zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set
```

### 6. Deployment Pipeline

We have another pipeline, the `deployment_pipeline.py`, that extends the training pipeline, and implements a continuous deployment workflow. It ingests and processes input data, trains a model and then (re)deploys the prediction server that serves the model if it meets our evaluation criteria. The criteria that we have chosen is a configurable threshold on the [MSE](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html) of the training. The first four steps of the pipeline are the same as above, but we have added the following additional ones:

- `deployment_trigger`: The step checks whether the newly trained model meets the criteria set for deployment.
- `model_deployer`: This step deploys the model as a service using MLflow (if deployment criteria is met).

In the deployment pipeline, ZenML's MLflow tracking integration is used for logging the hyperparameter values and the trained model itself and the model evaluation metrics -- as MLflow experiment tracking artifacts -- into the local MLflow backend. This pipeline also launches a local MLflow deployment server to serve the latest MLflow model if its accuracy is above a configured threshold.

The MLflow deployment server runs locally as a daemon process that will continue to run in the background after the example execution is complete. When a new pipeline is run which produces a model that passes the accuracy threshold validation, the pipeline automatically updates the currently running MLflow deployment server to serve the new model instead of the old one.

<img width="270" alt="Screenshot 2024-02-17 224448" src="https://github.com/Keerthanareddy95/Predictive-Spending-Tracker/assets/123613605/5e6e51b8-63b2-4106-bdd7-1ac272f867b2">


### 7. Streamlit Application -
To round it off, we deploy a Streamlit application that consumes the latest model service asynchronously from the pipeline logic. This can be done easily with ZenML within the Streamlit code:

```python
service = prediction_service_loader(
   pipeline_name="continuous_deployment_pipeline",
   pipeline_step_name="mlflow_model_deployer_step",
   running=False,
)
...
service.predict(...)  # Predict on incoming data from the application
```
