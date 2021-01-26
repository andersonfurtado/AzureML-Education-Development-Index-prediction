# Prediction of the Basic Education Development Index (Ideb) using Azure Machine Learning

# Table of Contents
<!--ts-->
- [Dataset](#dataset)
  * [Overview](#overview)
  * [Task](#task)
  * [Access](#access)
- [Automated ML](#automated-ml)
  * [Overview of AutoML Settings](#overview-of-automl-settings)
  * [Results](#results)
  * [RunDetails Widget](#rundetails-widget)
  * [Best Model](#best-model)
- [Hyperparameter Tuning](#hyperparameter-tuning)
  * [Overview of Hyperparameter Tuning Settings](#overview-of-hyperparameter-tuning-settings)
  * [Results](#results)
  * [RunDetails Widget](#rundetails-widget)
  * [Best Model](#best-model)
- [Model Deployment](#model-deployment)
  * [Overview of Deployed Model](#overview-of-deployed-model)
  * [Endpoint](#endpoint)
  * [Endpoint Query](#endpoint-query)  
- [Screen Recording](#screen-recording)
- [Suggestions to Improve](#suggestions-to-improve)
 
<!--te-->  

The project is about the prediction of the Basic Education Development Index (Ideb) of high schools in Brazil. The Ideb is calculated from data on school approval, obtained from the School Census, and performance averages in the Basic Education Assessment System (Saeb) in Brazil. Ideb adds to the pedagogical focus of large-scale evaluations the possibility of synthetic results, which are easily assimilable, and which allow the establishment of educational quality goals for education systems.

In this project, I considered a regression problem, that is, a process where a model learns to predict a continuous value output for a given input data. For this project, I ran essentially two processes:
<ol>
  <li>First I applied AutoML where several models are trained to fit the training data. Then I chose and saved the best model, that is, the model with the best score.
  </li><br>
  <li> Second, using HyperDrive, I adjusted the hyperparameters and applied the Random Forest Regressor which consists of a meta estimator that fits a number of classifying decision trees on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
  </li><br>
</ol>
Hyperdrive and Automl models have been trained and deployed using an endpoint in Azure Machine Learning.

The resolution of this problem is justified because educational indicators such as Ideb are desirable because they allow the monitoring of the country's education system. Its importance, in terms of diagnosis and guidance of political actions focused on improving the educational system, is in:
- detect schools and / or education networks whose students are underperforming in terms of performance and proficiency;
- monitor the temporal evolution of student performance in these schools and / or education networks.

## Project Set Up and Installation
The starter files that you need to run this project are the following:
- **automl.ipynb**: Jupyter Notebook to run the autoML experiment
- **hyperparameter_tuning.ipynb**: Jupyter Notebook to run the Hyperdrive experiment
- **train.py*: Script used in Hyperdrive
- **score.py**: Script used to deploy the model
- **ideb_dataset.csv**: The dataset prepared after the release of the Basic Education Development Index (Ideb) by the National Institute for Educational Studies and Research Anísio Teixeira (Inep).


## Dataset

### Overview
The data were obtained from the 2019 School Census and released on September 15, 2020 by Inep, which can be found at <http://download.inep.gov.br/educacao_basica/portal_ideb/planilhas_para_download/2019/divulgacao_ensino_medio_municipios_2019.zip>.

### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.
Considering that Ideb was developed to be an indicator that synthesizes information from performance in standardized exams with information on school performance (average rate
approval of students in the teaching stage), the features used to predict Basic Education Development Index were:
- **Total Approval Rate (2019)**: In the database called "TAprov2019 Total"
- **1st Series Approval Rate (2019)**: In the database called "TAprov2019_1_serie"
- **2nd Series Approval Rate (2019)**: In the database called "TAprov2019_2_serie"
- **2nd Series Approval Rate (2019)**: In the database called "TAprov2019_3_serie"
- **4th Grade Approval Rate (2019)**: In the database called "TAprov2019_4_serie"
- **Approval_Indicator**: In the database called "Indicador de Rendimento"
- **Grade SAEB Mathematics (2019)**: In the database called "SAEB2019_Matematica"
- **Grade SAEB Language Portuguese (2019)**: In the database called "SAEB2019_Lingua Portuguesa"
- **SAEB Standardized Average Score (N)**: In the database called "SAEB2019_Nota Media Padronizada"

### Access
Initially, I explored the Ideb database repository on the Inep website to obtain a set of data to train the models. As soon as I decided to work with the Ideb 2019 data set by high school, I found the link to download the data. The link was then passed to the from_delimited_files method of the Tabular class object of the Dataset class object.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

To configure the Automated ML run we need to specify what kind of a task we are dealing with, the primary metric, train and validation data sets (which are in TabularDataset form) and the target column name. Featurization is set to "auto", meaning that the featurization step should be done automatically. 

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

The best model overall was the `VotingEnsemble` model, with an R2 Score of 0.99787. The remainder of the models were between about 0.75571 and 0.99786, except VotingEnsemble.


The AutoML run has the following parameters:

* *task*: To help manage child runs and when they can be performed, we recommend you create a dedicated cluster per experiment. In this project, `4` was the number of concurrent child runs/iterations. 

* *max_concurrent_iterations*: This is the metrics which is optimised during the training of the model algorithm. For example, Accuracy, Area Under Curve(AUC), R2 Score, etc. In this project, `R2 Score` was used as an success metrics.
* *primary_metric*: This is the metrics which is optimised during the training of the model algorithm. For example, Accuracy, Area Under Curve(AUC), R2 Score, etc. In this project, `R2 Score` was used as an success metrics.

* *experiment_timeout_minutes*: This is the max time for which the AutoML can use different models to train on the dataset. In this project, the max timeout in hours was `0.5`.

* *training_data*: This is the training data on which all the different models are trained on. In this project, it was `ideb_dataset`.

* *label_column_name*: This is the variable which is predicted by the model. In this project, it was the `IDEB_2019`.

* *n_cross_validations*: This is the n in cross validation process. Cross validation is the process where different data points are put into training and testing dataset (resampling) to ensure that the model does not overfit over certain values. In this project, the n was `3`.

* *featurization*: This help certain algorithms that are sensitive to features on different scales. For example, you can enable more featurization, such as missing-values imputation, encoding, and transforms. In this project, the featurization was `auto` (specifies that, as part of preprocessing, data guardrails and featurization steps are to be done automatically). 


*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

This contains the `RunDetails` widget from the Jupyter notebook implementation in the case of AutoML run:

![automl_rundetails](./figs/automl_run_details.png)

The following screenshots shows the best run ID:

![automl_rundetails](./figs/automl_run_details.png)

For a complete code overview, we refer to the jypter notebook automl.ipynb.

## Hyperparameter Tuning
The objective of this project is to predict the Ideb per school at the Brazilian middle level. According to the data, the target variable "IDEB_2019" is a floating variable, which is a continuous variable ranging from 1.0, 1.6, 2.5, 3.5, 7.5 etc., therefore, it deals with yourself from a regression problem. In this case, then use regression models such as linear regression, random forest regression or any other regression model. Considering that in the classification model, I need to convert the target resource - "IDEB_2019" into a categorical resource with 1 or 0, not being suitable for the purpose of this project, it is evident that this is a `regression problem`.

Since the task was a regression problem, the model used the `Random Forest Regressor`, since: (i) It is one of the most accurate learning algorithms available. For many data sets, it produces a highly accurate classifier; (ii) It runs efficiently on large databases; (iii) It can handle thousands of input variables without variable deletion; (iv) It generates an internal unbiased estimate of the generalization error as the forest building progresses; (v) It has an effective method for estimating missing data and maintains accuracy when a large proportion of the data are missing.

Para o experimento de ajuste de hiperparâmetros, feito via HyperDrive, os tipos de parâmetros e seus intervalos utilizados tiveram os seguintes parâmetros:

* *primary_metric_name*: This is the metrics which is optimised during the training of the model algorithm. For example, Accuracy, Area Under Curve(AUC) etc. In this project, `R2 Score` is used as an success metrics.

* *primary_metric_goal*: This is the parameter which tells Hyperdrive how to optimise the algorithm using the primary_metric_name given. The goal can be anything from Maximize to Minimise the primary_metric_name. In this project, it is `PrimaryMetricGoal.MAXIMIZE`.

* *max_total_runs*: This is the maximum number of runs which Hyperdrive will run using different hyperparameters. In this project, the max_total_runs is `20`.

* *max_concurrent_runs*: This is the maximum number of run which run concurrently over different threads. In this project, the max_concurrent_runs is `4`.

* *hyperparameter_sampling*: This is the Parameter Sampler which specfies the techniques in which the hyperparameters are tuned. In this project, RandomParameterSampling was used to tune the hyperparameter '--max_depth' with `choice(range(1, 20)`, '--min_samples_split' with `choice(2, 5, 10, 15, 100)` and '--min_samples_leaf' with `choice(range(1, 10)`.

* *policy*: This is the early stopping policy used by Hyperdrive which is used to provide guidance as to how many iterations can be run before the model begins to overfit. In this project, BanditPolicy was used with argument evaluation_interval of `2` and slack_factor of `0.1`. BanditPolciy terminates any run whos primary metrics is less than the slack factor of best run.

* *RunConfig*: There are several methods for setting up a training job in Azure Machine Learning through the SDK. This is the training script that will be run with the sample hyperparameters. For exemple, Estimators, ScriptRunConfig, and the lower-level RunConfiguration. In this project `ScriptRunConfig` (It defines the resources per job (single or multi-node), and the compute target to use).

This contains the RunDetails implementation in the case of Hyperdrive run.

### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

The best model of HyperDrive run was VotingEnsemble with `0.99787 R2 Score`. This contains the `RunDetails` implementation of the the best model trained with it's parameters:
![automl_rundetails](./figs/automl_run_details.png)

This shows that the best model generated by Hyperdrive with its run id:
![automl_rundetails](./figs/automl_run_details.png)

Two ideas on how to perhaps improve the The best model of HyperDrive:




## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.
Summarizing the results, the best model generated by AutoML had the R2 Score of; and the best model when running HyperDrive was with R2 Score of. So, I implemented the best model - which happened to be the VotingEnsemble model for executing AutoML, as an endpoint.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.

The model can be imporved by increasing the number of iterations and setting the featurization to be auto.Using neural network based classification to improve the performance of the model.
