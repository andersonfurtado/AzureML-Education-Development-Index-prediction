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
  <li>Setup Training Script (Review and Modify the `train.py` script)
    <ul>
      <li> Import data from the specified URL using "TabularDatasetFactory" </li>
      <li> Split data into train and test sets (70% for training and 30% for testing) </li>
      <li> Splitting of data into train and test data </li>
      <li> Using scikit-learn logistic regression model for classification </li>
    </ul>
  </li><br>
  <li> Creation of a computing cluster, to HyperDrive, to running the train.py script containing the custom-coded Logistic Regression with a HyperDrive run
  </li><br>
  <li> Review and modify the `project.ipynb` notebook
    <ul>
      <li> Specify the parameter sample </li>
      <li> Specify the policy for early stopping </li>
      <li> Creation of estimator for the train.py script </li>
      <li> Creation of HyperDriveConfig </li>
      <li> Submition of HyperDriveConfig to run the experiment </li>
      <li> Using the RunDetails widget to see the progress of the run </li>
      <li> Using the .get_best_run_by_primary_metric() method of the run to select the best hyperparameters for the model </li>
      <li> Save the best model </li>
    </ul>
  </li><br>
</ol>
First I applied AutoML where several models are trained to fit the training data. Then I chose and saved the best model, that is, the model with the best score. Second, using HyperDrive, I adjusted the hyperparameters and applied the Random Forest Regressor which consists of a meta estimator that fits a number of classifying decision trees on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.

## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

## Dataset

### Overview
*TODO*: Explain about the data you are using and where you got it from.

### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.

### Access
*TODO*: Explain how you are accessing the data in your workspace.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
