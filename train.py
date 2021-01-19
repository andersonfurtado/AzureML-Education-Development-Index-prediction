from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import argparse
import os
import numpy as np
import joblib
import pandas as pd

from azureml.core import Workspace
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core import Dataset
from azureml.core.run import Run




run = Run.get_context()

ideb_dataset = Dataset.Tabular.from_delimited_files("https://raw.githubusercontent.com/andersonfurtado/Capstone/master/data/data.CSV", separator=';', encoding= 'latin1')
data = ideb_dataset.to_pandas_dataframe()


def clean_data(data):
   
    x_df = data.to_pandas_dataframe()
    
    y_df = x_df.pop("IDEB_2019")

    return x_df,y_df


x, y = clean_data(ideb_dataset)

# TODO: Split data into train and test sets.
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=403,shuffle=True) # 1 

run = Run.get_context()

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--fit_intercept", type=bool, default=True, help="Calculate the intercept for this model")
    parser.add_argument("--n_jobs", type=int, default=None, help="The number of jobs to use for the computation.")

    #primary_metric_name='Accuracy'
    args = parser.parse_args()

    run.log("fit_intercept:", np.float(args.fit_intercept))
    run.log("n_jobs:", np.int(args.n_jobs))



    model = LinearRegression(fit_intercept=args.fit_intercept, n_jobs=args.n_jobs).fit(x_train, y_train)
    
    accuracy = model.score(x_test, y_test)
    run.log('Accuracy', np.float(accuracy))
    os.makedirs('outputs', exist_ok=True)

    joblib.dump(value=model, filename='outputs/model.pkl')
    
if __name__ == '__main__':
    main()
