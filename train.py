from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
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
    
    parser.add_argument("--min_impurity_decrease", type=float, default=0.0, help="A node will be split if this split induces a decrease of the impurity greater than or equal to this value.")
    parser.add_argument("--min_samples_leaf", type=float, default=10, help="The minimum number of samples required to be at a leaf node.")
    parser.add_argument("--min_weight_fraction_leaf", type=float, default=0.0, help="The minimum weighted fraction of the sum total of weights of all the input samples required to be at a leaf node.")

    #primary_metric_name='Accuracy'
    args = parser.parse_args()


    run.log("min_impurity_decrease:", np.int(args.min_impurity_decrease))
    run.log("min_samples_leaf:", np.float(args.min_samples_leaf))
    run.log("min_weight_fraction_leaf:", np.int(args.min_weight_fraction_leaf))



    model = RandomForestRegressor(min_impurity_decrease=args.min_impurity_decrease, min_samples_leaf=args.min_samples_leaf, min_weight_fraction_leaf=args.min_weight_fraction_leaf).fit(x_train, y_train)
    
    accuracy = model.score(x_test, y_test)
    run.log('Accuracy', np.float(accuracy))
    os.makedirs('outputs', exist_ok=True)

    joblib.dump(value=model, filename='outputs/model.pkl')
    
if __name__ == '__main__':
    main()
