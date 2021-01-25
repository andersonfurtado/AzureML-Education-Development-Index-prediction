from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

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
    
    parser.add_argument('--max_depth',
                        type=int,
                        default=5,
                        help="The maximum depth of the tree.")
    parser.add_argument('--min_samples_split',
                        type=int,
                        default=2,
                        help="The minimum number of samples required to split an internal node.")
    parser.add_argument('--min_samples_leaf',
                        type=int,
                        default=1,
                        help="The minimum number of samples required to be at a leaf node.")


    #primary_metric_name='r2_score'
    args = parser.parse_args()


    run.log("max_depth:", np.int(args.max_depth))   
    run.log("min_samples_split:", np.int(args.min_samples_split))
    run.log("min_samples_leaf:", np.int(args.min_samples_leaf))


# Train Random Forest Model
    model = RandomForestRegressor(
                                   max_depth=args.max_depth,
                                   min_samples_split=args.min_samples_split,
                                   min_samples_leaf=args.min_samples_leaf).fit(x_train, y_train)

# calculate accuracy                                   
    accuracy = model.score(x_test, y_test)
    run.log('Accuracy', np.float(accuracy))

# calculate r2 score
    y_pred = model.predict(x_test)
    # Notice that my variable is named r2 to avoid confusion with the r2_score we imported
    r2 = r2_score(y_test, y_pred) 
    run.log('r2_score', np.float(r2))

# Save the trained model   
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(value=model, filename='outputs/model.pkl')
    
if __name__ == '__main__':
    main()
