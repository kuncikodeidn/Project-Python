from preprocessing import Preproccess_Data
from optimization import Optimizer
from evaluation import Evaluate
import pandas as pd
import os
import numpy as np

data = pd.read_csv(os.getcwd() + "/Dataset1.csv")

SKU = data["SKU_Code"]
total_demand = data['Di'].sum()
data['Weight_Ratio'] = data['Di'] / total_demand

assert np.isclose(data['Weight_Ratio'].sum(), 1), "The sum of weight ratios does not equal 1"

data.drop(columns="No", inplace=True)

preproccess = Preproccess_Data(data)
optimizer = Optimizer()

preproccess.cleaning()
X, y_ROP, y_Q = preproccess.train_test_split()

data = optimizer.optimize(data, X)

evaluate = Evaluate(y_ROP, optimizer.optimized_ROP, y_Q, optimizer.optimized_Q, optimizer.ebo_values, SKU)
evaluate.result_optimization(data)
evaluate.visualize(data)

print(data.drop(columns=["Di","d","DL","S","ROPi","Qi","Li","Oci","Hci","Opi","Hbi","HJi"]))

# pip install openpyxl 
evaluate.save(data.drop(columns=["Di","d","DL","S","ROPi","Qi","Li","Oci","Hci","Opi","Hbi","HJi"]))

