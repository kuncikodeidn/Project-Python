from scipy.optimize import minimize
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from scipy.integrate import quad
import pandas as pd


class Optimizer:
    def __init__(self):
        self.optimized_params = None
        self.ebo_values = None
        self.optimized_Q = None
        self.optimized_ROP = None
        self.X_scaled = None
        self.optimized_params_df = None
        self.W1, self.W2 = 0.7, 0.3 


    def cost_function(self, params, Di, DL, S, Oci, Hci, Opi, Hbi):
        Q, ROP = params
        EBO_value = EBO(Q, ROP, DL, S)
        total_cost = Oci * (Di / Q) + Hci * (Q / 2 + ROP - DL) + Opi * (EBO_value * Di / Q) + Hbi * Di
        return total_cost
    
    def optimize(self, data, X):
        print("start optimization")
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(X)

        self.optimized_params = []
        for i in range(len(data)):
            initial_guess = [50, 50]  
            row = data.loc[i]
            result = minimize(self.performance_score, initial_guess, args=(row['Di'], row['DL'], row['S'], row['Oci'], row['Hci'], row['Opi'], row['Hbi'], row['HJi'], self.W1, self.W2),
                            method='L-BFGS-B', bounds=[(1, 200), (1, 100)])
            self.optimized_params.append(result.x)
            print(f"Row {i}: Initial guess: {initial_guess}, Optimized: {result.x}, Cost: {result.fun}")

        self.optimized_params_df = pd.DataFrame(self.optimized_params, columns=['Optimized_Q', 'Optimized_ROP'])
        data['Optimized_Q'] = self.optimized_params_df['Optimized_Q']
        data['Optimized_ROP'] = self.optimized_params_df['Optimized_ROP']

        self.optimized_Q = data['Optimized_Q']
        self.optimized_ROP = data['Optimized_ROP']

        data['EBO'] = data.apply(lambda row: EBO(row['Optimized_Q'], row['Optimized_ROP'], row['DL'], row['S']), axis=1)
        data['EBO'] = round(data['EBO'], 5)

        self.ebo_values = data['EBO']
        data['FRi'] = 1 - data['EBO'] / data['Optimized_Q']

        total_fill_rate = np.sum((1 - data['EBO'] / data['Optimized_Q']) * data['Di']) / np.sum(data['Di'])
        print(f'Total Fill Rate: {total_fill_rate:.2f}')

        if total_fill_rate < 0.95:
            print('Total Fill Rate constraint not met')

        if np.sum(data['Optimized_Q'] * data['Di']) > 0.8 * np.sum(data['HJi'] * data['Di']):
            print('Total Cost constraint not met')

        if abs(self.W1 + self.W2 - 1) > 1e-6:
            raise ValueError('Weight ratios constraint not met')
        
        data['TCi'] = data.apply(lambda row: self.cost_function([row['Optimized_Q'], row['Optimized_ROP']], row['Di'], row['DL'], row['S'], row['Oci'], row['Hci'], row['Opi'], row['Hbi']), axis=1).round(5)
        data['PRi'] = round(data['HJi'] * data['Di'] - (data['Oci'] * (data['Di'] / data['Optimized_Q']) + data['Hci'] * (data['Optimized_Q'] / 2 + data['Optimized_ROP'] - data['DL']) + data['Opi'] * (data['EBO'] * data['Di'] / data['Optimized_Q']) + data['Hbi'] * data['Di']), 5)
        data['PSi'] = data.apply(lambda row: self.W1 * row['PRi'] + self.W2 * row['FRi'], axis=1).round(5)

        return data

    def performance_score(self, params, Di, DL, S, Oci, Hci, Opi, Hbi, HJi, W1, W2):
        Q, ROP = params
        EBO_value = EBO(Q, ROP, DL, S)
        total_cost = self.cost_function(params, Di, DL, S, Oci, Hci, Opi, Hbi)
        revenue = Di * HJi
        profit = revenue - total_cost
        fill_rate = 1 - (EBO_value / Q)
        PS = W1 * profit + W2 * fill_rate
        return -PS

def EBO(Q, ROP, DL, S):
    mean_demand = DL
    std_demand = np.sqrt(DL) * S
    
    def integrand(x):
        return (x - ROP) * norm.pdf(x, loc=mean_demand, scale=std_demand)
    
    ebo_value, _ = quad(integrand, ROP, ROP + Q)
    return ebo_value

