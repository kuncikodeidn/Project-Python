from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import pandas as pd

class Evaluate:
    def __init__(self, y_ROP, optimized_ROP, y_Q, optimized_Q, ebo_values, SKU):
        self.y_ROP = y_ROP
        self.optimized_ROP = optimized_ROP 
        self.y_Q = y_Q 
        self.optimized_Q = optimized_Q
        self.ebo_values = ebo_values
        self.SKU = SKU
    
    def result_optimization(self, data):
        mse_ROP = mean_squared_error(data['ROPi'], data['Optimized_ROP'])
        mse_Q = mean_squared_error(data['Qi'], data['Optimized_Q'])
        print(f'MSE untuk ROP: {mse_ROP:.2f}')
        print(f'MSE untuk Q: {mse_Q:.2f}')

        mape_ROP = mean_absolute_percentage_error(data['ROPi'], data['Optimized_ROP']) * 100
        mape_Q = mean_absolute_percentage_error(data['Qi'], data['Optimized_Q']) * 100
        print(f'MAPE untuk ROP: {mape_ROP:.2f}%')
        print(f'MAPE untuk Q: {mape_Q:.2f}%')

        print()
        print("Expected Back Order (EBO) values for each SKU:")
        for i, ebo in enumerate(self.ebo_values):
            print(f'SKU {self.SKU.iloc[i]}: {ebo}')

    def visualize(self, data):
        errors_ROP = data['ROPi'] - data['Optimized_ROP']
        plt.figure(figsize=(12, 6))
        plt.hist(errors_ROP, bins=20, edgecolor='k', alpha=0.7)
        plt.title('Distribusi Kesalahan ROP')
        plt.xlabel('Kesalahan (true - pred)')
        plt.ylabel('Frekuensi')
        plt.show()

        errors_Q = data['Qi'] - data['Optimized_Q']
        plt.figure(figsize=(12, 6))
        plt.hist(errors_Q, bins=20, edgecolor='k', alpha=0.7)
        plt.title('Distribusi Kesalahan Q')
        plt.xlabel('Kesalahan (true - pred)')
        plt.ylabel('Frekuensi')
        plt.show()

    def save(self, data):
        data.to_excel("ebo.xlsx", engine='openpyxl')


        