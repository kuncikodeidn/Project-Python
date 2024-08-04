import matplotlib.pyplot as plt
import seaborn as sns

class Preproccess_Data:
    def __init__(self, df):
        self.df = df
        self.X = None
        self.y_ROP = None
        self.y_Q = None
        self.features = ['Di', 'd', 'DL', 'S', 'Li', 'Oci','Hci','Opi','Hbi','HJi']
        self.target_rop = 'ROPi'
        self.target_q = 'Qi'
    
    def cleaning(self):
        isnan_data = self.df.isna().sum() > 0
        for col in isnan_data:
            if col:
                print(f"there are nan in column {col}")

        isnull_data = self.df.isnull().sum() > 0
        for col in isnull_data:
            if col:
                print(f"there are nan in column {col}")

        if self.df[self.df.duplicated()].__len__() > 0:
            self.df.drop_duplicates()
    
    def show_histogram(self):
        self.df.hist(bins=50, figsize=(20, 15))
        plt.show()
    
    def show_corr(self):
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.df.corr(), annot=True, cmap='coolwarm')
        plt.show()

    def train_test_split(self):
        self.X = self.df[self.features]
        self.y_ROP = self.df[self.target_rop]
        self.y_Q = self.df[self.target_q]

        return self.X, self.y_ROP, self.y_Q

        

