import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    df['Geography'] = le.fit_transform(df['Geography'])
    df['Exited'] = df['Exited'].astype(int)
    
    df.fillna(method='ffill', inplace=True)

    features = ['CreditScore', 'Geography', 'Gender', 'Age', 'Balance', 'NumOfProducts', 'IsActiveMember', 'EstimatedSalary']
    X = df[features]
    y = df['Exited']
    return X, y
