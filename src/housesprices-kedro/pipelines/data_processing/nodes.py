import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.impute import SimpleImputer
from prefect import flow

@flow

def remove_outliers(train, parameters):

    train = train.copy()
    train = train.drop(train[(train['GrLivArea'] > parameters['outliers']['GrLivArea']) &
                             (train['SalePrice'] < parameters['outliers']['SalePrice'])]
                       .index)
    return train

def create_target(train):
    house_prices_target = np.log1p(train["SalePrice"])
    return house_prices_target

def drop_cols(train, parameters):
    return train.drop(parameters['cols_to_drop'],axis=1)

def fill_na(train, parameters):
    train = train.copy()
    train[parameters['none_cols']] = train[parameters['none_cols']].fillna("None")
    train[parameters['zero_cols']] = train[parameters['zero_cols']].fillna(0)

    impute_int = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    impute_str = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

    int_cols = train.select_dtypes(include='number').columns
    str_cols = train.select_dtypes(exclude='number').columns

    train[int_cols] = impute_int.fit_transform(train[int_cols])
    train[str_cols] = impute_str.fit_transform(train[str_cols])

    impute_float = SimpleImputer(missing_values=np.nan, strategy='mean')
    train['LotFrontage'] = impute_float.fit_transform(train[['LotFrontage']])

    return train

def total_sf(train):
    train = train.copy()
    train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']
    return train
def preprocess_data_for_model(X_train: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    X_train_encoded = pd.DataFrame(encoder.fit_transform(X_train[categorical_cols]))

    X_train_encoded.columns = encoder.get_feature_names_out(categorical_cols)
    X_train = pd.concat([X_train.drop(categorical_cols, axis=1), X_train_encoded], axis=1)

    return X_train
