




import numpy as np

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib



def describe_numeric_col(x):
    """
    Parameters:
        x (pd.Series): Pandas col to describe.
    Output:
        y (pd.Series): Pandas series with descriptive stats. 
    """
    return pd.Series(
        [x.count(), x.isnull().count(), x.mean(), x.min(), x.max()],
        index=["Count", "Missing", "Mean", "Min", "Max"]
    )

def impute_missing_values(x, method="mean"):
    """
    Parameters:
        x (pd.Series): Pandas col to describe.
        method (str): Values: "mean", "median"
    """
    if (x.dtype == "float64") | (x.dtype == "int64"):
        x = x.fillna(x.mean()) if method=="mean" else x.fillna(x.median())
    else:
        x = x.fillna(x.mode()[0])
    return x



def feature_selection():
    data = pd.read_csv("./artifacts/raw_data_framed.csv")

    data = data.drop(
    [
        "is_active", "marketing_consent", "first_booking", "existing_customer", "last_seen"
    ],
    axis=1
    )

    data = data.drop(
    ["domain", "country", "visited_learn_more_before_booking", "visited_faq"],
    axis=1
    )


    data["lead_indicator"].replace("", np.nan, inplace=True)
    data["lead_id"].replace("", np.nan, inplace=True)
    data["customer_code"].replace("", np.nan, inplace=True)

    data = data.dropna(axis=0, subset=["lead_indicator"])
    data = data.dropna(axis=0, subset=["lead_id"])

    data = data[data.source == "signup"]
    result=data.lead_indicator.value_counts(normalize = True)

    print("Target value counter")
    for val, n in zip(result.index, result):
        print(val, ": ", n)

    


    vars = [
        "lead_id", "lead_indicator", "customer_group", "onboarding", "source", "customer_code"
    ]

    for col in vars:
        data[col] = data[col].astype("object")
        print(f"Changed {col} to object type")


    cont_vars = data.loc[:, ((data.dtypes=="float64")|(data.dtypes=="int64"))]
    cat_vars = data.loc[:, (data.dtypes=="object")]

    #print("\nContinuous columns: \n")
    #print(list(cont_vars.columns), indent=4)
    #print("\n Categorical columns: \n")
    #pprint(list(cat_vars.columns), indent=4)


    cont_vars = cont_vars.apply(lambda x: x.clip(lower = (x.mean()-2*x.std()),
                                                upper = (x.mean()+2*x.std())))
    outlier_summary = cont_vars.apply(describe_numeric_col).T
    outlier_summary.to_csv('./artifacts/outlier_summary.csv')
    outlier_summary


    cat_missing_impute = cat_vars.mode(numeric_only=False, dropna=True)
    cat_missing_impute.to_csv("./artifacts/cat_missing_impute.csv")
    cat_missing_impute

# Continuous variables missing values
    cont_vars = cont_vars.apply(impute_missing_values)
    cont_vars.apply(describe_numeric_col).T

    cat_vars.loc[cat_vars['customer_code'].isna(),'customer_code'] = 'None'
    cat_vars = cat_vars.apply(impute_missing_values)
    cat_vars.apply(lambda x: pd.Series([x.count(), x.isnull().sum()], index = ['Count', 'Missing'])).T
    cat_vars







    scaler_path = "./artifacts/scaler.pkl"

    scaler = MinMaxScaler()
    scaler.fit(cont_vars)

    joblib.dump(value=scaler, filename=scaler_path)
    print("Saved scaler in artifacts")

    cont_vars = pd.DataFrame(scaler.transform(cont_vars), columns=cont_vars.columns)
    cont_vars


    cont_vars = cont_vars.reset_index(drop=True)
    cat_vars = cat_vars.reset_index(drop=True)
    data = pd.concat([cat_vars, cont_vars], axis=1)
    print(f"Data cleansed and combined.\nRows: {len(data)}")
    data


    import json

    data_columns = list(data.columns)
    with open('./artifacts/columns_drift.json','w+') as f:           
        json.dump(data_columns,f)
        
    data.to_csv('./artifacts/training_data.csv', index=False)




    data['bin_source'] = data['source']
    values_list = ['li', 'organic','signup','fb']
    data.loc[~data['source'].isin(values_list),'bin_source'] = 'Others'
    mapping = {'li' : 'socials', 
            'fb' : 'socials', 
            'organic': 'group1', 
            'signup': 'group1'
            }

    data['bin_source'] = data['source'].map(mapping)


    data.to_csv('./artifacts/train_data_gold.csv', index=False)

    print('Features completed and saved as train_data_gold.csv')




if __name__ == "__main__":
    feature_selection()