import pandas as pd
import os
import warnings
import datetime
import json



#RAW_DATA_PATH = '../data/raw/raw_data.csv'
RAW_DATA_PATH = 'data/raw/raw_data.csv'

MIN_DATE = "2024-01-01"
MAX_DATE = "2024-01-31"



def read_data(data_path=RAW_DATA_PATH, min_date=MIN_DATE, max_date=MAX_DATE):


    # Create artifacts dir
    os.makedirs("artifacts",exist_ok=True)

    warnings.filterwarnings('ignore') # Ignore warnings
    pd.set_option('display.float_format',lambda x: "%.3f" % x) # Floats will display with 3 decimals

    data = pd.read_csv(data_path)

    if not max_date: # If max date not provided, set it to today's date
        max_date = pd.to_datetime(datetime.datetime.now().date()).date()
    else: # Otherwise convert given max date into datetime obj
        max_date = pd.to_datetime(max_date).date()

    min_date = pd.to_datetime(min_date).date() # Do the same to min date

    # Time limit data (filter rows so that its between min and max)
    data["date_part"] = pd.to_datetime(data["date_part"]).dt.date
    data = data[(data["date_part"] >= min_date) & (data["date_part"] <= max_date)]

    min_date = data["date_part"].min() # Recalculate min and max data
    max_date = data["date_part"].max() # after filtering
    date_limits = {"min_date": str(min_date), "max_date": str(max_date)} # Store

    with open("./artifacts/date_limits.json", "w") as f:
        json.dump(date_limits, f) # Save the artifact

    #print('Data successfully read, time filtered and saved as raw_data_framed.csv')

    # Save data into artifacts
    data.to_csv('./artifacts/raw_data_framed.csv', index=False)




if __name__ == "__main__":
    read_data()