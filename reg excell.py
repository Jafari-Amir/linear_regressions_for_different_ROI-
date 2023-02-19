import os
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns
from scipy import stats



import os
import pandas as pd
import numpy as np

# Define the selected locations
selected_locations = {
    "corners": ['c1', 'c2', 'c3', 'c4'],
    "border": ['b1', 'b2', 'b3', 'b4'],
    "center": ['center']
}

# Create an empty DataFrame to hold the results
results = pd.DataFrame(columns=['location', 'slope', 'intercept'])

path = '/Users/amir1/Downloads/Female_Nicotine_CNTL/'
all_files = glob.glob(os.path.join(path, "*.csv"))
s1 = pd.DataFrame()  
s2 = pd.DataFrame()  
s3 = pd.DataFrame() 
s4 = pd.DataFrame()  
s5 = pd.DataFrame()
s6 = pd.DataFrame()
for f in all_files:
    # Read the CSV file
    df = pd.read_csv(f)
    df = df[df['ROI_transition'] == True]
    df['Frame'] = df['Frame'] / 1800
    
    # Process each location and save the results
    for location_name, locations in selected_locations.items():
        # Filter the data for the current location
        df_location = df[df["ROI_location"].isin(locations)]
        if df_location.empty:
            continue

        # Pivot the data
        pivoted = df_location.pivot_table(values="ROI_transition", index=["Frame"], columns=["ROI_location"]).fillna(0)
        cumsum_df = pivoted.cumsum(axis=1)
        sum_of_locations = cumsum_df.iloc[:, -1].cumsum()
        pivoted['sum_of_locations'] = sum_of_locations

        # Linear regression on the sum_of_locations column
        x = pivoted.index.values.reshape(-1, 1)
        y = pivoted['sum_of_locations'].values
        slope, intercept = np.polyfit(x.ravel(), y, 1)

        # Append the results to the DataFrame
        results = results.append({
            'location': location_name,
            'slope': slope,
            'intercept': intercept
        }, ignore_index=True)

# Create a dictionary to hold the results for each location
location_results = {}

# Loop through each location and save the results to a dictionary
for location_name in selected_locations.keys():
    location_df = results[results['location'] == location_name]
    location_slopes = list(location_df['slope'])
    location_intercepts = list(location_df['intercept'])
    location_results[location_name] = {
        'slopes': location_slopes,
        'intercepts': location_intercepts
    }

# Create an Excel file for each location
for location_name, location_data in location_results.items():
    location_df = pd.DataFrame(location_data)
    location_df.to_excel(f"{location_name}_results.xlsx", index=False)
