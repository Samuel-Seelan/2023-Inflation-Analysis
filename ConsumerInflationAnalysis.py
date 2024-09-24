# -*- coding: utf-8 -*-

"""
Income and CPI Data Analysis

This script loads, processes, and analyzes income and CPI data to study the relationship
between cumulative CPI changes and AGI changes using various regression techniques.

Created on Nov 1, 2023

@author: Samuel Seelan
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.preprocessing import RobustScaler
from tabulate import tabulate
import matplotlib.pyplot as plt



# Dictionary of income filenames by year
income_data = {
    2020: "20zpallagi.csv",
    2019: "19zpallagi.csv",
    2018: "18zpallagi.csv",
    2017: "17zpallagi.csv",
    2016: "16zpallagi.csv",
    2015: "15zpallagi.csv",
    2014: "14zpallagi.csv",
    2013: "13zpallagi.csv",
    2012: "12zpallagi.csv",
    2011: "11zpallagi.csv",
}

CPI_data = pd.read_csv('CPALTT01USM657N.csv')

def load_income_data(data_dict):
    """
    Load income data from multiple files into a single dataframe.

    Parameters:
    - data_dict (dict): Dictionary of income filenames by year.

    Returns:
    - income_df (pd.DataFrame): Combined income data dataframe.
    """
    income_dataframes = []

    for year, file_name in data_dict.items():
        try:
            df_temp = pd.read_csv(file_name, nrows=0)
            available_columns = df_temp.columns

            usecols = ['zipcode', 'N1']
            if 'agi_stub' in available_columns:
                usecols.append('agi_stub')
            if 'A00100' in available_columns:
                usecols.append('A00100')

            df = pd.read_csv(file_name, usecols=usecols, dtype={'zipcode': str, 'agi_stub': str, 'A00100': float, 'N1': float})

            df['zipcode'] = df['zipcode'].apply(lambda x: x.zfill(5))


            df = df[(df['zipcode'] > '00501') & (df['zipcode'] < '99960')]

            if 'agi_stub' in df.columns:
                df['zip_agi'] = df['zipcode'] + df['agi_stub']
            else:
                df['zip_agi'] = df['zipcode']


            df['Year'] = year


            income_dataframes.append(df)

        except pd.errors.ParserError as e:
            print(f"Error reading file {file_name}: {e}")
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

    if income_dataframes:
        income_df = pd.concat(income_dataframes, ignore_index=True)
    else:
        raise ValueError("No valid dataframes to concatenate.")

    return income_df


income_df = load_income_data(income_data)

CPI_data['Year'] = pd.to_datetime(CPI_data['DATE']).dt.year
CPI_data['CPI_change'] = CPI_data['CPALTT01USM657N'].pct_change()

yearly_CPI = CPI_data.groupby('Year')['CPALTT01USM657N'].sum().reset_index(name='Yearly_Change')
yearly_CPI['CPI_cumulative'] = yearly_CPI['Yearly_Change'].cumsum()
yearly_CPI['CPI_cumulative_change'] = yearly_CPI['CPI_cumulative'].pct_change() * 100

income_df.sort_values(by=['zip_agi', 'Year'], inplace=True)

# Calculate yearly change of AGI data as a percentage
income_df['AGI_change'] = income_df.groupby('zip_agi')['A00100'].pct_change() * 100

# Merge the income dataframe with the yearly_CPI dataframe on the 'Year' column
merged_data = income_df[(~income_df['AGI_change'].isna()) & (income_df['A00100'].notnull())
                        & (income_df['A00100'] != 0) & (income_df['AGI_change'].between(-999, 999))].merge(yearly_CPI, on='Year', how='left')


def plot_linear_regression(X, y):
    """
    Plot linear regression of CPI_cumulative_change predicting AGI_change, non adjusted.

    Parameters:
    - X (np.ndarray): Input data for CPI_cumulative_change.
    - y (np.ndarray): Target data for AGI_change.
    """
    linear_model = LinearRegression()
    linear_model.fit(X, y)

    # Scatter plot of data
    plt.scatter(X, y, color='blue', label='Data')
    # Plot the linear regression line
    plt.plot(X, linear_model.predict(X), color='green', label='Linear Regression')
    plt.xlabel('CPI Change (%)')
    plt.ylabel('AGI Change (%)')
    plt.title('Linear Regression: CPI Cumulative Change Predicting AGI Change')
    plt.legend(loc='upper right')

    # Calculate correlation and R-squared
    correlation_linear = np.corrcoef(X.flatten(), y.flatten())[0, 1]
    r_squared_linear = linear_model.score(X, y)

    # Add correlation and R-squared as text on the plot
    plt.text(0.05, 0.9, f'Correlation: {correlation_linear:.4f}', transform=plt.gca().transAxes)
    plt.text(0.05, 0.85, f'R-squared: {r_squared_linear:.4f}', transform=plt.gca().transAxes)

    plt.show()


def plot_shaped_linear_regression(X, y):
    """
    Plot shaped linear regression of CPI_cumulative_change predicting AGI_change.
    Only considers more than 500 returns data and non-zero AGI_change.

    Parameters:
    - X (np.ndarray): Input data for CPI_cumulative_change.
    - y (np.ndarray): Target data for AGI_change.
    """
    # Trim the data to only include more than 500 returns and non-zero AGI_change
    trimmed_merged_data = merged_data[merged_data['N1'] > 500]
    trimmed_merged_data = trimmed_merged_data[trimmed_merged_data['AGI_change'] != 0]

    # Create modified X and y values for the model
    X = trimmed_merged_data['CPI_cumulative_change'].values.reshape(-1, 1)
    y = trimmed_merged_data['AGI_change'].values.reshape(-1, 1)

    # Scale the y values
    scaler = RobustScaler()
    y = scaler.fit_transform(y)

    linear_model = LinearRegression()
    linear_model.fit(X, y)

    # Scatter plot of data
    plt.scatter(X, y, color='blue', label='Data')
    # Plot the linear regression line
    plt.plot(X, linear_model.predict(X), color='green', label='Linear Regression')
    plt.xlabel('CPI Change (%)')
    plt.ylabel('AGI Change (%)')
    plt.title('Shaped Linear Regression: CPI Cumulative Change Predicting AGI Change')
    plt.legend(loc='upper right')

    # Calculate correlation and R-squared
    correlation_linear = np.corrcoef(X.flatten(), y.flatten())[0, 1]
    r_squared_linear = linear_model.score(X, y)

    # Add correlation and R-squared as text on the plot
    plt.text(0.05, 0.9, f'Correlation: {correlation_linear:.2f}', transform=plt.gca().transAxes)
    plt.text(0.05, 0.85, f'R-squared: {r_squared_linear:.2f}', transform=plt.gca().transAxes)

    plt.show()


# Function to plot RANSAC regression of CPI_cumulative_change predicting AGI_change
def plot_ransac_regression(X, y):
    """
    Plot RANSAC regression of CPI_cumulative_change predicting AGI_change. Shapes data a bit better,
    only considers more than 500 returns data and non-zero AGI_change.

    Parameters:
    - X (np.ndarray): Input data for CPI_cumulative_change.
    - y (np.ndarray): Target data for AGI_change.
    """
    # Trim the data to only include more than 500 returns and non-zero AGI_change
    trimmed_merged_data = merged_data[merged_data['N1'] > 500]
    trimmed_merged_data = trimmed_merged_data[trimmed_merged_data['AGI_change'] != 0]

    # Create modified X and y values for the model
    X = trimmed_merged_data['CPI_cumulative_change'].values.reshape(-1, 1)
    y = trimmed_merged_data['AGI_change'].values.reshape(-1, 1)

    # Scale the y values
    scaler = RobustScaler()
    y = scaler.fit_transform(y)

    # Create a RANSAC Regression model and fit it to the data
    ransac_model = RANSACRegressor(LinearRegression())
    ransac_model.fit(X, y)

    # Scatter plot of data
    plt.scatter(X, y, color='blue', label='Data')
    # Plot the RANSAC regression line
    plt.plot(X, ransac_model.predict(X), color='red', label='RANSAC Regression')
    plt.xlabel('CPI Change (%)')
    plt.ylabel('AGI Change (%)')
    plt.title('RANSAC Non-zero Adjusted Regression: CPI Cumulative Change Predicting AGI Change')
    plt.legend(loc='upper right')

    # Calculate correlation and R-squared
    correlation_ransac = np.corrcoef(X.flatten(), y.flatten())[0, 1]
    r_squared_ransac = ransac_model.score(X, y)

    # Add correlation and R-squared as text on the plot
    plt.text(0.05, 0.9, f'Correlation: {correlation_ransac:.2f}', transform=plt.gca().transAxes)
    plt.text(0.05, 0.85, f'R-squared: {r_squared_ransac:.2f}', transform=plt.gca().transAxes)

    plt.show()
    # Sort the merged_data dataframe by AGI_change correlation to CPI_cumulative_change in descending order
    sorted_data = merged_data.sort_values(by='CPI_cumulative_change', ascending=False)

    # Get the top 10 zip codes with the highest AGI_change correlation
    top_10_zipcodes = sorted_data['zip_agi'].head(10)


    # Print the top 10 zip codes with their correlations and R-squared values
    print("Top 10 Zip Codes with Highest AGI_Change Correlation to CPI_cumulative_change:")
    table_data = []
    for zipcode in top_10_zipcodes:
        correlation = merged_data.loc[merged_data['zip_agi'] == zipcode, 'CPI_cumulative_change'].values[0]
        r_squared = merged_data.loc[merged_data['zip_agi'] == zipcode, 'AGI_change'].values[0]
        table_data.append([zipcode, correlation, r_squared])

    table_headers = ["Zip Code", "Correlation", "R-squared"]
    print(tabulate(table_data, headers=table_headers))



# Create X and y values for the model
X = merged_data['CPI_cumulative_change'].values.reshape(-1, 1)
y = merged_data['AGI_change'].values.reshape(-1, 1)


def calculate_correlation_r_squared(merged_data):
    """
    Calculate the correlation and R-squared values for each agi_stub in the merged_data.

    Parameters:
    merged_data (DataFrame): The merged data containing the necessary columns.

    Returns:
    correlation_dict (dict): A dictionary containing the correlation values for each agi_stub.
    r_squared_dict (dict): A dictionary containing the R-squared values for each agi_stub.
    """
    # Create a dictionary to store the correlation and R-squared values for each agi_stub
    correlation_dict = {}
    r_squared_dict = {}

    # Create a list of unique agi_stub values
    agi_stubs = merged_data['agi_stub'].unique()

    # Iterate over each agi_stub value
    for agi_stub in agi_stubs:
        # Filter the data for the current agi_stub value
        filtered_data = merged_data[merged_data['agi_stub'] == agi_stub]

        # Trim the data to only include more than 500 returns and non-zero AGI_change
        trimmed_data = filtered_data[filtered_data['N1'] > 500]
        trimmed_data = trimmed_data[trimmed_data['AGI_change'] != 0]

        # If the filtered data is empty, skip it
        if trimmed_data.empty:
            print(f"No data available for agi_stub {agi_stub} after filtering. Skipping...")
            continue

        # Create modified X and y values for the model
        X = trimmed_data['CPI_cumulative_change'].values.reshape(-1, 1)
        y = trimmed_data['AGI_change'].values.reshape(-1, 1)

        # If y is empty after filtering, skip this agi_stub
        if len(y) == 0:
            print(f"No valid data for agi_stub {agi_stub}. Skipping...")
            continue

        # Scale the y values
        scaler = RobustScaler()
        y = scaler.fit_transform(y)

        # Create a RANSAC Regression model and fit it to the data
        ransac_model = RANSACRegressor(LinearRegression())
        ransac_model.fit(X, y)

        # Calculate correlation and R-squared
        correlation = np.corrcoef(X.flatten(), y.flatten())[0, 1]
        r_squared = ransac_model.score(X, y)

        # Store the correlation and R-squared values in the dictionaries
        correlation_dict[agi_stub] = correlation
        r_squared_dict[agi_stub] = r_squared

        # Scatter plot of data
        plt.scatter(X, y, label=f'agi_stub {agi_stub}')

        # Plot the RANSAC regression line
        plt.plot(X, ransac_model.predict(X))

    plt.xlabel('CPI Change (%)')
    plt.ylabel('AGI Change (%)')
    plt.title('RANSAC Regression: CPI Cumulative Change Predicting AGI Change')
    plt.legend()
    plt.show()

    # Create a table of correlation and R-squared values
    table_data = []
    for agi_stub, correlation in correlation_dict.items():
        r_squared = r_squared_dict[agi_stub]
        table_data.append([agi_stub, correlation, r_squared])

    # Print correlation and R-squared values to the terminal
    print('Correlation and R-squared values:')
    print('-' * 40)
    print(f"| {'agi_stub':<8} | {'Correlation':<12} | {'R-squared':<10} |")
    print('-' * 40)
    for agi_stub, correlation in correlation_dict.items():
        r_squared = r_squared_dict[agi_stub]
        print(f"| {agi_stub:<8} | {correlation:.2f}        | {r_squared:.2f}     |")
    print('-' * 40)

    overall_correlation = np.mean(list(correlation_dict.values()))
    overall_r_squared = np.mean(list(r_squared_dict.values()))
    print(f"| {'Overall':<8} | {overall_correlation:.2f}        | {overall_r_squared:.2f}     |")
    print('-' * 40)

    return correlation_dict, r_squared_dict


def main():
    # Load income data into separate dataframes
    income_df = load_income_data(income_data)

    # Load CPI data
    CPI_data = pd.read_csv('CPALTT01USM657N.csv')

    # Calculate yearly change of CPI data
    CPI_data['Year'] = pd.to_datetime(CPI_data['DATE']).dt.year
    CPI_data['CPI_change'] = CPI_data['CPALTT01USM657N'].pct_change()

    # Calculate sum of CPI values for each year to create a yearly CPI view
    yearly_CPI = CPI_data.groupby('Year')['CPALTT01USM657N'].sum().reset_index(name='Yearly_Change')
    yearly_CPI['CPI_cumulative'] = yearly_CPI['Yearly_Change'].cumsum()
    yearly_CPI['CPI_cumulative_change'] = yearly_CPI['CPI_cumulative'].pct_change() * 100

    # Sort the dataframe by zip_agi and year
    income_df.sort_values(by=['zip_agi', 'Year'], inplace=True)

    # Calculate yearly change of AGI data as a percentage
    income_df['AGI_change'] = income_df.groupby('zip_agi')['A00100'].pct_change() * 100

    # Merge the income dataframe with the yearly_CPI dataframe on the 'Year' column
    merged_data = income_df[(~income_df['AGI_change'].isna()) & (income_df['A00100'].notnull())
                            & (income_df['A00100'] != 0) & (income_df['AGI_change'].between(-999, 999))].merge(yearly_CPI, on='Year', how='left')

    # Create X and y values for the model
    X = merged_data['CPI_cumulative_change'].values.reshape(-1, 1)
    y = merged_data['AGI_change'].values.reshape(-1, 1)

    # Call the necessary functions
    plot_linear_regression(X, y)
    plot_shaped_linear_regression(X, y)
    plot_ransac_regression(X, y)
    calculate_correlation_r_squared(merged_data)


if __name__ == "__main__":
    main()
