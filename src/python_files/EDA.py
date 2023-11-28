"""
Exploratory Data Analysis (EDA) Module

This module contains the EDA class which is designed for performing comprehensive exploratory data 
analysis on graduate employment data. The class offers a range of functionalities to analyze, 
process, and visualize data related to various aspects of graduate outcomes, including employment 
rates, salary comparisons, and handling of missing values.

The EDA class provides the following methods:

- `get_summary_statistics()`: Computes summary statistics like mean, median, standard deviation, etc., 
  for each numeric column in the DataFrame.
- `get_missing_values_info()`: Calculates the count of missing values in each column of the DataFrame.
- `fill_missing_values()`: Fills missing values in the DataFrame, handling both numerical and categorical data.
- `analyze_and_sort_majors()`: Analyzes and sorts academic majors based on employment rate and median salary.
- `calculate_average_grad_premium()`: Calculates and sorts the average graduate premium by major categories.
- `compare_graduate_outcomes()`: Compares employment outcomes between graduates and non-graduates for each major.
- `calculate_employment_rates()`: Calculates employment rates for different major categories and sorts them.

The module is designed to provide an easy and intuitive interface for data analysts to perform 
detailed analyses of graduate employment data. It can be particularly useful in educational and policy-making 
contexts where understanding the employment landscape of graduates is crucial.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class EDA:
    def __init__(self,df):
        self.df = df 

    def get_summary_statistics(self):
        """
        Computes and returns summary statistics for each numeric column in the given DataFrame.

        This function provides descriptive statistics that summarize the central tendency,
        dispersion, and shape of a dataset's distribution, excluding NaN values.

        Parameters:
        - self.df (DataFrame): A pandas DataFrame for which the summary statistics are to be computed.

        Returns:
        - DataFrame: A DataFrame containing the summary statistics for each numeric column.
        This includes count, mean, std (standard deviation), min, 25% (1st quartile),
        50% (median), 75% (3rd quartile), and max.
        """
        summary_stats = self.df.describe()
        return summary_stats


    def get_missing_values_info(self):
        """
        Calculates and returns the count of missing (null) values in each column of the given DataFrame.

        This function is useful in data preprocessing and analysis to identify columns with missing data,
        which can be crucial for ensuring data quality and making decisions about data cleaning or imputation.

        Parameters:
        - self.df (DataFrame): The pandas DataFrame for which missing values are to be counted.

        Returns:
        - Series: A pandas Series where the index contains the column names and the values are the counts of missing values in those columns.
        """
        missing_values_info = self.df.isnull().sum()
        return missing_values_info


    def fill_missing_values(self):
        """
        Fills missing values in a pandas DataFrame.

        This function handles missing values in both numerical and categorical columns. For numerical columns,
        missing values are filled with the median value of the column. For categorical columns, missing values
        are filled with the most frequent value (mode).

        Parameters:
        - self.df (DataFrame): The pandas DataFrame with missing values to be filled.

        Returns:
        - DataFrame: The DataFrame with missing values filled.
        """

        # Impute numerical columns with median
        numerical_columns = self.df.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_columns:
            self.df[col].fillna(self.df[col].median(), inplace=True)

        # Impute categorical columns with mode
        categorical_columns = self.df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            self.df[col].fillna(self.df[col].mode()[0], inplace=True)

        return self.df


    def analyze_and_sort_majors(self):
        """
        Analyzes and sorts academic majors based on employment rate and median salary.

        This function calculates the employment rate for each major as the ratio of employed graduates to total graduates.
        It then selects relevant columns for the analysis and sorts the majors based on their employment rates and
        median salaries in descending order.

        Parameters:
        - self.df (DataFrame): A pandas DataFrame containing data about different academic majors, 
        including the number of graduates and employed graduates.

        Returns:
        - DataFrame: A sorted DataFrame with each major's employment rate and median salary.
        """

        # Calculate employment rate as the ratio of employed graduates to total graduates
        self.df['Employment_Rate'] = self.df['Grad_employed'] / self.df['Grad_total']

        # Select relevant columns for analysis
        analysis_columns = ['Major', 'Major_category', 'Grad_total', 'Grad_employed', 'Employment_Rate', 'Grad_median']

        # Analyze each major
        majors_analysis = self.df[analysis_columns]

        # Sort the data by Employment Rate and Median Salary for better visualization
        sorted_majors = majors_analysis.sort_values(by=['Employment_Rate', 'Grad_median'], ascending=[False, False])

        return sorted_majors


    def calculate_average_grad_premium(self):
        """
        Calculates and sorts the average graduate premium by major categories.

        This function computes the graduate premium for each major as the difference between
        the median salaries of graduates and non-graduates. It then groups the data by major
        category and calculates the average premium for each category. Finally, it sorts these
        averages in descending order for better visualization.

        Parameters:
        - data (DataFrame): A pandas DataFrame containing data about different academic majors,
        including median salaries for graduates and non-graduates.

        Returns:
        - Series: A pandas Series containing the sorted average graduate premiums by major category.
        """

        # Calculate the graduate premium for each major
        self.df['Grad_Premium'] = self.df['Grad_median'] - self.df['Nongrad_median']

        # Group the data by major category and calculate the average graduate premium
        average_premium_by_category = self.df.groupby('Major_category')['Grad_Premium'].mean()

        # Sort the results for better visualization
        sorted_premium_by_category = average_premium_by_category.sort_values(ascending=False)

        return sorted_premium_by_category


    def compare_graduate_outcomes(self):
        """
        Compares employment outcomes between graduates and non-graduates for each major in the given dataset.

        This function calculates the employment and unemployment rates for both graduates and non-graduates.
        It also computes the difference in median salaries between these two groups. The results are organized
        in a new DataFrame that includes these calculated metrics along with the total number of graduates per major.

        Parameters:
        self.df (pd.DataFrame): A DataFrame containing data on graduates and non-graduates across various majors.
                                The DataFrame should have the following columns:
                                'Grad_employed', 'Grad_total', 'Nongrad_employed', 'Nongrad_total',
                                'Grad_unemployed', 'Nongrad_unemployed', 'Grad_median', 'Nongrad_median', 'Major'.

        Returns:
        pd.DataFrame: A DataFrame with the calculated employment rates, unemployment rates,
                    median salary differences, and total graduates for each major.

        """

        # Calculating employment and unemployment rates
        self.df['Grad_employment_rate'] = self.df['Grad_employed'] / self.df['Grad_total']
        self.df['Nongrad_employment_rate'] = self.df['Nongrad_employed'] / self.df['Nongrad_total']
        self.df['Grad_unemployment_rate'] = self.df['Grad_unemployed'] / self.df['Grad_total']
        self.df['Nongrad_unemployment_rate'] = self.df['Nongrad_unemployed'] / self.df['Nongrad_total']

        # Comparing median salaries
        self.df['Median_salary_difference'] = self.df['Grad_median'] - self.df['Nongrad_median']

        # Organizing the data for analysis
        comparison_df = self.df[['Major', 'Grad_employment_rate', 'Nongrad_employment_rate', 
                                'Grad_unemployment_rate', 'Nongrad_unemployment_rate', 
                                'Median_salary_difference', 'Grad_total']]

        return comparison_df
    


    def calculate_employment_rates(self):
        """
        Calculate the employment rates for different major categories.

        This function takes a DataFrame containing graduate employment data, groups it by major categories,
        and calculates the employment rate for each category. The employment rate is calculated as the proportion
        of employed graduates to the total number of graduates in each category.

        Parameters:
        self.df (pd.DataFrame): A DataFrame containing columns 'Major_category', 'Grad_employed', 'Grad_unemployed',
                            and 'Grad_total'.

        Returns:
        pd.DataFrame: A DataFrame with each major category and its corresponding employment rate, sorted in
                    descending order of the employment rate.

        
        """

        # Grouping the data by 'Major_category' and calculating the aggregated data
        grouped_data = self.df.groupby('Major_category').agg({
            'Grad_employed': 'sum',  # Total number of graduates employed
            'Grad_unemployed': 'sum',  # Total number of graduates unemployed
            'Grad_total': 'sum'  # Total number of graduates
        }).reset_index()

        # Calculating the employment rate for each major category
        grouped_data['Employment_Rate'] = grouped_data['Grad_employed'] / grouped_data['Grad_total']

        # Sorting the data by Employment Rate in descending order
        sorted_data = grouped_data.sort_values(by='Employment_Rate', ascending=False)

        return sorted_data[['Major_category', 'Employment_Rate']]










    