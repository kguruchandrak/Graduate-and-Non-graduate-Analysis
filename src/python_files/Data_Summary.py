"""
Data Summary Module.

This module contains the DataSummary class which provides various methods for summarizing a dataset. 
The class is designed to be used with pandas DataFrames and offers functionalities like printing a 
summary of use cases for the dataset, determining data types of attributes, 
computing summary statistics, and identifying missing values.

The class can be particularly useful in scenarios such as data preprocessing, exploratory 
data analysis, and providing insights for decision-making in areas such as career guidance,
educational policy analysis, and academic research.

"""

class DataSummary:
    """
    A class for performing data analysis and summarization tasks on a pandas DataFrame.

    This class provides methods to analyze a dataset for various use cases, including:
    - Understanding the structure of the dataset (data types, missing values).
    - Computing summary statistics for numerical columns.
    - Printing a summary of potential use cases of the dataset.

    The class is designed to work with pandas DataFrames, making it a versatile tool for
    preliminary data exploration and analysis in data science and research projects.

    Methods
    -------
    use_case():
        Prints a summary of the dataset, outlining potential use cases like career guidance,
        educational policy analysis, and academic research.

    Attributes
    ----------
    df : DataFrame
        A pandas DataFrame containing the data to be analyzed.
    """

    def __init__(self,df):
        self.df = df 

    def use_case(self):
        """
        Print a summary of the dataset.

        This method provides information about the dataset, including the data source,
        common attributes for graduates and non-graduates, data types of each attribute,
        and the summary of the DataFrame's data types.
        """
        print('''Use Cases of the Dataset''')
        print("----------------------------------------------------")
        print('''The dataset appears to be designed for use cases like:
    1. Career Guidance: Providing insights into employment outcomes for various academic majors.
    2. Educational Policy Analysis: Helping policymakers understand the job market outcomes of different fields of study.
    3. Academic Research: Facilitating research in areas like labor economics, education, and social sciences.''')
        return 
    

    def get_data_types(self):
        """
        Returns the data types of each attribute in the given DataFrame.

        Parameters:
        - self.df (DataFrame): The dataset whose data types are to be determined.

        Returns:
        - Series: A pandas Series containing the data types of each column in the dataset.
        """
        data_types = self.df.dtypes
        return data_types


    





    

