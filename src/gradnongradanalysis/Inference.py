"""
Data Visualization Module for Graduate Employment Data Analysis

This module contains the Inference class, designed to provide a range of visualization functionalities
for exploring and interpreting graduate employment data. The class offers methods to create informative
and interactive plots that can help in understanding the trends, relationships, and distributions within
the data.

The Inference class includes methods for the following visualizations:

- `plot_employment_rate_vs_median_salary(sorted_majors)`: Generates a scatter plot showing the relationship
  between employment rates and median salaries for different majors.

- `plot_graduate_premium(sorted_premium_by_category)`: Creates a bar plot to visualize the average graduate
  premium by major categories.

- `plot_graduate_outcomes(top_n)`: Produces a series of bar plots comparing employment outcomes for the top 'n'
  majors based on the total number of graduates. It includes employment rates, unemployment rates, and median salary
  differences.

- `plot_employment_rates(sorted_data)`: Generates a horizontal bar plot to visualize employment rates by major
  category.

Each method in the Inference class is designed to make the analysis of graduate employment data more intuitive and
accessible, providing insights through clear and concise visual representations. These visualizations can be
particularly useful for educational institutions, policy makers, and students in understanding the employment
landscape and making informed decisions.

"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class Inference:
    """
    The Inference class is designed for visualizing various aspects of employment outcomes based on academic majors.
    It provides a series of plotting methods to explore and present data in an intuitive and informative manner.
    This class is especially useful for analyzing relationships between academic qualifications and employment metrics
    such as employment rates, median salaries, and graduate premiums.

    Methods include:
    - plot_employment_rate_vs_median_salary: Creates a scatter plot to visualize the relationship between employment rates
      and median salaries for different majors.
    - plot_graduate_premium: Generates a bar plot showing the average graduate premium by major categories.
    - plot_graduate_outcomes: Produces a series of bar plots comparing employment outcomes for top majors.
    - plot_employment_rates: Visualizes employment rates by major category using a horizontal bar plot.

    Attributes:
    - df (DataFrame): A pandas DataFrame containing the data to be analyzed and visualized.

    This class is valuable for researchers, educators, and policy makers who wish to understand and communicate
    the impact of higher education on career outcomes. It allows for a clear and comparative visualization of
    how different academic fields fare in the job market.

    Parameters:
    - df (DataFrame): A pandas DataFrame containing relevant data for visualization.
    """

    def __init__(self, df):
        self.df = df

    def plot_employment_rate_vs_median_salary(self, sorted_majors):
        """
        Generates a scatter plot visualizing the relationship between employment rate and median salary for different majors.

        This function creates a scatter plot where each point represents a major. The x-axis represents the employment rate,
        and the y-axis represents the median salary. Points are colored based on major categories.

        Parameters:
        - sorted_majors (DataFrame): A pandas DataFrame containing sorted data of majors,
        including their employment rates and median salaries.

        """

        plt.figure(figsize=(15, 11))
        sns.scatterplot(
            data=sorted_majors,
            x="Employment_Rate",
            y="Grad_median",
            hue="Major_category",
            s=100,
            alpha=0.7,
        )

        plt.title("Employment Rate vs Median Salary by Major")
        plt.xlabel("Employment Rate")
        plt.ylabel("Median Salary")
        plt.legend(title="Major Category", bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.show()

    def plot_graduate_premium(self, sorted_premium_by_category):
        """
        Generates a bar plot for the average graduate premium by major categories.

        This function creates a bar plot where each bar represents a major category.
        The height of the bar indicates the average graduate premium in that category.
        The plot is useful for visualizing and comparing the economic value of higher
        education across different fields.

        Parameters:
        - sorted_premium_by_category (Series): A pandas Series containing the sorted
        average graduate premiums by major category.

        """

        plt.figure(figsize=(10, 6))
        sorted_premium_by_category.plot(kind="bar")
        plt.title("Average Graduate Premium by Major Category")
        plt.xlabel("Major Category")
        plt.ylabel("Average Graduate Premium")
        plt.xticks(rotation=90)
        plt.show()

    def plot_graduate_outcomes(self, top_n=10):
        """
        Generates a series of plots comparing employment outcomes for the top 'n' majors
        based on the total number of graduates.

        This function creates three bar plots:
        1. Employment Rate by Major
        2. Unemployment Rate by Major
        3. Median Salary Difference (Graduates vs Non-Graduates) by Major

        Parameters:
        dataframe (pd.DataFrame): A DataFrame containing comparison data between graduates and non-graduates.
                                This DataFrame should include the following columns:
                                'Major', 'Grad_employment_rate', 'Nongrad_employment_rate',
                                'Grad_unemployment_rate', 'Nongrad_unemployment_rate',
                                'Median_salary_difference', 'Grad_total'.
        top_n (int): Number of top majors to be included in the plot based on the total number of graduates.


        """

        # Selecting a subset of majors for a clearer plot
        top_majors = self.df.nlargest(top_n, "Grad_total")

        # Plotting
        plt.figure(figsize=(15, 8))

        # Employment Rate Plot
        plt.subplot(1, 3, 1)
        sns.barplot(
            data=top_majors,
            x="Grad_employment_rate",
            y="Major",
            color="blue",
            label="Graduates",
        )
        sns.barplot(
            data=top_majors,
            x="Nongrad_employment_rate",
            y="Major",
            color="red",
            alpha=0.6,
            label="Non-Graduates",
        )
        plt.title("Employment Rate by Major")
        plt.xlabel("Employment Rate")
        plt.ylabel("Major")
        plt.legend()

        # Unemployment Rate Plot
        plt.subplot(1, 3, 2)
        sns.barplot(
            data=top_majors,
            x="Grad_unemployment_rate",
            y="Major",
            color="green",
            label="Graduates",
        )
        sns.barplot(
            data=top_majors,
            x="Nongrad_unemployment_rate",
            y="Major",
            color="orange",
            alpha=0.6,
            label="Non-Graduates",
        )
        plt.title("Unemployment Rate by Major")
        plt.xlabel("Unemployment Rate")
        plt.ylabel("")
        plt.yticks([])
        plt.legend()

        # Median Salary Difference Plot
        plt.subplot(1, 3, 3)
        sns.barplot(
            data=top_majors, x="Median_salary_difference", y="Major", color="purple"
        )
        plt.title("Median Salary Difference (Graduates - Non-Graduates)")
        plt.xlabel("Median Salary Difference")
        plt.ylabel("")
        plt.yticks([])

        plt.tight_layout()
        plt.show()

    def plot_employment_rates(self, sorted_data):
        """
        Plot the employment rates by major category.

        This function creates a horizontal bar plot to visualize the employment rates for different major categories.
        The categories are sorted by their employment rates, and the plot displays these rates along the x-axis.

        Parameters:
        - sorted_data (pd.DataFrame): A DataFrame containing at least two columns:
            * 'Major_category': The name of the major category.
            * 'Employment_Rate': The employment rate for the respective category.

        Returns:
        - None: This function does not return anything. It directly displays the plot.

        Note:
        - This function assumes that 'sorted_data' is already sorted in the desired order for visualization.
        """
        plt.figure(figsize=(12, 8))
        plt.barh(
            sorted_data["Major_category"],
            sorted_data["Employment_Rate"],
            color="skyblue",
        )
        plt.xlabel("Employment Rate")
        plt.ylabel("Major Category")
        plt.title("Employment Rates by Major Category")
        plt.gca().invert_yaxis()  # Invert y-axis for better readability
        plt.show()
