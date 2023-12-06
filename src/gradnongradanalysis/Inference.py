"""
Employment Outcomes Analysis

This module provides functionality for analyzing and visualizing employment outcomes 
based on academic majors and other relevant factors. It includes the Inference class, 
which contains methods for generating insightful plots and statistical models to 
understand the relationship between academic qualifications and employment metrics.

Key features:
- Visualization of employment rates, median salaries, and graduate premiums based on academic majors.
- Comparative analysis of employment outcomes for graduates versus non-graduates.
- Machine learning model training and evaluation to predict employment outcomes.

The module is intended for use by researchers, educators, and policymakers who are 
interested in the impact of academic qualifications on career outcomes. It is also 
useful for students and career advisors seeking data-driven insights into the job market.

Dependencies:
- pandas for data manipulation
- numpy for numerical operations
- seaborn and matplotlib for data visualization
- scikit-learn for machine learning models and preprocessing
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import warnings

warnings.filterwarnings("ignore")


class Inference:
    """
    Inference Class for Employment Outcomes Analysis

    This class is designed to analyze and visualize employment outcomes based on
    academic majors. It provides methods for plotting various aspects of employment
    data, helping users to explore relationships between academic qualifications and
    employment metrics such as employment rates, median salaries, and graduate premiums.

    The class is valuable for visual analysis, offering an intuitive understanding
    of how different academic fields fare in terms of career prospects.

    Attributes:
    - df (pandas.DataFrame): A DataFrame containing the data for analysis.

    Methods:
    - plot_employment_rate_vs_median_salary: Visualizes the relationship between
      employment rates and median salaries for different majors.
    - plot_graduate_premium: Shows the average graduate premium by major category.
    - plot_graduate_outcomes: Compares employment outcomes for top majors.
    - plot_employment_rates: Displays employment rates by major category.
    - prepare_data: Prepares the dataset for analysis by creating a binary target variable.
    - select_features: Selects and returns relevant features for modeling.
    - split_data: Splits the dataset into training and testing sets.
    - train_and_evaluate: Trains and evaluates multiple machine learning models.

    The Inference class is intended for researchers, educators, policy makers, as well
    as students and career advisors, providing a clear visualization and analysis of
    employment outcomes across various academic disciplines.
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

    def select_features(self, target_column):
        """
        Selects relevant features for the classification task.

        Parameters:
        target_column (str): The name of the target column.

        Returns:
        pd.DataFrame: A DataFrame with selected features.
        """
        features = self.df.select_dtypes(include=["float64", "int64"]).drop(
            [target_column, "Target"], axis=1
        )
        return features

    def split_data(self, features, target, test_size=0.3, random_state=42):
        """
        Splits the data into training and test sets.

        Parameters:
        features (pd.DataFrame): The features of the dataset.
        target (pd.Series): The target variable.
        test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.3.
        random_state (int, optional): Controls the shuffling applied to the data before applying the split. Defaults to 42.

        Returns:
        tuple: A tuple containing the training and test sets (X_train, X_test, y_train, y_test).
        """
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """
        Trains multiple classifiers and evaluates their performance.

        Parameters:
        X_train (pd.DataFrame): Training set features.
        X_test (pd.DataFrame): Test set features.
        y_train (pd.Series): Training set target.
        y_test (pd.Series): Test set target.

        Returns:
        dict: A dictionary containing the accuracy scores and classification reports for each classifier.
        """
        classifiers = {
            "Logistic Regression": LogisticRegression(),
            "SVM": SVC(),
            "Random Forest": RandomForestClassifier(),
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Naive Bayes": GaussianNB(),
        }

        results = {}
        for name, clf in classifiers.items():
            pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", clf)])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            results[name] = {"accuracy": accuracy, "report": report}

        return results
