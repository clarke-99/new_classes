# New Custom Classes

## Overview
This repository contains the latest iteration of my custom classes aimed at automating data preprocessing, exploratory data analysis (EDA), and feature engineering. These classes are designed to enhance efficiency, reduce manual intervention, and improve the overall workflow in machine learning projects.

## Classes Included

- **Class 1: `eda_feateng`**
  - **Purpose:** Streamlines the preprocessing and exploratory data analysis of datasets, providing a comprehensive suite of tools for initial data exploration and preparation.
  - **Key Features:** 
    - Displays summary statistics for a quick overview of the dataset.
    - Determines the normality of distributions to guide further analysis.
    - Creates various visualizations:
      - Correlation heatmaps
      - QQ plots
      - Box plots
      - Violin plots
      - Histograms
    - Analyses feature distributions to identify potential data issues or characteristics that may influence model performance.
  - **New Features:**
    - Automatically determines the type of categorical data (ordinal, ratio, datetime, interval, or nominal).
    - Allows users to modify the assumed data type for greater flexibility.
    - Uses stored data types to automate encoding, including the option for users to generate custom mappings for ordinal encoding.
    - Tests for normality using the most appropriate statistical test based on sample size, ensuring accurate results.

## Future Work
Development of these classes is ongoing. The next steps include:
- Implementing methods for identifying and removing outliers using a variety of techniques (parametric and non-parametric) to ensure that true outliers are detected while preserving important data points.
- Introducing advanced data engineering methods to enhance feature creation and selection.
- Adding multivariate analysis tools to better understand the relationships between multiple variables.
- Developing methods for assessing collinearity to prevent issues with multicollinearity in models.
- Incorporating polynomial analysis for more complex feature engineering and model improvement.

## Contributing
Contributions are welcome! If you have suggestions for improving or refactoring these classes, please feel free to open issues or submit pull requests. Your input will help make these tools even more robust and versatile.
