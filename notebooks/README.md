# Sentiment Analysis of News Articles by Publisher

This Jupyter Notebook analyzes the sentiment distribution of news articles by publisher. The goal is to identify dominant publishers and categorize the sentiments of their articles into positive, negative, or neutral.

## Overview

The notebook performs the following steps:

1. **Calculate Publisher Statistics**: Identify the number and percentage of articles published by each publisher.
2. **Identify Dominant Publishers**: Filter publishers who have contributed to more than 10% of the total articles.
3. **Sentiment Categorization**: Classify each article's sentiment as positive, negative, or neutral based on a numerical `sentiment` score.
4. **Sentiment Distribution by Publisher**: Calculate and display the distribution of sentiment categories for each dominant publisher.

## Key Functions

- **Publisher Statistics**: Uses `value_counts()` to count articles per publisher and compute their percentage of the total articles.
- **Sentiment Classification**: Utilizes a lambda function to categorize each article's sentiment score into positive, negative, or neutral.
- **Sentiment Distribution**: Aggregates sentiment counts for each dominant publisher and displays the results.

## Example Output

The notebook outputs the dominant publishers along with the distribution of sentiments for their articles. An example of the output format:

## Requirements

To run this notebook, you need the following:

- **Python 3.x**: Ensure you have Python 3.x installed.
- **Pandas**: Install pandas using `pip install pandas`.
- **Jupyter Notebook**: Install Jupyter Notebook using `pip install notebook`.

## Getting Started

1. Clone this repository:
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Jupyter Notebook:
    ```bash
    jupyter notebook analysis.ipynb
    ```

## How to Use

- Open the notebook and run all cells to analyze the data and visualize the sentiment distribution by publisher.
- Modify the threshold for dominant publishers or sentiment classification criteria as needed to suit your data.

## Contributing

Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

---

Happy analyzing! 📊