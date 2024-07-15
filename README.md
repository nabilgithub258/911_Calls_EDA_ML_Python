# 911 Calls Dataset Analysis and Prediction

## Overview

This project involves an extensive analysis and predictive modeling on a dataset containing 100,000 records of 911 emergency calls. Our objective is to extract meaningful insights and develop predictive models to enhance the efficiency of emergency response services. The project includes comprehensive Exploratory Data Analysis (EDA), data cleaning, visualization, and predictive modeling using advanced machine learning techniques.

## Table of Contents

- [Project Description](#project-description)
- [Dataset](#dataset)
- [Installation](#installation)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Modeling](#modeling)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Description

The goal of this project is to analyze the 911 calls dataset to identify patterns and trends, and to build predictive models that can forecast critical aspects of emergency calls such as call volume, peak times, and high-incident areas. This can help in optimizing emergency response strategies and improving public safety.

## Dataset

The dataset contains information on 100,000 emergency 911 calls. Key features include:

- `lat`: Latitude of the call
- `lng`: Longitude of the call
- `desc`: Description of the emergency
- `zip`: Zip code where the call was made
- `title`: Title of the emergency
- `timeStamp`: Timestamp of the call
- `twp`: Township where the call occurred
- `addr`: Address of the emergency
- `e`: Variable indicating the emergency

## Installation

To run this project locally, please ensure you have Python installed.


## Exploratory Data Analysis

The EDA involves:

- Cleaning the dataset to handle missing values and incorrect data.
- Visualizing the distribution of call types, response times, and geographic patterns.
- Identifying trends and anomalies in the data.

## Modeling

We developed various predictive models to forecast reasons for the call:

- EMS
- Fire
- Traffic

The models used include:

- Logistic Regression
- Random Forests
- GridsearchCV + Random Forest
- KNN

## Results

Our analysis and models provided valuable insights into the 911 calls dataset. We were able to predict reasons for the calls before it even happened, although our model wasn't perfect by any means but it gave us an idea what the reasons might be for the calls.
We were limited by our computing power and as the dataset was massive we couldn't do more advanced methods to make the model to our desired satisfaction.

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request for any improvements or bug fixes. 

---
