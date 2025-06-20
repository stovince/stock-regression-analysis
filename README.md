# stock-regression-analysis

## 🚀 Project Oversight
A project built in Python for fetching, processing and analyzing historical stock data with the help of regression. This project leverages pandas, NumPy, matplotlib and scikit-learn to create, evaluate and visualize predictive models.

## Main Objectives
- Prediction of future stock movement by using historical data.
- Comparison of different regression-models (linear, Ridge, RandomForest, etc.)
- Learn to utilize and leverage regression for analysis.

![image](https://github.com/user-attachments/assets/ae1c0dff-a6c2-4f00-a4bf-31886388e5a2)


## Explanation
This project performs a regression-based analysis of stock performance over a defined historical period. The main objective is to explore whether a linear relationship can be identified between time and Apple’s stock price, using statistical modeling to capture underlying trends in its historical development.
Historical stock price data for Apple is retrieved, cleaned, and structured to ensure consistency. The project focuses on modeling the relationship between the stock’s price and time by transforming the time variable into a numerical format suitable for regression analysis. In this context, time serves as the independent variable, and the adjusted closing price of Apple acts as the dependent variable.
A simple linear regression model is implemented using Python's scikit-learn library. This model attempts to fit a straight line through the historical data to quantify the long-term trend — essentially asking: “Can we represent Apple’s stock price behavior over this period with a linear function of time?”
To evaluate how well the model captures the actual price movement, the R² score (coefficient of determination) is calculated. A higher R² indicates that the linear model explains a greater portion of the variation in Apple’s stock price. The analysis is visualized through a scatter plot of the actual price data, overlaid with the regression line to clearly show the trend and model fit.

## Installation:
1. Cloning repo
  git clone https://github.com/stovince/stock-regression-analysis.git
  cd stock-regression-analysis
2. Create virtual environment
  python3 -m venv venv
  For mac: source venv/bin/activate      
  For windows: venv\Scripts\activate
3. Install dependencies
   pip install -r requirements.txt

## Experiments & Results
- Simple LR on MA_20
25/06/08
R^2 = 0.00245, MSE = 0.000512, Coef = -7.39 
No signal, prediction is flat at 0.

- RandomForest on MA_5, MA_10 and MA_20
25/06/08
R^2 = 0.32, MSE = 0.00034
Promising results, check for overfitting still required.
