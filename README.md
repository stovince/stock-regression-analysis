# stock-regression-analysis

## 🚀 Project Oversight
A project built in Python for fetching, processing and analyzing historical stock data with the help of regression. This project leverages pandas, NumPy, matplotlib and scikit-learn to create, evaluate and visualize predictive models.

## Main Objectives
- Prediction of future stock movement by using historical data.
- Comparison of different regression-models (linear, Ridge, RandomForest, etc.)
- Learn to utilize and leverage regression for analysis.

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
