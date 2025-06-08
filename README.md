# stock-regression-analysis
A project built in Python for fetching, processing and analyzing historical stock data with the help of regression. This project leverages pandas, NumPy, matplotlib and scikit-learn to create, evaluate and visualize predictive models.

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
# Simple LR on MA_20
25/06/08
R^2 = 0.00245, MSE = 0.000512, Coef = -7.39
Short summary: No signal, prediction is flat at 0.
