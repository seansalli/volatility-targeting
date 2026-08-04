Volatility Targeting Backtest
------------------------------
Backtest of volatility-targeted position sizing on the S&P 500 from 2005 to 2025.


What the project is
-------------------
In this project, I built a trading strategy that changes how much money is invested based on the market's current volatility; the strategy would invest less when volatility is higher, and more when volatility is lower. 

I compared three volatility estimators (a rolling average, a EWMA, and a GARCH model) and tested a hybrid strategy which switches between them. Then, I stress-tested everything with transaction costs, an out-of-sample test period, and statistical significance testing. 

What's in this repository
-------------------------
data/    (Raw price data and processed vol estimates)
src/     (Six Python scripts, run in order below)


How to run
----------
1. Install the required packages (pip install -r requirements.txt)

2. From inside the src/ folder, run the scripts in this order:
	python vol_estimators.py
	python vol_targeting_backtest.py
	python hybrid_vol_targeting.py
	python rolling_sharpe.py
	python transaction_costs.py
	python significance_testing.py

Each script builds off the output of another before it, so running them in order is important.

What was found in this project
-------------------------------
Volatility targeting does reduce portfolio drawdowns and volatility compared to just holding the market in a meaningful manner. A more complex version which switches based on current market volatility looked even better at first; however, that advantage disappeared once trading costs were included, leading to none of the differences between strategies to be statistically significant.

Author: Sean Salli
Summer 2026
