import requests
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# S&P 500 stock symbols (from June 2024)
sp500_symbols = [
    'A', 'AAL', 'AAP', 'AAPL', 'ABBV', 'ABC', 'ABMD', 'ABT', 'ACGL', 'ACN', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AGR', 
    'AIG', 'AIZ', 'AJG', 'AKAM', 'ALB', 'ALGN', 'ALK', 'ALL', 'ALLE', 'AMAT', 'AMCR', 'AMD', 'AME', 'AMGN', 'AMP', 'AMT', 'AMZN', 'ANET', 'ANSS', 'AON', 
    'AOS', 'APA', 'APD', 'APH', 'APTV', 'ARE', 'ATO', 'ATVI', 'AVB', 'AVGO', 'AVY', 'AWK', 'AXP', 'AZO', 'BA', 'BAC', 'BALL', 'BAX', 'BBWI', 'BBY', 'BDX', 
    'BEN', 'BF.B', 'BIIB', 'BIO', 'BK', 'BKNG', 'BKR', 'BLK', 'BLL', 'BMY', 'BR', 'BRK.B', 'BRO', 'BSX', 'BWA', 'BXP', 'C', 'CAG', 'CAH', 'CAR', 'CARR', 
    'CAT', 'CB', 'CBOE', 'CBRE', 'CCI', 'CCL', 'CDAY', 'CDNS', 'CDW', 'CE', 'CEG', 'CF', 'CFG', 'CHD', 'CHRW', 'CHTR', 'CI', 'CINF', 'CL', 'CLX', 'CMA', 
    'CMCSA', 'CME', 'CMG', 'CMS', 'CNC', 'CNP', 'COF', 'COO', 'COP', 'COST', 'CPB', 'CPRT', 'CPT', 'CRL', 'CRM', 'CSCO', 'CSGP', 'CSX', 'CTAS', 'CTLT', 'CTRA',
    'CTSH', 'CTVA', 'CTXS', 'CZR', 'D', 'DAL', 'DD', 'DE', 'DELL', 'DHI', 'DHR', 'DIS', 'DISA', 'DISCA', 'DISH', 'DLR', 'DLTR', 'DOV', 'DOW', 'DRE', 'DRI', 
    'DTE', 'DUK', 'DVA', 'DVN', 'DXC', 'DXCM', 'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EIX', 'EL', 'EMN', 'EMR', 'ENPH', 'EOG', 'EPAM', 'EQIX', 'EQR', 'ES', 'ESS', 
    'ETN', 'ETR', 'ETSY', 'EVRG', 'EXC', 'EXPD', 'EXPE', 'EXR', 'F', 'FANG', 'FAST', 'FBHS', 'FCX', 'FDX', 'FE', 'FFIV', 'FIS', 'FISV', 'FITB', 'FLT', 'FMC', 
    'FOX', 'FOXA', 'FRC', 'FRT', 'FSLR', 'FST', 'FTNT', 'FTV', 'GD', 'GE', 'GEHC', 'GILD', 'GIS', 'GL', 'GLW', 'GM', 'GNRC', 'GOOGL', 'GOOG', 'GPC', 'GPN', 
    'GRMN', 'GS', 'GWW', 'HAL', 'HAS', 'HBAN', 'HBI', 'HCA', 'HD', 'HES', 'HIG', 'HII', 'HOLX', 'HON', 'HPE', 'HPQ', 'HRL', 'HSIC', 'HST', 'HSY', 'HUM', 'HWM',
    'HYMC', 'IAC', 'IEX', 'IFF', 'ILMN', 'INCY', 'INTC', 'INTU', 'IP', 'IPG', 'IPGP', 'IQV', 'IR', 'IRM', 'ISRG', 'IT', 'ITW', 'IVZ', 'J', 'JCI', 'JKHY', 
    'JNJ', 'JNPR', 'JPM', 'K', 'KDP', 'KEY', 'KEYS', 'KHC', 'KIM', 'KLAC', 'KMB', 'KMI', 'KMX', 'KO', 'KR', 'L', 'LB', 'LBRDA', 'LCID', 'LDOS', 'LEN', 'LH',
    'LHX', 'LIN', 'LIT', 'LIZ', 'LKQ', 'LLY', 'LMT', 'LNC', 'LNT', 'LOW', 'LRCX', 'LUMN', 'LUV', 'LVS', 'LW', 'LYB', 'LYV', 'M', 'MA', 'MAA', 'MAR', 'MAS',
    'MASI', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDT', 'MET', 'META', 'MGM', 'MHK', 'MKC', 'MKTX', 'MLM', 'MMC', 'MMM', 'MNST', 'MO', 'MOH', 'MOS', 'MPC', 'MPWR',
    'MRK', 'MRO', 'MS', 'MSCI', 'MSFT', 'MSI', 'MTB', 'MTCH', 'MTD', 'MU', 'NCLH', 'NDAQ', 'NDSN', 'NEE', 'NEM', 'NFLX', 'NI', 'NKE', 'NOC', 'NOV', 'NRG',
    'NSC', 'NTAP', 'NTRS', 'NUE', 'NVDA', 'NVR', 'NWL', 'NWS', 'NWSA', 'O', 'ODFL', 'OGN', 'OHI', 'OKE', 'OMC', 'ON', 'ORCL', 'ORLY', 'OTIS', 'OXY',
    'PARA', 'PAYC', 'PAYX', 'PCAR', 'PCG', 'PEAK', 'PEG', 'PENN', 'PEP', 'PFE', 'PFG', 'PG', 'PGR', 'PH', 'PHM', 'PKG', 'PKI', 'PLD', 'PM', 'PNC', 'PNR', 
    'PNW', 'POOL', 'PPG', 'PPL', 'PRGO', 'PRU', 'PSA', 'PSX', 'PTC', 'PWR', 'PXD', 'PYPL', 'QCOM', 'QRVO', 'RCL', 'RE', 'REG', 'REGN', 'RHI', 'RJF', 'RL', 
    'RMD', 'ROK', 'ROL', 'ROP', 'ROST', 'RSG', 'RTX', 'SBAC', 'SBUX', 'SCHW', 'SIVB', 'SHW', 'SIG', 'SPG', 'SRE', 'STE', 'STT', 'STZ', 'SYF', 'SYK', 'SYY', 
    'T', 'TAP', 'TDG', 'TDY', 'TECH', 'TER', 'TFX', 'TGT', 'TJX', 'TMO', 'TMUS', 'TROW', 'TRV', 'TSCO', 'TSLA', 'TSM', 'TSS', 'TT', 'TTWO', 'TXN', 'TXT', 
    'TYL', 'UAL', 'UDR', 'UHS', 'ULTA', 'UNH', 'UNM', 'UNP', 'UPS', 'URI', 'USFD', 'V', 'VFC', 'VIAC', 'VLO', 'VMC', 'VNO', 'VRSK', 'VRSN', 'VRTX', 'VTR', 
    'VZ', 'WAB', 'WAT', 'WBA', 'WBD', 'WDC', 'WEC', 'WELL', 'WFC', 'WHR', 'WLTW', 'WM', 'WMB', 'WMT', 'WRB', 'WRK', 'WST', 'WTW', 'WY', 'WYNN', 'XEL', 'XOM', 
    'XRAY', 'XYL', 'YUM', 'ZBH', 'ZBRA', 'ZION', 'ZTS' 
]

# Historical data helper 
def get_historical_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="1y")
    return data

# Trainer / preducter  using closing prices, train-test-split, linear training, price prediction & MSE optimization
def predict_stock_prices(symbol):
    data = get_historical_data(symbol)
    data['Date'] = data.index
    data['Target'] = data['Close'].shift(-1)
    data = data.dropna()
    X = data[['Close']]
    y = data['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)    
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    next_day_prediction = model.predict(data[['Close']].tail(1))
    return next_day_prediction[0], mse

# Based on budget, determine portion of shares
def determine_stocks_to_buy(budget, threshold=0.02, top_n=40):
    stocks_to_buy = []
    stock_predictions = []
    
    for symbol in sp500_symbols:
        try:
            next_day_price, mse = predict_stock_prices(symbol)
            current_price = get_historical_data(symbol)['Close'].iloc[-1]
            increase_percentage = (next_day_price - current_price) / current_price
            stock_predictions.append((symbol, current_price, next_day_price, increase_percentage, mse))
        except Exception as e:
            print(f'Could not process data for {symbol}: {e}')
    
    sorted_stocks = sorted(stock_predictions, key=lambda x: x[3], reverse=True)
    top_stocks = sorted_stocks[:top_n]
    
    for stock in top_stocks:
        symbol, current_price, next_day_price, increase_percentage, mse = stock
        if increase_percentage >= threshold:
            quantity_to_buy = budget // current_price
            stocks_to_buy.append((symbol, quantity_to_buy, next_day_price, mse))
    
    return stocks_to_buy

budget = 100000
stocks_to_buy = determine_stocks_to_buy(budget)

if stocks_to_buy:
    print('Stocks to buy:')
    for stock in stocks_to_buy:
        symbol, quantity, predicted_price, mse = stock
        print(f'Stock: {symbol}, Quantity: {quantity}, Predicted Price: {predicted_price:.2f}, MSE: {mse:.2f}')
else:
    print('No stocks met the criteria for buying.')