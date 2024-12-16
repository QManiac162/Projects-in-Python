# Real-Time AI Stock Advisor with Ollama (Llama 3) & streamlit

 This project uses LLM-powered insights to fetch stock data every minute, analyze trends, and provide real-time, easy-to-understand explanations.

### Step 1: Fetching the Stock Data for Analysis
 The historical stock data for both Apple (AAPL) and the Dow Jones Index (DJI) has been fetched. Used yfinance to get stock data for the previous day in 1-minute intervals.

### Step 2: Processing the Real-Time Stock Updates
 Here real-time simulation of updates is done by processing one stock data point per minute. Metrics like rolling averages and momentum are added to understand market trends.

### Step 3: Analyzing the Stock Trends
 To make sense of the real-time data, we calculate moving averages, price changes, volume changes, and technical indicators like Exponential Moving Average (EMA), Bollinger Bands, and RSI (Relative Strength Index). Here’s how these indicators work:

1. Exponential Moving Average (EMA):
Puts more weight on recent prices to identify short-term trends.
2. Relative Strength Index (RSI): Measures price movement speed and oscillates between 0 and 100 to identify overbought/oversold conditions.
3. Bollinger Bands: Help assess market volatility with upper and lower bands around the moving average.

### Step 4: Getting Natural Language Insights Using Ollama
 To make this project more interactive, we integrate Llama 3 model to generate natural language insights every 5 minutes.

### Step 5: Setting up the Streamlit UI
 Basic UI with Streamlit to display stock updates and insights in real time is deployed.