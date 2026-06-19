# Apex Markets — Financial Analysis

**[Live Demo](https://apex-markets-florian-kovacevic.streamlit.app/)**

Advanced financial analysis application with detailed fundamental metrics, statistical tools, and multi-asset comparison.

Built as a personal project to deepen my understanding of equity analysis and quantitative metrics.
 
## Features

### Fundamental Analysis
- Complete financial profile: P/E, P/B, P/S, EV/EBITDA, PEG ratio
- Profitability metrics: ROE, ROA, margins (gross, operating, net)
- Balance sheet overview: debt ratios, free cash flow, cash position
- Dividend analysis: yield, payout ratio, ex-date tracking

### Statistical Analysis
- Risk metrics: Sharpe, Sortino, VaR (95%/99%), CVaR, downside risk
- Distribution analysis: skewness, kurtosis, best/worst days
- Correlation matrix between selected assets
- Linear regression and performance attribution

### Multi-Asset Comparison
- Side-by-side performance comparison across custom timeframes
- Relative strength analysis
- Drawdown comparison and recovery tracking

### Research Tools
- Ticker search with Yahoo Finance autocomplete
- News feed integration per asset
- Analyst consensus and price targets
- Exportable metric tables

## Tech Stack

| Layer | Tools |
|-------|-------|
| **Language** | Python |
| **Framework** | Streamlit |
| **Data** | yfinance (Yahoo Finance API) |
| **Visualization** | Plotly (interactive charts) |
| **Computation** | Pandas, NumPy, SciPy |

## Getting Started

```bash
# Clone the repository
git clone https://github.com/Florian-KOVACEVIC/apex-markets.git
cd apex-markets

# Create virtual environment and install dependencies
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt

# Launch the app
python -m streamlit run app2.py
```

The app opens at `http://localhost:8501`.

## Disclaimer

This tool is for educational and personal use only. It does not constitute financial advice. Market data is provided by Yahoo Finance with potential delays.
