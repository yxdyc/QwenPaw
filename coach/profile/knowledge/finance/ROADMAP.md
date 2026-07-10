# Finance Learning Roadmap

Target profiles: CPA-level financial analysis, Quantitative finance (Quant), Portfolio management

## Difficulty Scale

Each topic is rated 1-10. K+1 means advancing by one level from current mastery.

## Learning Path

### Tier 1: Foundations (Difficulty 1-3)

#### T1.1 Financial Statement Analysis
- **Difficulty**: 2
- **Prerequisites**: None
- **Key concepts**: Income statement, balance sheet, cash flow statement, ratio analysis (liquidity, profitability, leverage ratios)
- **Assessment criteria**: Can read and interpret all three statements; can calculate and explain key ratios
- **Resources**: Penman "Financial Statement Analysis", CFA Level I FRA readings

#### T1.2 Time Value of Money
- **Difficulty**: 2
- **Prerequisites**: Basic algebra
- **Key concepts**: PV, FV, annuities, perpetuities, NPV, IRR, discount rates
- **Assessment criteria**: Can compute PV/FV for standard instruments; can explain why money has time value
- **Resources**: Brealey/Myers "Principles of Corporate Finance" Ch. 2-4

#### T1.3 Basic Accounting Principles
- **Difficulty**: 2
- **Prerequisites**: None
- **Key concepts**: Accrual vs cash basis, revenue recognition, matching principle, depreciation methods
- **Assessment criteria**: Can explain accrual accounting; can identify accounting treatments for common transactions
- **Resources**: Kieso "Intermediate Accounting" Ch. 1-5

#### T1.4 Probability & Statistics for Finance
- **Difficulty**: 3
- **Prerequisites**: Basic calculus
- **Key concepts**: Distributions (normal, log-normal, t), expected value, variance, covariance, correlation, CLT
- **Assessment criteria**: Can compute expected returns and portfolio variance; understands distribution assumptions
- **Resources**: DeGroot & Schervish, CFA Quant Methods

### Tier 2: Intermediate (Difficulty 4-6)

#### T2.1 Portfolio Theory (Markowitz)
- **Difficulty**: 4
- **Prerequisites**: T1.2, T1.4
- **Key concepts**: Efficient frontier, mean-variance optimization, diversification, correlation effects, two-fund separation
- **Assessment criteria**: Can derive efficient frontier for 2-3 assets; can explain diversification benefits mathematically
- **Resources**: Markowitz (1952), Bodie/Kane/Marcus Ch. 7

#### T2.2 Equity Valuation Models
- **Difficulty**: 5
- **Prerequisites**: T1.1, T1.2
- **Key concepts**: DDM (Gordon growth, multi-stage), DCF, relative valuation (P/E, EV/EBITDA), residual income
- **Assessment criteria**: Can build DCF model; can compare valuation methods and explain when each applies
- **Resources**: Damodaran "Investment Valuation", Penman Ch. 4-6

#### T2.3 Fixed Income Fundamentals
- **Difficulty**: 5
- **Prerequisites**: T1.2, T1.4
- **Key concepts**: Bond pricing, YTM, duration, convexity, yield curve, credit spreads
- **Assessment criteria**: Can price a bond; can compute and interpret duration/convexity; understands yield curve shapes
- **Resources**: Fabozzi "Fixed Income", Hull Ch. 4-6

#### T2.4 Financial Econometrics
- **Difficulty**: 6
- **Prerequisites**: T1.4, linear algebra basics
- **Key concepts**: Regression (OLS, GLS), time series (AR, MA, ARIMA), stationarity, cointegration, GARCH
- **Assessment criteria**: Can run and interpret regression; understands stationarity tests; can model volatility with GARCH
- **Resources**: Tsay "Analysis of Financial Time Series"

#### T2.5 Risk Management Basics
- **Difficulty**: 5
- **Prerequisites**: T1.4, T2.1
- **Key concepts**: VaR (parametric, historical, Monte Carlo), CVaR/Expected Shortfall, stress testing, risk budgeting
- **Assessment criteria**: Can compute VaR using multiple methods; understands limitations of each approach
- **Resources**: Jorion "Value at Risk", McNeil/Frey/Embrechts

### Tier 3: Advanced (Difficulty 7-9)

#### T3.1 Derivatives Pricing
- **Difficulty**: 7
- **Prerequisites**: T2.3, T1.4, stochastic calculus basics
- **Key concepts**: Forwards, futures, options (Black-Scholes), Greeks, put-call parity, implied volatility
- **Assessment criteria**: Can price vanilla options; can explain and compute Greeks; understands BSM assumptions and limitations
- **Resources**: Hull "Options, Futures, and Other Derivatives"

#### T3.2 Quantitative Portfolio Management
- **Difficulty**: 7
- **Prerequisites**: T2.1, T2.4
- **Key concepts**: Factor models (Fama-French, APT), alpha generation, risk parity, smart beta, transaction cost modeling
- **Assessment criteria**: Can build multi-factor model; can implement risk parity; understands alpha vs factor exposure
- **Resources**: Grinold/Kahn "Active Portfolio Management", Ang "Asset Management"

#### T3.3 Stochastic Calculus for Finance
- **Difficulty**: 8
- **Prerequisites**: T1.4, real analysis basics
- **Key concepts**: Brownian motion, Ito's lemma, martingales, risk-neutral pricing, Feynman-Kac theorem
- **Assessment criteria**: Can apply Ito's lemma; can derive BSM from first principles; understands measure change
- **Resources**: Shreve "Stochastic Calculus for Finance I & II"

#### T3.4 Alternative Investments & Hedge Fund Strategies
- **Difficulty**: 7
- **Prerequisites**: T2.1, T3.1
- **Key concepts**: PE/VC, real estate, commodities, long/short equity, global macro, event-driven, fund of funds
- **Assessment criteria**: Can analyze hedge fund strategies; understands fee structures, liquidity terms, and risk profiles
- **Resources**: Lhabitant "Hedge Funds", CAIA curriculum

#### T3.5 Asset Allocation & Wealth Management
- **Difficulty**: 8
- **Prerequisites**: T2.1, T3.2
- **Key concepts**: Strategic vs tactical allocation, liability-driven investing, tax optimization, estate planning, behavioral biases
- **Assessment criteria**: Can design IPS; can model tax-efficient withdrawal; can identify and mitigate behavioral biases
- **Resources**: Swensen "Pioneering Portfolio Management", CFA Wealth Management

### Tier 4: Expert (Difficulty 9-10)

#### T4.1 Machine Learning for Finance
- **Difficulty**: 9
- **Prerequisites**: T2.4, Python/R, ML fundamentals
- **Key concepts**: Feature engineering for financial data, overfitting in low-SNR environments, cross-validation for time series, ensemble methods
- **Assessment criteria**: Can build and validate ML models for financial prediction; understands why financial ML is different from standard ML
- **Resources**: De Prado "Advances in Financial ML", Dixon et al. "ML & Data Science Blue Book"

#### T4.2 Market Microstructure
- **Difficulty**: 9
- **Prerequisites**: T3.2, stochastic processes
- **Key concepts**: Order book dynamics, bid-ask spread models, price impact, high-frequency trading, adverse selection
- **Assessment criteria**: Can model order flow; understands market impact; can analyze HFT strategies
- **Resources**: O'Hara "Market Microstructure Theory", Hasbrouck "Empirical Market Microstructure"

#### T4.3 Advanced Risk Modeling
- **Difficulty**: 9
- **Prerequisites**: T3.3, T2.5, extreme value theory
- **Key concepts**: Copulas, tail dependence, systemic risk, contagion modeling, network analysis, macroprudential regulation
- **Assessment criteria**: Can model tail risk with copulas; understands systemic risk measures (CoVaR, SRISK)
- **Resources**: McNeil/Frey/Embrechts "Quantitative Risk Management"

## Project Anchors

Connect learning to real projects using data in `.local/projects/`:
- **Portfolio Simulation**: T2.1, T3.2, T3.5 — practice asset allocation with synthetic or user-approved data
- **Company Analysis**: T2.2, T3.4 — compare valuation assumptions using public company filings
- **Risk Planning**: T3.5 — reason about generic insurance and balance-sheet scenarios without collecting personal data by default
- **Periodic Review**: Use `review-checklist.md` for structured monthly/quarterly/annual portfolio reviews

## K+1 Progression Guide

Typical progression paths:
- **CPA track**: T1.1 → T1.3 → T2.2 → T2.5 → T3.5
- **Quant track**: T1.4 → T2.4 → T3.3 → T4.1 → T4.2
- **Portfolio Manager track**: T1.2 → T2.1 → T3.2 → T3.5 → T4.1

Cross-track bridges:
- T2.1 (Portfolio Theory) connects CPA and PM tracks
- T2.4 (Econometrics) connects all three tracks
- T4.1 (ML for Finance) is the advanced convergence point
