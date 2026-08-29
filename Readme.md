# UNDERWRITING FORECASTING ANALYSIS
Stage 1 + Stage 2A + Stage 2B

Data validation, behavioural analysis, traditional time-series forecasting and neural forecasting

| Transaction Data | Validation | Monthly Series | Diagnostics |
| :--- | :--- | :--- | :--- |
| **ARIMA / SARIMA / ETS** | **RNN / LSTM / GRU** | **Error Metrics** | **Final Ranking** |

Mentor Review Report
Prepared from the implemented notebooks, verified outputs and submitted plot evidence.

Scope note: ECM/VECM is intentionally excluded. The report covers only the models retained in the implemented forecasting comparison.

## 1. Executive Summary
This report documents the complete underwriting forecasting workflow implemented from transaction-level data through model comparison. The forecasting target is the monthly credit amount. The workflow validates and enriches transaction data, aggregates it into a monthly time series, examines the statistical behaviour of that series, and then uses those observations to guide model selection.

The forecasting stage contains two modelling paths. Stage 2A uses traditional time-series models: ARIMA, SARIMA and ETS using Holt's double exponential smoothing. Stage 2B uses recurrent neural models: Simple RNN, LSTM and GRU.

All retained models are evaluated against the same held-out June 2026 actual credit value of ₹59,000 using MAE, RMSE and MAPE. The observed ranking places LSTM first at 0.0146% MAPE, followed by ETS at 0.0336%.

The result is a prototype experiment, not a production-level generalisation claim. Only six monthly observations are available, with five used for training and one for testing. The neural path therefore has only two training sequences. This is enough to validate the implementation, but not enough to establish robust long-run model superiority.

## 2. Project Objective and Scope
The forecasting component is designed to turn historical account behaviour into a reproducible estimate of next-month credit activity. The goal is not only to produce a forecast, but to create a traceable path from raw transactions to a model-backed underwriting signal.

**The implementation follows these principles:**
• Use transaction-level data as the source of truth.
• Validate the transaction structure before modelling.
• Aggregate raw events into a consistent monthly time series.
• Use statistical diagnostics to guide traditional model selection.
• Use a common held-out observation and common error metrics for model comparison.

The prototype covers six months of prepared transaction data and forecasts the June 2026 monthly credit amount.

## 3. Complete End-to-End Pathway
Each stage produces an output that becomes the input to the next stage. Traditional models consume the ordered monthly series directly, while neural models consume scaled three-step sequences.

| | | | |
| :--- | :--- | :--- | :--- |
| **Raw Transactions** | **Data Validation** | **Share-Market Flag** | **Monthly Aggregation** |
| **Monthly Credit** | **Trend / Moving Average** | **ACF / Stationarity** | **Stage 2A** |
| **ARIMA / SARIMA / ETS** | **Forecast** | **MAE / RMSE / MAPE** | **Stage 2A Comparison** |
| **Monthly Credit** | **Scaling** | **3-Month Windows** | **RNN / LSTM / GRU** |
| **Neural Forecast** | **MAE / RMSE / MAPE** | **Neural Comparison** | **Final Ranking** |

Important hand-off: Stage 1 does not directly become a neural-network tensor. Stage 1 produces the cleaned monthly series and statistical understanding. Stage 2A uses the series directly. Stage 2B transforms the same series into sequences suitable for recurrent networks.

## 4. Stage 1 — Data Preparation and Validation
The starting transaction structure contains four core fields: date, debit, credit and balance. These represent individual account movements. Before forecasting, the records are checked for structural consistency and converted into a regular monthly representation.

### 4.1 Core fields

| Field | Role in the workflow |
| :--- | :--- |
| Date | Defines transaction timing and enables monthly grouping. |
| Debit | Represents outgoing transaction value. |
| Credit | Represents incoming transaction value and becomes the primary forecasting target after aggregation. |
| Balance | Provides account-level consistency context and supports validation of transaction behaviour. |

### 4.2 Share-market behavioural flag
A binary share-market flag was introduced for selected debit transactions. A value of 1 indicates that the debit is treated as related to share-market activity, while 0 indicates that it is not. This is a behavioural feature introduced under the mentor's assumption; it is not the forecast target.

**Why it matters**
• Captures a class of outgoing financial behaviour relevant to underwriting.
• Separates selected investment-related activity from general spending.
• Can be aggregated into monthly share-market debit amount and share-market debit ratio.
• Enriches the behavioural representation without replacing monthly credit as the forecasting target.

## 4. Stage 1 — Monthly Aggregation
Transaction records are grouped by month. Credit values are summed to obtain total monthly credit. The same aggregation produces total debit and share-market debit measures.

| Month | Total Credit | Total Debit | Share-Market Debit | Share-Market Ratio |
| :--- | :--- | :--- | :--- | :--- |
| 2026-01 | ₹55,000 | ₹41,984 | ₹12,480 | 29.73% |
| 2026-02 | ₹54,500 | ₹38,286 | ₹12,415 | 32.43% |
| 2026-03 | ₹57,000 | ₹41,967 | ₹12,729 | 30.33% |
| 2026-04 | ₹56,000 | ₹41,930 | ₹12,638 | 30.14% |
| 2026-05 | ₹58,500 | ₹48,966 | ₹12,820 | 26.18% |
| 2026-06 | ₹59,000 | ₹39,111 | ₹12,922 | 33.04% |

### Validation evidence
• Dates are treated as temporal keys and converted into monthly periods.
• Credit and debit amounts are treated numerically before aggregation.
• Monthly totals are checked for consistency before forecasting.
• The share-market flag is binary and distinguishes the selected debit transactions.
• The final forecasting target is a six-point monthly credit series.

The resulting monthly credit series is: ₹55,000, ₹54,500, ₹57,000, ₹56,000, ₹58,500 and ₹59,000.

## 5. Stage 1 — Behavioural and Statistical Analysis
The monthly series is analysed before forecasting because the modelling families make different assumptions. Trend, smoothing, distribution, autocorrelation and stationarity provide the reasoning bridge into Stage 2.

### 5.1 Key findings

| Diagnostic | Observed result | Implication |
| :--- | :--- | :--- |
| Trend | Overall upward direction | Supports testing a trend-capable model such as Holt's double exponential smoothing. |
| 3-month moving average | Rises from ₹55,500 to ₹57,833.33 | Confirms the smoothed upward direction. |
| Distribution | W=0.9335, p=0.6075 | No strong evidence against normality, but sample is only six points. |
| ACF lag 1 | 0.551 | Positive nearby-lag dependence. |
| ACF lag 2 | 0.860 | Strong short-lag relationship; motivated further temporal analysis. |
| ADF, original | stat0.4248, p=0.9824 | No evidence against a unit-root style non-stationarity interpretation. |
| ADF, first difference | stat7.7068, p1.29×10⁻¹¹ | Differencing strongly changes the stationarity result. |

The diagnostic results are descriptive because six observations are an extremely small sample.

### 5.2 Monthly Credit Trend
The fitted straight-line slope is approximately ₹885.71 per month. The series is not monotonic: February is below January and April is below March, followed by strong increases in May and June.


### 5.3 Three-Month Moving Average
The moving average values are ₹55,500, ₹55,833.33, ₹57,166.67 and ₹57,833.33 for March through June. The smoothing supports considering a trend-aware model.


### 5.4 Distribution of Monthly Credit
The six observations lie between ₹54,500 and ₹59,000. Shapiro-Wilk gives W=0.9335 with p=0.6075. This is descriptive only because the sample is very small.


### 5.5 Autocorrelation
The observed autocorrelation is approximately 0.551 at lag 1 and 0.860 at lag 2. Positive nearby-lag relationships support testing autoregressive models.


### 5.6 First Differencing
The original ADF result is approximately 0.4248 with p=0.9824, while the first-differenced series gives approximately 7.7068 with p1.29×10⁻¹¹. This is the statistical motivation for differencing in ARIMA-family modelling.


## 6. Stage 1 Stage 2 Input Mapping
Different model families use different aspects of the Stage 1 output. The same monthly credit target is retained throughout, while preprocessing changes according to model requirements.

| Stage 1 output | Used by | Purpose |
| :--- | :--- | :--- |
| Monthly total credit | All models | Primary time-series target. |
| Observed upward trend | ETS | Supports level + trend smoothing. |
| Autocorrelation | ARIMA / SARIMA | Supports autoregressive modelling. |
| Non-stationary level behaviour | ARIMA / SARIMA | Motivates differencing. |
| Seasonality check | SARIMA | Allows a seasonal extension to be tested. |
| Scaled monthly values | RNN / LSTM / GRU | Numerically stable neural input. |
| Three-month windows | RNN / LSTM / GRU | Converts the series into supervised sequences. |

Traditional models receive the ordered monthly series. Neural models receive the same series after scaling and conversion into three-step sequences.

## 7. Stage 2A — Traditional Forecasting
Traditional forecasting provides interpretable statistical baselines. The retained models are ARIMA, SARIMA and ETS using Holt's double exponential smoothing.

| Model | Primary input | What it emphasises | Why included |
| :--- | :--- | :--- | :--- |
| ARIMA(1,1,1) | Monthly credit | Autocorrelation + differencing | Baseline non-seasonal time-series model. |
| SARIMA(1,1,1)(1,1,1,2) | Monthly credit | Autocorrelation + differencing + seasonal structure | Tests a seasonal extension. |
| ETS (Holt's Double) | Monthly credit | Level + trend | Directly represents the upward direction observed in Stage 1. |

The implemented notebook outputs are used as the source of the numerical results reported below.

### 7.1 ARIMA(1,1,1)
ARIMA stands for Autoregressive Integrated Moving Average. The implemented configuration is ARIMA(1,1,1): one autoregressive term, one order of differencing and one moving-average term. The model addresses the non-stationary component through differencing and then models the transformed series using past values and past error behaviour.

**ARIMA(1,1,1) — June Forecast**
Verified result: forecast ₹56,900.91; actual ₹59,000; MAE ₹2,099.09; RMSE ₹2,099.09; MAPE 3.5578%.


### 7.2 SARIMA(1,1,1)(1,1,1,2)
SARIMA extends ARIMA with seasonal autoregressive, differencing and moving-average terms. The implemented model is SARIMA(1,1,1)(1,1,1,2). The seasonal extension is treated as a comparative experiment rather than proof of a true seasonal cycle because six observations are not enough to establish seasonality reliably.

**SARIMA — June Forecast**
Verified result: forecast ₹57,500; actual ₹59,000; MAE ₹1,500; RMSE ₹1,500; MAPE 2.5424%.


### 7.3 ETS — Holt's Double Exponential Smoothing
ETS refers to the Error, Trend and Seasonal framework of exponential smoothing. The implemented model uses Holt's double exponential smoothing, which represents level and trend and gives more weight to recent observations.

**ETS (Holt's Double) — June Forecast**
Verified result: forecast ₹59,019.84; actual ₹59,000; MAE ₹19.84; RMSE ₹19.84; MAPE 0.0336%.


### 7.4 Traditional Model Comparison

| Model | Forecast | Actual | MAE | RMSE | MAPE |
| :--- | :--- | :--- | :--- | :--- | :--- |
| ETS (Holt's Double) | ₹59,019.84 | ₹59,000.00 | ₹19.84 | ₹19.84 | 0.0336% |
| SARIMA(1,1,1)(1,1,1,2) | ₹57,500.00 | ₹59,000.00 | ₹1,500.00 | ₹1,500.00 | 2.5424% |
| ARIMA(1,1,1) | ₹56,900.91 | ₹59,000.00 | ₹2,099.09 | ₹2,099.09 | 3.5578% |

ETS is the strongest traditional model in this prototype. Its very small error is consistent with the upward direction and the short, trend-dominated series.

**Traditional Forecasting — MAPE Comparison**
Lower MAPE is better. ETS is substantially closer to the held-out June actual than ARIMA or SARIMA in this run.


## 8. Forecast Error Metrics
The same evaluation metrics are used after both traditional and neural models generate forecasts. This creates a common basis for comparison.

| Metric | Meaning | Interpretation |
| :--- | :--- | :--- |
| MAE | Mean Absolute Error | Average absolute forecast error in rupees. |
| RMSE | Root Mean Squared Error | Penalises larger errors more strongly. |
| MAPE | Mean Absolute Percentage Error | Average absolute percentage error; lower is better when actual values are non-zero. |

MAPE is used as the primary ranking metric because it gives a percentage-based comparison. MAE and RMSE are retained to show absolute and squared-error perspectives.

## 9. Stage 2B — Neural Forecasting
Neural forecasting is a second modelling family rather than a replacement for traditional models. The purpose is to test whether recurrent networks can learn the sequential relationship between recent monthly credit values and the next month's value.

### 9.1 Common preparation
• Scale the monthly credit values before neural training.
• Create supervised sequences using a three-month input window.
• Reshape the data into the recurrent-network tensor format (samples, timesteps, features).
• Use the same held-out June value and the same error metrics for all three neural models.

With five training months and a window size of three, there are two training sequences. The third window [March, April, May] is the June prediction input. This is a fundamental data-size constraint and should be stated explicitly in any mentor review.

### 9.2 Tensor shapes

| Tensor | Shape | Meaning |
| :--- | :--- | :--- |
| X_train | (2, 3, 1) | Two training sequences, three time steps, one feature. |
| y_train | (2, 1) | Two next-month targets. |
| X_test | (1, 3, 1) | One held-out three-month sequence for June. |

**Neural Forecasting Input Pathway**
The neural path uses scaling, sequence creation and reshaping before the recurrent layer. These steps are common to RNN, LSTM and GRU.


### 9.3 Simple RNN
The Simple Recurrent Neural Network processes the sequence one time step at a time and carries a hidden state forward. A Dense output layer converts the recurrent representation into one numerical forecast.

Training configuration in the implementation: recurrent layer with a small number of units, Adam optimiser, MSE loss, 200 epochs and batch size 1. The loss curve is monitored to confirm training behaviour.

**Simple RNN — Training Loss**
The loss drops rapidly in the early epochs and then flattens, showing convergence on the tiny training set. Convergence here does not imply generalisation.


**Simple RNN — June Forecast**
Verified result: forecast ₹57,026.42; actual ₹59,000; MAE ₹1,973.58; RMSE ₹1,973.58; MAPE 3.3450%.


### 9.4 LSTM
LSTM stands for Long Short-Term Memory. It is a gated recurrent architecture that controls what information is retained, updated and exposed across sequence steps. The same three-month input representation is used as in the Simple RNN; the difference is the internal memory mechanism.

**LSTM — June Forecast**
The standalone LSTM screenshot was not part of the final submitted image set. The figure is therefore a clean reconstruction of the verified result: forecast ₹59,008.61; actual ₹59,000; MAE ₹8.61; RMSE ₹8.61; MAPE 0.0146%.


### 9.5 GRU
GRU stands for Gated Recurrent Unit. It uses a simpler gated structure than LSTM. The update gate controls how much old information is retained versus updated, while the reset gate controls how much previous information is considered when forming the new state.

The implemented GRU layer contains 912 parameters and the Dense output layer contains 17 parameters, for 929 trainable parameters.

**GRU — June Forecast**
The standalone GRU screenshot was not part of the final submitted image set. The figure is therefore a clean reconstruction of the verified result: forecast ₹57,619.36; actual ₹59,000; MAE ₹1,380.64; RMSE ₹1,380.64; MAPE 2.3401%.


### 9.6 Neural Model Comparison

| Model | Forecast | Actual | MAE | RMSE | MAPE |
| :--- | :--- | :--- | :--- | :--- | :--- |
| LSTM | ₹59,008.61 | ₹59,000.00 | ₹8.61 | ₹8.61 | 0.0146% |
| GRU | ₹57,619.36 | ₹59,000.00 | ₹1,380.64 | ₹1,380.64 | 2.3401% |
| Simple RNN | ₹57,026.42 | ₹59,000.00 | ₹1,973.58 | ₹1,973.58 | 3.3450% |

LSTM has the lowest observed error. GRU performs better than Simple RNN in this run, but the six-month dataset is too small for a general architectural conclusion.

**Neural Forecasting — MAPE Comparison**
LSTM is the best neural model in this experiment, with a MAPE of 0.0146%.


## 10. Traditional vs Neural Comparison
The two modelling families can be compared because they use the same held-out June value and the same error definitions.

| Rank | Model | Family | MAPE |
| :--- | :--- | :--- | :--- |
| 1 | LSTM | Neural | 0.0146% |
| 2 | ETS (Holt's Double) | Traditional | 0.0336% |
| 3 | GRU | Neural | 2.3401% |
| 4 | SARIMA(1,1,1)(1,1,1,2) | Traditional | 2.5424% |
| 5 | Simple RNN | Neural | 3.3450% |
| 6 | ARIMA(1,1,1) | Traditional | 3.5578% |

The observed ranking is led by LSTM at 0.0146% MAPE, followed by ETS at 0.0336%. LSTM's forecast differs from the actual June value by ₹8.61, while ETS differs by ₹19.84.

**Final Traditional vs Neural Comparison**
The final comparison shows that LSTM produced the closest June forecast in this particular prototype, while ETS is the strongest traditional baseline.


## 11. Model Selection Summary

| Model | Stands out for | Main limitation in this prototype |
| :--- | :--- | :--- |
| ARIMA | Autocorrelation + differencing | Sensitive to very small samples and statistical specification. |
| SARIMA | Seasonal extension | Six observations are insufficient to establish seasonality strongly. |
| ETS | Level + trend | Less expressive for complex nonlinear relationships. |
| Simple RNN | Basic sequence learning | Very little data and possible difficulty retaining information. |
| LSTM | Controlled memory | More complex and data-hungry. |
| GRU | Efficient gated recurrence | May not match LSTM under every dataset or configuration. |

The modelling progression is deliberate: establish statistical baselines, inspect error behaviour, then test recurrent neural alternatives under the same held-out evaluation.

## 12. Model Input and Processing Summary

| Model | Input | Processing | Output |
| :--- | :--- | :--- | :--- |
| ARIMA | Monthly credit series | AR + differencing + MA | Next-month credit |
| SARIMA | Monthly credit series | ARIMA + seasonal terms | Next-month credit |
| ETS | Monthly credit series | Level + trend smoothing | Next-month credit |
| Simple RNN | 3-month sequence | Hidden-state recurrence | Next-month credit |
| LSTM | 3-month sequence | Gated memory | Next-month credit |
| GRU | 3-month sequence | Update + reset gates | Next-month credit |

The forecast-generation mechanism changes across models, but the evaluation stage is common: predicted value versus actual value, followed by MAE, RMSE and MAPE.

## 13. Implementation Evidence Record
The submitted notebook screenshots provide direct evidence for the executed diagnostics, traditional forecasts, neural training and model comparisons.

| Evidence | What it confirms |
| :--- | :--- |
| Monthly credit trend plot | Monthly series and upward direction were visualised. |
| Three-month moving average | Smoothing calculation was executed. |
| Distribution plot | Monthly credit distribution was visualised. |
| ARIMA forecast plot | ARIMA(1,1,1) was executed and visualised. |
| SARIMA forecast plot | SARIMA was executed and visualised. |
| ETS forecast plot | Holt's double exponential smoothing was executed and visualised. |
| Traditional MAPE plot | Traditional model errors were compared. |
| Simple RNN loss plot | Neural training loss was monitored. |
| Simple RNN forecast plot | RNN forecast was compared with actual June. |
| Neural MAPE plot | RNN, LSTM and GRU errors were compared. |
| Final comparison plot | All retained models were ranked using MAPE. |

Evidence note: the LSTM and GRU standalone forecast figures in this report are clean reconstructions of verified notebook results because those two standalone screenshots were not included in the final submitted image set. This is stated explicitly rather than presenting reconstructed graphics as original screenshots.

## 14. Prototype Limitations and Interpretation
• Only six monthly observations are available.
• Only five months are used for fitting and one month is held out for testing.
• The neural path has only two training sequences with a three-month window.
• A single held-out month cannot provide a robust long-run estimate of forecasting performance.
• Neural results can be sensitive to initialization and optimisation settings, especially with tiny datasets.
• Seasonality is difficult to establish from six observations, so the SARIMA seasonal extension should be interpreted as a comparison experiment.
• The share-market flag is currently an assumed behavioural classification and should be strengthened with verified labels or robust transaction-description rules in production.
• The current forecasting target is monthly total credit. The additional behavioural features are available for future extensions but are not used as multivariate predictors in this comparison.

The strongest supported conclusion is that LSTM produced the lowest observed June forecast error in this prototype, while ETS produced the strongest traditional-model result. Both should be re-evaluated on a larger historical dataset before a production choice is made.

## 15. Research Foundation and Related Work
Box, G. E. P. and Jenkins, G. M. (1970). Time Series Analysis: Forecasting and Control.
Foundational work for Box–Jenkins time-series modelling and ARIMA-family model identification, estimation and checking.
Reference: Google Books

Dickey, D. A. and Fuller, W. A. (1979). Distribution of the Estimators for Autoregressive Time Series with a Unit Root.
Foundation for unit-root testing and the statistical motivation behind stationarity analysis.
Reference: Journal of the American Statistical Association / DOI 10.1080/01621459.1979.10482531

Holt, C. C. Work on forecasting trends using exponentially weighted moving averages.
Foundation for exponential-smoothing approaches and trend-aware forecasting.
Reference: International Journal of Forecasting / DOI 10.1016/j.ijforecast.2003.09.015

Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8), 1735–1780.
Original LSTM paper describing the gated memory mechanism designed to address long-term dependency problems in recurrent learning.
Reference: DOI 10.1162/neco.1997.9.8.1735

Cho, K. et al. (2014). Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation.
Important early work associated with the gated recurrent unit architecture used as the GRU comparison family.
Reference: ACL Anthology, D14-1179

Hyndman, R. J. and Koehler, A. B. (2006). Another look at measures of forecast accuracy. International Journal of Forecasting, 22(4), 679–688.
Research foundation for comparing forecast-error measures and understanding their limitations.
Reference: DOI 10.1016/j.ijforecast.2006.03.001

These references provide methodological foundations for the model families and evaluation measures. They do not turn the six-month prototype into a statistically conclusive experiment.

## 16. Final Conclusion
The implemented underwriting forecasting pipeline connects transaction-level preparation with statistical diagnostics, traditional forecasting and neural forecasting. The workflow is modular: transaction validation produces a monthly representation; Stage 1 explains its behaviour; Stage 2A establishes statistical baselines; Stage 2B tests recurrent neural alternatives; and the final comparison evaluates all retained models using the same held-out observation and error metrics.

The observed final ranking is: LSTM 0.0146% MAPE, ETS 0.0336%, GRU 2.3401%, SARIMA 2.5424%, Simple RNN 3.3450%, and ARIMA 3.5578%.

The result demonstrates why both modelling families are useful. Traditional models provide transparent statistical baselines and can perform extremely well on short, trend-dominated series. Neural models provide a flexible sequence-learning alternative, but their advantage becomes more meaningful when substantially more historical observations and richer behavioural features are available.

The next production-oriented evolution should focus on expanding historical data, validating the share-market classification, adding justified behavioural predictors, using rolling or walk-forward evaluation, and repeating the comparison across multiple forecast horizons rather than relying on one held-out month.

END OF REPORT
