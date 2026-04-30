# Integrated Technical Report: Stock Market Direction Prediction with LSTM and FinBERT Sentiment Fusion

## Source Material Note

This report is based on the two provided notebooks: `dl_v3.ipynb`, which implements the deep learning stock-direction model, and `nlp.ipynb`, which implements the FinBERT sentiment pipeline and the sentiment-technical fusion model. The assignment PDF was mentioned in the request but was not available in the workspace at the time of writing, so the report focuses on the actual notebook implementation and the results recorded in the notebook outputs.

## Abstract

This integrated project combines two complementary machine learning areas: deep learning for financial time-series modeling and natural language processing for financial sentiment analysis. The main objective is to predict short-term stock price direction by combining technical market features with sentiment extracted from financial news. The deep learning component builds a stacked LSTM classifier using historical OHLCV stock data and engineered technical indicators. The NLP component fine-tunes FinBERT on Financial PhraseBank and then uses the resulting sentiment model to score ticker-linked financial news headlines. Finally, a fusion model combines the learned LSTM representation from technical data with a scalar sentiment score, allowing the model to use both price-history patterns and market-language signals.

The deep learning notebook uses five major technology stocks: AAPL, MSFT, GOOGL, AMZN, and TSLA. Daily OHLCV data is downloaded from Yahoo Finance for the period from 2021-01-01 to 2026-01-01. Each stock initially has 1,255 trading days, and after technical indicator construction and missing-value removal, each stock has 1,222 usable rows. The target is binary: `1` if the next day's close is higher than the current day's close, and `0` otherwise. This creates a next-day direction prediction problem rather than a price-level regression problem. The model uses a 30-day lookback window, meaning that each training sample contains the previous 30 trading days of engineered features. Across all five stocks, the final sequence dataset contains 4,170 training sequences, 595 validation sequences, and 1,195 test sequences.

The LSTM model uses 18 engineered features, including daily returns, short and medium-term returns, rolling volatility, rolling mean return, RSI, MACD, volume dynamics, price-to-moving-average position, intraday range, ATR, and Bollinger Band percentage position. These features are designed to expose different dimensions of market behavior: momentum, trend, volatility, mean reversion, volume confirmation, and intraday uncertainty. The model architecture is a two-layer stacked LSTM with hidden size 128, dropout 0.35, batch normalization, and a fully connected classification head. It has 216,449 trainable parameters and is trained with `BCEWithLogitsLoss`, AdamW, gradient clipping, a warmup plus cosine learning-rate schedule, and early stopping based on validation AUC.

The standalone LSTM achieves an out-of-sample AUC-ROC of 0.5069 and accuracy of 0.5121 using the tuned threshold of 0.49. This result is weak in a classical machine learning sense because the AUC is close to 0.5, but it is still realistic for daily stock-direction prediction. Financial markets are noisy, adaptive, partially efficient, and strongly affected by exogenous events that are not fully captured by historical daily technical indicators. The class report also shows a bias toward the `Up` class: the model recalls 62 percent of upward days but only 39 percent of downward days. This is consistent with the dataset having slightly more upward days than downward days and with equity markets often having an upward drift over longer periods.

The NLP notebook fine-tunes `ProsusAI/finbert` on the `mteb/financial_phrasebank` dataset using the `sentences_allagree` subset. This subset contains 1,129 labeled financial sentences, with 692 neutral, 285 positive, and 152 negative examples. The train-validation split is stratified, producing 903 training samples and 226 validation samples. After three epochs of fine-tuning, FinBERT reaches 95.13 percent validation accuracy and 95.17 percent weighted F1. This high score is plausible because FinBERT is already domain-pretrained on financial language, and the `sentences_allagree` subset contains examples where annotators fully agree, making labels cleaner and less ambiguous than typical real-world news.

For fusion, the notebook loads ticker-matched news from `kasrah/fnspid_financial_news_stock_merged`, producing 2,954 headlines across the selected stocks. Each headline is scored using FinBERT by computing the positive probability minus the negative probability. This creates a continuous sentiment score where positive values indicate bullish language, negative values indicate bearish language, and near-zero values indicate neutral or uncertain sentiment. The fusion dataset contains 2,439 overlapping price-news samples. The fusion model freezes the LSTM backbone from the baseline technical model and trains only a new classification head that receives the 128-dimensional LSTM representation concatenated with the one-dimensional sentiment score. On the overlapping dataset, the baseline LSTM achieves 52.97 percent accuracy, while the fusion LSTM achieves 56.05 percent accuracy. The improvement is modest but meaningful in the context of financial prediction, where incremental gains can matter if they are stable, robust, and tradable after costs.

Overall, the project demonstrates a complete multimodal financial ML pipeline: market data acquisition, technical feature engineering, sequence modeling with LSTM, financial-domain transformer fine-tuning, headline-level sentiment scoring, and technical-sentiment fusion. The results are honest and educational: the NLP classifier performs strongly on clean financial sentiment labels, while the stock-direction model remains difficult and near-random on true out-of-sample prediction. This contrast is important. It shows that understanding sentiment is not the same as reliably predicting tradable price movement. The main value of the project is therefore not only the final accuracy, but also the systematic integration of deep learning and NLP methods, the careful treatment of noisy financial data, and the critical analysis of why financial forecasting remains hard.

## 1. Project Context and Objective

This project sits at the intersection of CSR311 Deep Learning and CSR322 Natural Language Processing. The deep learning side focuses on modeling stock-price time series using an LSTM, while the NLP side focuses on financial sentiment extraction using FinBERT. The integrated goal is to test whether combining historical technical signals with financial news sentiment improves next-day stock-direction prediction.

The prediction task is binary classification. For each stock and each trading day, the model predicts whether the next day's closing price will be higher than the current day's closing price. This is encoded as:

```text
target = 1 if tomorrow's close is greater than today's close
target = 0 otherwise
```

This formulation is important because the model is not trying to predict the exact price. Predicting exact stock prices is often unstable because prices are non-stationary, depend on market regime, and can change scale dramatically over time. Direction prediction is more directly connected to trading decisions: if the model predicts `Up`, one might consider a long position; if it predicts `Down`, one might avoid the position or consider a short or defensive strategy. However, direction prediction is still extremely difficult because daily price movement contains a large random component.

The project uses two information sources. The first source is structured numerical market data: open, high, low, close, and volume. From this, the notebook creates technical indicators that summarize momentum, volatility, trend, volume behavior, and price position. The second source is unstructured financial text: news headlines associated with specific tickers and dates. FinBERT converts these headlines into sentiment scores, and the fusion model uses those scores alongside technical features.

The central research question is:

```text
Can financial sentiment, when added to a technical LSTM model, improve next-day stock direction prediction?
```

The notebook results suggest that sentiment fusion improves accuracy on the overlapping news-price dataset from 52.97 percent to 56.05 percent. This should be interpreted carefully. The improvement is encouraging, but the fusion evaluation is not as strict as the chronological out-of-sample LSTM test because the fusion model is trained and evaluated on the same overlapping set in the notebook. Therefore, the improvement demonstrates potential signal, but not yet a fully validated trading edge.

## 2. Data Understanding

### 2.1 Deep Learning Stock Dataset

The deep learning notebook downloads daily OHLCV data using `yfinance`. The selected stocks are:

```text
AAPL, MSFT, GOOGL, AMZN, TSLA
```

These are large-cap technology stocks with high liquidity, frequent news coverage, and substantial trading volume. They are reasonable choices for this project because they have enough historical data for sequence modeling, they are actively followed by investors and analysts, and they are also likely to appear in financial news datasets. This matters because the final goal is not only technical prediction but sentiment fusion. A stock with sparse news coverage would be less useful for testing the NLP component.

The deep learning notebook downloads data from 2021-01-01 to 2026-01-01. Each ticker has 1,255 trading days in the raw downloaded dataset. After feature engineering, rolling-window indicators, and missing-value removal, each ticker has 1,222 rows. Missing rows are expected because indicators such as 20-day rolling volatility, 20-day moving averages, RSI, MACD, ATR, and Bollinger Bands require a warmup period. For example, a 20-day rolling statistic cannot be computed reliably on the first few rows because there are not yet 20 historical observations.

The notebook prints the following post-processing class balance:

| Ticker | Rows After Feature Engineering | Up Percentage |
|---|---:|---:|
| AAPL | 1,222 | 53.1% |
| MSFT | 1,222 | 52.1% |
| GOOGL | 1,222 | 53.8% |
| AMZN | 1,222 | 51.5% |
| TSLA | 1,222 | 51.5% |

The class distribution is close to balanced, but it is not exactly 50-50. There are slightly more upward days than downward days. This is common in equity markets because stocks often have a long-term upward drift, especially large technology stocks over multi-year windows. This small imbalance is important because a model can obtain slightly above-random accuracy by leaning toward the upward class. Therefore, accuracy alone is not enough. AUC, macro F1, precision, recall, and confusion matrices are needed to understand whether the model is genuinely discriminating between classes.

After converting rows into 30-day sequences, the combined dataset across all five stocks is:

| Split | Shape | Up Percentage |
|---|---:|---:|
| Train | `(4170, 30, 18)` | 51.99% |
| Validation | `(595, 30, 18)` | 54.96% |
| Test | `(1195, 30, 18)` | 52.80% |

The shape `(4170, 30, 18)` means there are 4,170 training examples, each example contains 30 time steps, and each time step contains 18 engineered features. The chronological split is important because stock data is time ordered. Randomly shuffling before splitting would leak future market regimes into training and produce overly optimistic results. The notebook correctly creates sequences first and then splits chronologically within each ticker.

### 2.2 NLP Financial PhraseBank Dataset

The NLP notebook uses `mteb/financial_phrasebank` with the `sentences_allagree` configuration. Financial PhraseBank is a labeled financial sentiment dataset containing sentences from financial news. Labels are:

```text
0 = negative
1 = neutral
2 = positive
```

The `sentences_allagree` subset contains only examples where annotators agreed on the label. This makes the dataset cleaner but also easier than real-world financial sentiment. In real news, many headlines are ambiguous. For example, a headline may be positive for one company and negative for another, or positive in the short term but negative in the long term. The all-agree subset filters out many such ambiguous cases.

The dataset size in the notebook is:

| Label | Count |
|---|---:|
| Neutral | 692 |
| Positive | 285 |
| Negative | 152 |
| Total | 1,129 |

The class distribution is imbalanced. Neutral examples dominate the dataset. This is realistic because most financial statements are factual rather than strongly positive or negative. However, imbalance means accuracy can be misleading. A model that overpredicts neutral could achieve decent accuracy while failing to identify rare negative examples. The notebook therefore reports precision, recall, F1, macro average, and weighted average in addition to accuracy.

The train-validation split is stratified, producing 903 training samples and 226 validation samples. Stratification is important because it preserves class proportions in both splits. Without stratification, the small negative class could become underrepresented in validation, making the evaluation unstable.

### 2.3 FNSPID News Dataset for Fusion

For real-world news scoring, the NLP notebook uses:

```text
kasrah/fnspid_financial_news_stock_merged
```

The notebook filters this dataset to the selected stocks and maps `GOOG` to `GOOGL` for consistency. It loads 2,954 ticker-matched headlines. The per-ticker coverage is:

| Ticker | Headline Count | Earliest Date | Latest Date |
|---|---:|---:|---:|
| AAPL | 446 | 2020-03-09 | 2023-12-15 |
| AMZN | 218 | 2020-04-27 | 2023-12-15 |
| GOOGL | 1,237 | 2019-01-02 | 2023-12-15 |
| MSFT | 414 | 2022-04-26 | 2023-12-15 |
| TSLA | 639 | 2019-07-01 | 2023-12-15 |

This dataset is valuable because it connects text to ticker symbols and dates. That makes it possible to align sentiment with daily stock features. However, it also introduces challenges. News coverage is uneven across companies. GOOGL has many more headlines than AMZN, and MSFT has coverage only from 2022-04-26 onward in the filtered output. This unevenness means the fusion model may learn more from heavily covered stocks and less from sparsely covered stocks.

Another challenge is timing. The notebook aligns sentiment to the same calendar date as the price sample. In real trading, the exact release time matters. A headline released after market close should not be used to predict that same day's close-to-close movement if it was not available before the trading decision. This is one of the most important data leakage risks in financial NLP. The notebook demonstrates the fusion concept, but a production-grade system would need timestamp-level alignment.

## 3. Feature Engineering: Detailed Explanation

Feature engineering is central to the deep learning notebook. The raw OHLCV data contains only open, high, low, close, and volume. These raw values are useful, but they do not directly express market concepts such as momentum, volatility, trend strength, range expansion, or relative price position. The notebook therefore constructs 18 curated features.

The feature list is:

```text
returns
rolling_std_20
rolling_mean_20
rsi
macd
macd_signal
macd_diff
volume_change
volume_ratio
price_to_ma20
high_low_range
open_close_diff
ret_5d
ret_10d
std_5
std_10
atr_14
bb_pct
```

Each feature captures a different behavioral aspect of the stock.

### 3.1 Daily Returns

Daily return measures how much the closing price changed from the previous trading day to the current trading day. This is one of the most fundamental financial features because models should generally learn from relative changes rather than raw price levels. A raw close price of 200 dollars has different meaning for AAPL than for TSLA or AMZN, but a daily return of 2 percent is comparable across stocks.

This is important because neural networks can otherwise confuse scale with behavior. If one stock has a higher nominal price than another, the model might treat it as fundamentally different even when the percentage movement pattern is similar. Returns normalize the movement and make the feature more stationary than raw prices. Financial time series are rarely fully stationary, but returns are usually closer to stationary than price levels.

For next-day direction prediction, recent returns provide short-term momentum and reversal information. If a stock has risen sharply over the last few days, it may continue due to momentum, or it may reverse due to profit-taking. The LSTM can learn such temporal patterns from sequences of returns.

### 3.2 5-Day and 10-Day Returns

The features `ret_5d` and `ret_10d` measure medium-short-term price movement over approximately one trading week and two trading weeks. These features help the model see broader movement than a single daily return.

This is useful because one-day returns are noisy. A single day may move because of random market fluctuation, sector rotation, index rebalancing, or a temporary liquidity effect. A 5-day or 10-day return smooths over some of this randomness and gives the model a clearer view of recent trend direction. If a stock has consistently moved upward over 10 days, that may indicate momentum. If it has moved too far too quickly, it may indicate overextension. The feature itself does not decide which interpretation is correct; it gives the LSTM information that can be interpreted in context with volatility, RSI, MACD, and volume.

### 3.3 Rolling Mean of Returns

The 20-day rolling mean of returns summarizes the average return over roughly one trading month. It captures the recent drift of the stock. A positive rolling mean means that the stock has had, on average, positive daily returns over the last 20 trading days. A negative rolling mean means the recent drift has been downward.

This helps the model because markets often move in regimes. A stock may be in a bullish regime, bearish regime, or sideways regime. The rolling mean gives a compact representation of that regime. It is not enough by itself because the average may hide large fluctuations, but combined with rolling standard deviation it tells the model whether the stock is moving consistently or erratically.

### 3.4 Rolling Volatility: 20-Day, 10-Day, and 5-Day Standard Deviation

The notebook computes `rolling_std_20`, `std_10`, and `std_5` using rolling standard deviations of returns. These features measure recent volatility at different horizons.

Volatility is important because the same return has different meaning in different volatility regimes. A 2 percent move in a calm stock may be a strong signal, while a 2 percent move in a highly volatile stock like TSLA may be normal noise. Volatility also affects predictability. During high-volatility periods, price movement can become more event-driven and less stable. During low-volatility periods, small trends may persist more smoothly.

Using multiple windows helps the model compare short-term volatility against medium-term volatility. For example, if 5-day volatility rises sharply above 20-day volatility, the stock may be entering a new unstable regime. If short-term volatility is low while the 20-day trend is positive, the stock may be moving upward in a controlled trend. These relationships are hard for a linear model to express manually, but an LSTM can learn them from sequences.

### 3.5 RSI

RSI, or Relative Strength Index, is a momentum oscillator. It tries to measure whether recent gains are dominating recent losses or vice versa. High RSI values generally indicate strong upward momentum or possible overbought conditions. Low RSI values generally indicate strong downward momentum or possible oversold conditions.

RSI is useful because stock movement is not only about direction but also about exhaustion. A stock that has gone up for many days may continue rising because of strong demand, or it may reverse because buyers are exhausted. RSI helps the model identify this tension. In a pure trend-following regime, high RSI may be bullish. In a mean-reverting regime, high RSI may be bearish. This is exactly why sequence modeling is useful: the model can learn how RSI behaves in combination with prior returns, volatility, and trend indicators.

### 3.6 MACD, MACD Signal, and MACD Difference

MACD is a trend and momentum indicator based on the relationship between faster and slower moving averages. The notebook includes three MACD-related features:

| Feature | Meaning |
|---|---|
| `macd` | Main MACD line, representing momentum based on moving-average separation |
| `macd_signal` | Smoothed signal line used to compare against MACD |
| `macd_diff` | Difference between MACD and signal line |

The intuition is that trend strength can be detected by comparing shorter-term and longer-term price behavior. When short-term price behavior is stronger than longer-term behavior, momentum may be increasing. When short-term behavior weakens relative to longer-term behavior, momentum may be fading.

The `macd_diff` feature is especially useful because it captures the gap between the current momentum estimate and its smoothed signal. A widening positive difference can indicate strengthening bullish momentum. A widening negative difference can indicate bearish pressure. However, like RSI, MACD is not universally predictive by itself. It becomes useful when interpreted in the surrounding sequence context.

### 3.7 Volume Change

Volume change measures how much trading volume changed relative to the previous day. Volume is important because price movements with strong volume often have more credibility than price movements with weak volume. If a stock rises on unusually high volume, it may indicate institutional participation, news reaction, or strong conviction. If it rises on low volume, the move may be weaker.

This helps the model distinguish between quiet price drift and active market repricing. A large price move with no volume confirmation may be less reliable. A large move with significant volume may indicate a real information event.

### 3.8 Volume Ratio

The volume ratio compares current volume to its 20-day average. This is different from volume change because it compares the current day to a broader recent baseline rather than only to yesterday.

This is important because volume has its own regimes. Some stocks naturally trade more during certain periods, around earnings, or during market stress. A volume ratio above 1 means the current volume is higher than recent average volume. A ratio below 1 means volume is lower than normal. The model can use this to identify abnormal attention or abnormal quietness.

### 3.9 Price-to-MA20

The `price_to_ma20` feature compares the closing price to its 20-day moving average. This measures where the current price sits relative to its recent trend baseline.

If the ratio is above 1, the stock is trading above its 20-day average. If below 1, it is trading below its 20-day average. This can represent trend strength or overextension. In a trend-following interpretation, price above the moving average is bullish because buyers are maintaining price above recent fair value. In a mean-reversion interpretation, price far above the moving average could be vulnerable to pullback. The LSTM receives the sequence and can learn which interpretation is more useful in different contexts.

### 3.10 High-Low Range

The `high_low_range` feature measures the intraday trading range relative to the close. It captures how much the stock moved within the day, regardless of where it finally closed.

This matters because the closing price alone hides intraday uncertainty. A stock that opened calmly, spiked upward, crashed downward, and closed flat had a very different day from a stock that traded flat all day. High intraday range can indicate uncertainty, disagreement among market participants, stop-loss triggering, news reaction, or volatility expansion.

For next-day prediction, range expansion can signal that the market is repricing information. It can also signal instability. When combined with ATR and rolling volatility, this feature helps the model understand whether the stock is in a calm or turbulent state.

### 3.11 Open-Close Difference

The `open_close_diff` feature measures how much the stock moved from open to close during the same day. It gives information about intraday direction.

This is useful because the daily return from previous close to current close combines overnight movement and intraday movement. The open-close feature isolates the regular-session directional move. If a stock opens low but closes strong, it may indicate intraday buying pressure. If it opens high but closes weak, it may indicate selling pressure or fading optimism. This is especially relevant when combined with volume and high-low range.

### 3.12 ATR-14

ATR, or Average True Range, measures average price range over a window. The notebook normalizes ATR by close price, which makes it comparable across stocks with different price levels.

ATR is a volatility feature, but it differs from standard deviation of returns. Standard deviation focuses on variability of returns around their mean. ATR focuses on actual trading range, including gaps. This is important in stocks because overnight gaps are common after earnings, macro news, analyst upgrades, or sector events. ATR therefore gives the model a broader view of risk and movement magnitude.

ATR is useful for stock prediction because next-day direction is often harder during high range periods. It also affects the practical meaning of a prediction. A correct prediction during high ATR may have larger trading impact than a correct prediction during low ATR.

### 3.13 Bollinger Band Percentage

The Bollinger Band percentage feature, `bb_pct`, measures where the price sits inside its Bollinger Band range. Bollinger Bands combine a moving average with volatility bands. The percentage position tells whether price is near the lower band, middle band, or upper band.

This helps the model understand relative overextension. A price near the upper band may indicate strong momentum, but it may also indicate that the stock is stretched. A price near the lower band may indicate weakness, but it may also indicate a potential bounce. The interpretation depends on regime. In a trending market, riding the upper band can be bullish. In a mean-reverting market, touching the upper band can precede a pullback. The LSTM can learn these patterns only if such features are provided over time.

## 4. Preprocessing and Sequence Construction

### 4.1 Target Construction

The target is created using the next day's closing price:

```python
df['target'] = (c.shift(-1) > c).astype(int)
```

This means that for each row, the label answers the question: "Will the next close be higher than today's close?" The final row for each stock cannot have a valid next-day label, and rolling indicators create earlier missing values. The notebook handles this by replacing infinite values with missing values and dropping incomplete rows.

This target design is simple and explainable. It is also hard. A next-day direction target has low signal-to-noise ratio because daily returns are heavily influenced by random fluctuations and unexpected events. The model is being asked to extract a weak statistical edge, not an obvious deterministic relationship.

### 4.2 Why Sequence Modeling Is Needed

A normal tabular classifier would see only one row at a time. That means it would receive today's RSI, today's return, today's volatility, and so on, but it would not directly see how those values evolved over the last month. Stock behavior is temporal. The meaning of a feature depends on its path.

For example, RSI of 70 has different meaning depending on whether RSI has been rising steadily from 40, oscillating around 70 for weeks, or suddenly jumping from 30 after a reversal. Similarly, a high volatility value has different meaning if volatility is gradually declining versus suddenly exploding. Sequence modeling allows the model to learn from trajectories rather than isolated snapshots.

The notebook uses a lookback of 30 trading days. This is approximately six trading weeks. It is long enough to include short-term trend and volatility context, but short enough to remain computationally manageable. Each sample is shaped as:

```text
(30 time steps, 18 features)
```

For a batch of 64 samples, the input shape becomes:

```text
(64, 30, 18)
```

The LSTM processes the 30 time steps in order and updates its hidden representation as it reads each day. The final hidden representation is then used for classification.

### 4.3 Chronological Splitting

The notebook uses chronological splitting with approximately 70 percent train, 10 percent validation, and 20 percent test for each stock. This is crucial for time-series modeling. In standard machine learning, random train-test splits are common. In financial forecasting, random splits are dangerous because they allow future market regimes to appear in training while earlier periods appear in testing. That creates unrealistic evaluation.

Chronological splitting better simulates the real forecasting situation. The model trains on earlier history, tunes on later validation data, and is tested on the most recent held-out period. This does not remove all risks, but it is much closer to a real deployment scenario than random splitting.

### 4.4 Scaling and RobustScaler

The code uses `RobustScaler()` to scale features. There is a notebook labeling inconsistency: the cell heading, print message, and checkpoint metadata mention `StandardScaler`, but the actual code constructs and uses `RobustScaler`. For technical correctness, the implemented scaler is RobustScaler.

This matters because financial features often contain outliers. Stock returns, volume changes, ATR, high-low ranges, and volatility can spike during earnings announcements, market crashes, macroeconomic shocks, or company-specific news. StandardScaler uses the mean and standard deviation. Both are sensitive to outliers. A few extreme observations can shift the mean or inflate the standard deviation, causing most normal observations to be compressed into a narrow range.

RobustScaler instead uses more robust statistics based on the median and interquartile range. The intuition is that it centers the data around a typical value and scales it according to the spread of the middle portion of the data, rather than allowing extreme tails to dominate. This is useful in financial data because extreme events are real and should not be deleted automatically, but they also should not distort the scale of every other observation.

The scaler is fitted only on the training sequences and then applied to validation and test sequences. This is the correct pattern because fitting the scaler on all data would leak future distribution information into the training process.

## 5. LSTM Model Architecture

### 5.1 Why LSTM Is Used

The deep learning notebook uses an LSTM because the input is sequential and the prediction depends on patterns across time. LSTM stands for Long Short-Term Memory. It is a recurrent neural network architecture designed to handle temporal dependencies better than a basic RNN.

This is important because financial sequences have both short-term and medium-term effects. A sudden return spike, a volatility increase, or a momentum shift may affect the next few days. A basic feedforward model would not naturally remember the order of events. An LSTM reads the sequence step by step and maintains a hidden state. That hidden state acts like a learned summary of the past 30 days.

The LSTM is especially suitable when the dataset is not enormous. Transformers can be powerful, but they usually require more data and stronger regularization to avoid overfitting. In this project, the deep learning dataset has 5,960 total sequences across the five stocks after splitting. That is small by modern deep learning standards. A two-layer LSTM is a reasonable architecture because it is expressive enough to model temporal dependencies but not as data-hungry as a large transformer.

### 5.2 Why Not a Linear Model

A linear model would assume that each feature contributes in a mostly additive and fixed way to the prediction. That is too restrictive for technical trading signals. The meaning of RSI depends on volatility. The meaning of volume depends on price movement. The meaning of a positive return depends on whether it follows a long downtrend or an ongoing uptrend.

Financial indicators interact in nonlinear ways. For example, high RSI plus rising volume plus low volatility may mean something different from high RSI plus falling volume plus high volatility. A linear model can include handcrafted interaction terms, but the number of possible interactions grows quickly. An LSTM can learn nonlinear temporal interactions directly from sequences.

### 5.3 Why Not a CNN

A one-dimensional CNN can model local temporal patterns and is often useful for time series. However, CNNs are most natural when local patterns are the main signal. They use filters that slide across time and detect short motifs. Stock direction may depend not only on local motifs but also on how market state evolves across the entire 30-day window.

An LSTM explicitly maintains a sequential hidden state. This makes it natural for modeling progression: momentum building, volatility fading, trend weakening, or reversal forming. A CNN could still work, but the LSTM is more directly aligned with the idea of reading the market history in order.

### 5.4 Why Not a Transformer

Transformers are powerful for sequence modeling because attention can compare every time step with every other time step. However, they usually require larger datasets and careful regularization. The stock dataset here is modest. A transformer could easily overfit, especially because financial signals are weak and noisy.

Transformers also have more architectural choices: positional encoding, attention heads, layer depth, embedding size, pooling strategy, and regularization. For an integrated academic project, the LSTM provides a strong, interpretable baseline. It is easier to explain, easier to train, and appropriate for a 30-step sequence length.

### 5.5 Architecture Breakdown

The model is defined as:

```text
Input -> 2-layer LSTM -> final time-step hidden state -> BatchNorm -> Dropout -> FC head -> logit
```

The exact configuration is:

| Component | Configuration |
|---|---|
| Input size | 18 features |
| Lookback | 30 days |
| LSTM hidden size | 128 |
| LSTM layers | 2 |
| LSTM dropout | 0.35 |
| Batch normalization | `BatchNorm1d(128)` |
| Dropout after LSTM | 0.35 |
| Head layer 1 | `Linear(128, 64)` |
| Activation | ReLU |
| Head dropout | 0.35 |
| Output layer | `Linear(64, 1)` |
| Parameters | 216,449 |

The input shape during training is:

```text
(batch_size, sequence_length, feature_count)
```

With the notebook settings:

```text
(64, 30, 18)
```

The LSTM produces an output for each time step. The model selects the output from the final time step:

```python
h = out[:, -1, :]
```

This final hidden vector is treated as the model's learned summary of the previous 30 trading days. It has 128 dimensions. Batch normalization stabilizes this representation, dropout regularizes it, and the fully connected head maps it to a single logit.

### 5.6 Hidden States

The hidden state is the LSTM's internal memory. As the LSTM reads each day in the 30-day sequence, it updates this memory based on the current input and the previous memory. The final hidden representation should contain information about recent trend, volatility, momentum, volume behavior, and their evolution.

This helps because the model does not need to manually store every time step in the final classifier. Instead, the LSTM compresses the sequence into a learned representation. If the model finds that the last five days are more important than the first five days, it can learn that. If it finds that a volatility spike 20 days ago still matters, it can preserve that information in the hidden state.

### 5.7 Dropout

Dropout randomly disables a fraction of activations during training. The notebook uses dropout of 0.35 in the LSTM and in the classification head. This is important because the dataset is small relative to the complexity of the model. Without regularization, the model could memorize noise patterns in the training set.

In financial ML, overfitting is especially dangerous because historical patterns can disappear quickly. A model may learn patterns specific to one market regime, such as the post-pandemic technology rally, and fail in another regime. Dropout forces the model to avoid relying too heavily on any one path through the network.

### 5.8 Batch Normalization

Batch normalization is applied to the final LSTM hidden vector. It stabilizes the distribution of hidden activations before they enter the classification head. This can make optimization smoother and reduce sensitivity to scale changes.

In this model, batch normalization is useful because the LSTM hidden representation is learned from multiple stocks and multiple market regimes. Normalizing the representation can help the classification head receive more stable inputs.

## 6. Training Strategy

### 6.1 BCEWithLogitsLoss

The model outputs a raw logit, not a probability. A logit can be any real number. Positive logits indicate higher probability of the `Up` class; negative logits indicate lower probability. To convert a logit to probability, the sigmoid function is applied.

The notebook uses `BCEWithLogitsLoss`, which combines sigmoid activation and binary cross-entropy loss in a numerically stable way. This is better than manually applying sigmoid and then using normal binary cross-entropy because very large positive or negative logits can cause numerical instability. `BCEWithLogitsLoss` handles the computation more safely internally.

This is important for stable training, especially when using neural networks that may temporarily produce extreme logits during early epochs.

### 6.2 Class Imbalance Handling

The notebook computes:

```python
pos_weight = num_neg / num_pos
```

The printed value is:

```text
pos_weight = 0.9234
```

Because there are slightly more positive examples than negative examples in the training set, the positive class receives a weight below 1. This means positive errors are weighted slightly less than negative errors. The goal is to prevent the model from simply leaning too much toward the majority class.

The imbalance is not severe, but even small imbalance matters when the target is difficult and accuracy is near 50 percent. If the model always predicts `Up`, it could achieve around 52 percent accuracy on this dataset. Class weighting helps push the model to pay attention to both directions.

### 6.3 AdamW Optimizer

The notebook uses AdamW with learning rate `8e-4` and weight decay `1e-3`. AdamW is a variant of Adam that decouples weight decay from the adaptive gradient update. This usually gives better regularization behavior than traditional Adam with L2 penalty.

AdamW is a good choice here because the model is nonlinear, the dataset is noisy, and different parameters may need different effective learning rates. Adaptive optimizers are often easier to train than plain SGD in such settings. Weight decay discourages overly large weights, which helps reduce overfitting.

### 6.4 Warmup and Cosine Learning Rate Schedule

The notebook uses a learning-rate schedule with warmup for the first 8 epochs and then cosine decay. Warmup gradually increases the learning rate instead of starting immediately at the maximum value. This helps prevent unstable early updates when the model weights are still uncalibrated.

After warmup, cosine decay gradually reduces the learning rate. The intuition is that early training should allow larger steps to discover useful regions of the parameter space, while later training should use smaller steps to refine the solution. In noisy financial data, this is helpful because aggressive late updates can cause the model to chase noise in the validation set.

### 6.5 Gradient Clipping

The training loop uses:

```python
nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

Gradient clipping is especially important in recurrent networks such as LSTMs. During backpropagation through time, gradients can sometimes become very large. This is called exploding gradients. If gradients explode, the optimizer may take huge parameter updates, causing training instability or divergence.

Clipping limits the gradient norm to a maximum value. This does not eliminate learning; it prevents extreme updates. In practice, this makes LSTM training more stable, especially on noisy sequences.

### 6.6 Early Stopping

The notebook tracks validation AUC and saves the best model state. Training stops after validation AUC fails to improve for 25 epochs. The best validation AUC is 0.5490 at epoch 9, and early stopping occurs at epoch 34.

Early stopping is important because training loss can keep improving even while validation performance worsens. That means the model is memorizing training patterns that do not generalize. In financial data, this is common because random historical noise can look like a pattern. Early stopping helps preserve the model state that performed best on unseen validation data.

## 7. Threshold Tuning

The default binary classification threshold is 0.5. This means the model predicts `Up` when the predicted probability is greater than 50 percent. However, 0.5 is not always optimal.

The reason is that neural network probabilities are not guaranteed to be perfectly calibrated. A model's output of 0.51 does not necessarily mean the true probability is exactly 51 percent. Also, the best threshold depends on the metric. If the goal is accuracy, one threshold may be best. If the goal is macro F1, another threshold may be better. If the goal is trading profit, the best threshold might be even higher because trades should only be taken when confidence is strong enough to overcome costs.

The notebook tunes the threshold on the validation set using macro F1. It searches thresholds from 0.30 to 0.69 and finds:

```text
Optimal threshold: 0.49
Validation macro-F1: 0.5635
Default 0.50 macro-F1: 0.5526
```

The improvement from 0.50 to 0.49 means the model benefits from being slightly more willing to predict the `Up` class. This is consistent with the dataset having more upward days. The threshold shift is small, but in a near-balanced and noisy problem, small threshold changes can affect recall and precision.

On the test set:

```text
Accuracy at threshold 0.50 = 0.5071
Accuracy at threshold 0.49 = 0.5121
```

This shows that threshold tuning gives a small out-of-sample improvement. However, the improvement is modest and should not be overinterpreted.

## 8. LSTM Performance Analysis

### 8.1 Test Results

The final standalone LSTM test results are:

| Metric | Value |
|---|---:|
| AUC-ROC | 0.5069 |
| Accuracy at threshold 0.50 | 0.5071 |
| Accuracy at threshold 0.49 | 0.5121 |
| Optimal threshold | 0.49 |
| Validation macro-F1 at optimal threshold | 0.5635 |

The classification report at threshold 0.49 is:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Down | 0.48 | 0.39 | 0.43 | 564 |
| Up | 0.53 | 0.62 | 0.57 | 631 |
| Accuracy |  |  | 0.51 | 1,195 |
| Macro avg | 0.51 | 0.51 | 0.50 | 1,195 |
| Weighted avg | 0.51 | 0.51 | 0.51 | 1,195 |

### 8.2 Why AUC Is Near 0.5

AUC measures the model's ability to rank positive examples above negative examples across all thresholds. An AUC of 0.5 means random ranking. The model's AUC of 0.5069 is only slightly above random. This is one of the most important findings in the project.

The AUC is near 0.5 because next-day stock direction is extremely difficult to predict from daily technical indicators alone. Markets are noisy and adaptive. If a simple pattern in RSI, MACD, returns, or volatility reliably predicted tomorrow's movement, traders would exploit it, and the edge would likely shrink. This is related to the efficient market hypothesis, though real markets are not perfectly efficient.

Another reason is that many drivers of next-day movement are missing from the feature set. Earnings announcements, macroeconomic data releases, interest-rate expectations, geopolitical shocks, analyst upgrades, liquidity flows, options positioning, and sector rotation can all move stocks. The LSTM only sees daily technical features. It cannot know that an earnings report is coming tomorrow unless that information is somehow reflected in price and volume.

The target itself is also noisy. A stock can have a technically bullish setup and still fall the next day because the broader market sells off. Conversely, a technically weak stock can rise due to unexpected good news. Therefore, even a well-designed model may struggle to achieve high AUC.

### 8.3 What 51 Percent Accuracy Means in Finance

In ordinary classification tasks, 51 percent accuracy may look poor. In finance, the interpretation is more nuanced. A small directional edge can be valuable if it is stable, statistically significant, and tradable after transaction costs and risk constraints. For example, a strategy that is correct 51 percent of the time could be profitable if average winning trades are larger than average losing trades, or if the model helps avoid large downside periods.

However, 51 percent accuracy by itself is not enough. It may be within random variation. It may disappear on a different time period. It may not survive transaction costs, slippage, taxes, or execution delay. The notebook's backtest gives a first look at strategy behavior, but it does not model transaction costs. Therefore, the result should be presented honestly: the LSTM has learned a weak signal, but not a robust standalone trading system.

### 8.4 Bias Toward the Up Class

The classification report shows that the model has higher recall for the `Up` class:

```text
Down recall = 0.39
Up recall = 0.62
```

Recall means: among all true examples of a class, how many did the model correctly identify? The model identifies 62 percent of true upward days but only 39 percent of true downward days. This indicates a bias toward predicting `Up`.

There are several reasons for this. First, the dataset has more upward examples than downward examples. Second, equities often have upward drift over long horizons. Third, the threshold was tuned to 0.49, which slightly increases the number of `Up` predictions. Fourth, technical indicators may capture continuation patterns more easily than downturn patterns, because downturns are often caused by sudden news or market-wide shocks.

This bias is not automatically bad. If the model is used as a long-only filter, predicting `Up` more often may be acceptable. But if the goal is balanced direction prediction, the weak `Down` recall is a limitation.

### 8.5 Precision vs Recall Tradeoff

Precision answers: when the model predicts a class, how often is it correct? Recall answers: among all true examples of a class, how many does the model catch?

For the `Up` class:

```text
Precision = 0.53
Recall = 0.62
```

This means the model catches many upward days, but its `Up` predictions are only slightly more correct than random. For the `Down` class:

```text
Precision = 0.48
Recall = 0.39
```

This means the model is weaker at identifying downward days. In a trading context, this could mean the model may stay long during some days that turn negative. That matters because avoiding losses can be as important as capturing gains.

## 9. Backtesting Interpretation

The notebook includes a simple backtest where the model takes exposure when the predicted probability exceeds the threshold. The results by stock are:

| Stock | LSTM Return | Buy-and-Hold Return | LSTM Sharpe | Buy-and-Hold Sharpe |
|---|---:|---:|---:|---:|
| AAPL | 15.87% | 18.74% | 0.523 | 0.563 |
| MSFT | 2.61% | 13.56% | 0.001 | 0.465 |
| GOOGL | 59.07% | 60.32% | 1.766 | 1.536 |
| AMZN | -1.18% | 2.16% | -0.062 | 0.093 |
| TSLA | 22.87% | 5.44% | 0.579 | 0.323 |

The backtest is mixed. The LSTM outperforms buy-and-hold on TSLA in return and Sharpe, and has a higher Sharpe than buy-and-hold on GOOGL despite slightly lower return. However, it underperforms on AAPL, MSFT, and AMZN. This shows that the model is not consistently superior across stocks.

This is important because a model that works only on one or two tickers may be capturing ticker-specific noise rather than a generalizable pattern. A stronger evaluation would test on unseen tickers, different market regimes, and include transaction costs.

## 10. NLP Pipeline: FinBERT

### 10.1 What FinBERT Is

FinBERT is a BERT-based language model adapted for financial text. Standard BERT is trained on general language, such as books and Wikipedia. It learns general grammar, syntax, and semantic relationships. Financial language, however, has domain-specific meanings. Words like "beat," "miss," "guidance," "downgrade," "margin pressure," "restructuring," and "liquidity" carry specialized sentiment implications.

FinBERT is useful because it has been exposed to financial language and is therefore better suited for financial sentiment classification than a general-purpose model. This is important because sentiment in finance is not the same as sentiment in movie reviews or social media. A phrase like "lower costs" is positive, while "lower revenue" is negative. The model must understand the financial object being described.

### 10.2 Why Domain-Specific BERT Is Needed

Domain-specific language models are needed because meaning depends on context. In general English, "liability" may simply mean responsibility. In finance, liabilities are balance sheet obligations. "Volatile" may be negative in a risk report but expected in an options context. "Loss narrowed" is positive even though the word "loss" appears negative.

A general sentiment model might misclassify such examples because it relies on ordinary emotional polarity. FinBERT is designed to understand financial polarity. This helps it distinguish between factual neutral statements, genuinely positive business developments, and negative financial signals.

### 10.3 Fine-Tuning on Financial PhraseBank

The notebook loads `ProsusAI/finbert` and fine-tunes it for 3-class sequence classification. Fine-tuning means the pretrained model's parameters are updated using labeled task-specific examples. The model already knows a lot about financial language, but fine-tuning adapts it specifically to the label scheme: negative, neutral, positive.

During fine-tuning, the tokenizer converts each sentence into token IDs, attention masks, and padded sequences up to a maximum length of 128. The classification head outputs three logits, one for each sentiment class. The model is trained for three epochs with batch size 16 for training and 64 for evaluation. The training uses weight decay and warmup steps.

The recorded validation performance improves across epochs:

| Epoch | Training Loss | Validation Loss | Accuracy | Weighted F1 |
|---:|---:|---:|---:|---:|
| 1 | 1.975909 | 0.412015 | 0.845133 | 0.816784 |
| 2 | 0.325196 | 0.163687 | 0.942478 | 0.943063 |
| 3 | 0.099998 | 0.135730 | 0.951327 | 0.951669 |

The final validation classification report is:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Negative | 0.90 | 0.90 | 0.90 | 30 |
| Neutral | 0.99 | 0.96 | 0.97 | 139 |
| Positive | 0.90 | 0.95 | 0.92 | 57 |
| Accuracy |  |  | 0.95 | 226 |
| Macro avg | 0.93 | 0.94 | 0.93 | 226 |
| Weighted avg | 0.95 | 0.95 | 0.95 | 226 |

### 10.4 Why FinBERT Accuracy Is High

The FinBERT accuracy is high for several reasons. First, the base model is already financial-domain pretrained, so it starts with useful knowledge. Second, Financial PhraseBank is directly aligned with the task of financial sentiment classification. Third, the notebook uses the `sentences_allagree` subset, which contains cleaner labels. Fourth, many sentences in Financial PhraseBank contain explicit sentiment cues such as "rose," "increased," "fell," "decreased," "profit," or "loss."

This high accuracy should not be confused with perfect real-world understanding. Real news headlines are shorter, noisier, more ambiguous, and sometimes require external context. For example, "Company cuts costs" may be positive if it improves margins, but negative if it signals distress. The high validation accuracy shows strong performance on a clean benchmark, not guaranteed perfect sentiment extraction in live markets.

### 10.5 POS-Tagging Analysis

The notebook includes a POS-tagging analysis of positive and negative sentences. It prints frequent adjectives and verbs in positive and negative examples. For positive headlines, verbs such as "rose," "increased," "grew," and "won" appear. For negative headlines, verbs such as "decreased," "fell," and "slipped" appear.

This analysis is useful because it provides interpretability. It shows that the dataset contains finance-relevant lexical cues. However, POS word counts are shallow compared with FinBERT. They do not understand context deeply. For example, the word "increased" is positive if profits increased but negative if costs increased. FinBERT can use context to interpret the object of the verb.

## 11. Sentiment Scoring

### 11.1 Positive Minus Negative Score

After scoring headlines with FinBERT, the notebook computes:

```text
SentimentScore = P(positive) - P(negative)
```

This creates a continuous score between approximately -1 and +1. A strongly positive headline will have high positive probability and low negative probability, producing a score near +1. A strongly negative headline will produce a score near -1. Neutral or uncertain headlines will produce scores near 0.

This is useful because the fusion model needs a numerical input, not a discrete label. A continuous score preserves confidence. For example, a headline with 55 percent positive probability is not as strongly bullish as one with 98 percent positive probability. The score captures this difference.

### 11.2 Why Neutral Is Ignored in the Score

Neutral is not directly included in the formula. This does not mean neutral is useless. Neutral influences the score indirectly because probabilities sum across classes. If a headline is highly neutral, both positive and negative probabilities are usually low, so the score becomes close to zero.

Ignoring neutral in the difference score is reasonable because the fusion model mainly needs directional sentiment. Positive sentiment may support upward movement, negative sentiment may support downward movement, and neutral sentiment should not strongly push either direction. A near-zero score naturally represents that.

### 11.3 Daily Aggregation

The notebook averages headline sentiment by ticker and date:

```python
daily_sentiment = news_df.groupby(['Ticker', 'Date'])['SentimentScore'].mean()
```

This produces one sentiment score per stock per day. Averaging is simple and interpretable. It assumes that multiple headlines on the same day combine into an overall daily sentiment signal. If one headline is positive and another is negative, the average may move toward neutral, representing mixed news.

However, averaging also loses information. It does not account for headline importance, source credibility, article timing, repeated duplicate headlines, or whether one headline is much more market-moving than another. A future system could use weighted aggregation or attention over headlines.

## 12. Fusion Model

### 12.1 Why Combining Sentiment and Technical Data Makes Sense

Technical data reflects what market participants have done: prices, volume, ranges, volatility, and trend. Sentiment data reflects what market participants are reading and reacting to: news, announcements, expectations, and narratives. These two sources can be complementary.

This is important because price history may not fully reveal the cause of movement. A stock may show rising volatility before earnings, but the headline sentiment after an announcement may explain whether the market reaction should be positive or negative. Similarly, sentiment may identify information that has not yet been fully absorbed into price, especially if the news is released outside regular trading hours or if investors react gradually.

Markets are influenced by expectations. Positive news can increase expected future cash flows, improve investor confidence, or attract buying pressure. Negative news can reduce expectations, increase perceived risk, or trigger selling. Sentiment is therefore not just emotional; in finance, it can be a proxy for information flow.

### 12.2 Fusion Dataset Construction

The fusion pipeline downloads price data from 2020-01-01 to 2024-01-01 for the same five stocks. Each ticker has 1,006 price rows before feature engineering. The same 18 technical features are created. The notebook then aligns each 30-day technical sequence with a FinBERT sentiment score for the current ticker and date.

Only dates with both price features and news sentiment are included. This produces:

```text
Overlapping samples: 2,439
```

The per-ticker overlap is:

| Ticker | Overlapping Samples |
|---|---:|
| GOOGL | 927 |
| TSLA | 452 |
| AAPL | 428 |
| MSFT | 414 |
| AMZN | 218 |

This overlap-based construction is logical because fusion requires both modalities. However, it introduces selection bias. The model is trained only on days with news. Days with no news are excluded. News days may be more volatile or event-driven than ordinary trading days, so fusion performance on this subset may not represent all trading days.

### 12.3 Fusion Architecture

The fusion model uses the pretrained LSTM backbone from the technical model. It freezes:

```text
LSTM layers
BatchNorm layer
```

It then replaces the classification head with a new head:

```text
Concatenate 128-dimensional LSTM hidden vector with 1 sentiment score
Input size = 129
Linear(129, 32)
ReLU
Dropout(0.2)
Linear(32, 1)
```

The forward pass works as follows. First, the technical sequence goes through the frozen LSTM. The final time-step hidden vector is extracted and normalized. This vector represents the technical market state. Second, the scalar sentiment score is appended to this vector. Third, the new head learns how to combine technical state and sentiment into a final prediction logit.

### 12.4 Why the LSTM Backbone Is Frozen

Freezing the LSTM backbone is a transfer learning strategy. The baseline LSTM has already learned a representation of technical patterns. The fusion dataset has only 2,439 overlapping samples, which is smaller than the full technical dataset. If the entire LSTM were retrained on this smaller set, it could overfit quickly.

By freezing the LSTM, the model preserves the learned technical representation and trains only the smaller fusion head. This reduces the number of trainable parameters and makes the fusion step more stable. It also isolates the effect of sentiment: the model is not completely relearning technical features; it is learning how sentiment should modify the decision boundary on top of existing technical information.

### 12.5 Fusion Training

The fusion model trains only the head parameters using AdamW with learning rate 0.005, weight decay 0.001, cosine annealing for 80 epochs, and `BCEWithLogitsLoss` with class weighting. The loss decreases from around 0.6452 at epoch 10 to around 0.6377 at epoch 80, with some fluctuation.

Because the fusion model is trained on all overlapping samples and then evaluated on those same samples in the notebook, the reported fusion performance should be treated as an in-sample or demonstration result rather than a strict out-of-sample result. This is a critical limitation for viva discussion.

### 12.6 Fusion Results

On the 2,439 overlapping news-price samples, the notebook reports:

| Model | Optimal Threshold | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| Baseline LSTM on overlap | 0.50 | 0.5297 | 0.52 | 0.53 |
| Fusion LSTM | 0.50 | 0.5605 | 0.56 | 0.56 |

The baseline overlap classification report is:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Down | 0.50 | 0.46 | 0.48 | 1,137 |
| Up | 0.56 | 0.59 | 0.57 | 1,302 |
| Accuracy |  |  | 0.53 | 2,439 |
| Macro avg | 0.53 | 0.53 | 0.52 | 2,439 |
| Weighted avg | 0.53 | 0.53 | 0.53 | 2,439 |

The fusion classification report is:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Down | 0.52 | 0.62 | 0.57 | 1,137 |
| Up | 0.60 | 0.51 | 0.55 | 1,302 |
| Accuracy |  |  | 0.56 | 2,439 |
| Macro avg | 0.56 | 0.56 | 0.56 | 2,439 |
| Weighted avg | 0.57 | 0.56 | 0.56 | 2,439 |

The fusion model improves accuracy by about 3.08 percentage points on the overlapping dataset. In financial ML, this is not trivial. Moving from 52.97 percent to 56.05 percent can be meaningful if it generalizes. However, because the evaluation is not strictly out-of-sample, the improvement should be described as promising rather than conclusive.

An interesting change is that fusion improves `Down` recall from 0.46 to 0.62 while reducing `Up` recall from 0.59 to 0.51. This suggests sentiment may be helping the model identify negative or risk-off days more effectively. That makes intuitive sense because negative news can trigger sudden downward movement that may not be visible from technical indicators alone.

## 13. Mandatory Results Table

| Component | Dataset / Evaluation Set | Main Method | Key Metrics | Interpretation |
|---|---|---|---|---|
| FinBERT sentiment classifier | Financial PhraseBank `sentences_allagree`, 226 validation samples | Fine-tuned `ProsusAI/finbert` for 3-class sentiment | Accuracy 0.9513, weighted F1 0.9517, macro F1 0.93 | Strong sentiment classifier on clean financial benchmark; high performance is expected due to domain pretraining and all-agree labels. |
| LSTM baseline | Out-of-sample chronological stock test set, 1,195 sequences | 2-layer LSTM with 18 technical features and 30-day lookback | AUC 0.5069, accuracy 0.5121 at threshold 0.49 | Weak but realistic stock-direction performance; model is only slightly above random and biased toward `Up`. |
| Baseline LSTM on fusion overlap | 2,439 dates with both news and price data | Loaded technical LSTM evaluated on overlap subset | Accuracy 0.5297, macro F1 0.52 | Slightly stronger than full test result, but measured on news-overlap subset rather than the original chronological test. |
| Fusion LSTM | 2,439 overlapping news-price samples | Frozen LSTM backbone plus sentiment-augmented head | Accuracy 0.5605, macro F1 0.56, weighted F1 0.56 | Sentiment improves performance on overlap samples, especially `Down` recall, but needs stricter out-of-sample validation. |

## 14. Critical Thinking: Why the Model Is Weak

The stock prediction model is weak because the task is genuinely hard, not because the notebook is poorly designed. Daily stock direction is close to random from the perspective of publicly available historical daily features. Many price movements are caused by information that is not present in technical indicators. Even when information is present, markets may incorporate it quickly.

The model also uses only daily data. Intraday dynamics are missing. For example, a stock may react strongly in the first 30 minutes after news and then mean-revert by close. Daily OHLCV compresses all of that into one row. Important microstructure information, such as bid-ask spread, order-book imbalance, intraday momentum, and volume profile, is absent.

Macroeconomic context is also missing. Interest rates, inflation expectations, Federal Reserve announcements, bond yields, unemployment data, oil prices, currency movement, and index-level risk sentiment can all affect technology stocks. A model that only sees individual stock technical indicators may misinterpret movements caused by macro shocks.

Event timing is another missing piece. Earnings dates, product announcements, regulatory events, lawsuits, analyst calls, and management guidance can dominate next-day returns. The sentiment model sees some headlines, but the fusion pipeline does not fully model whether the headline arrived before or after the trading decision. Without precise timestamps, the model may use information that would not have been available in a real trading scenario.

Finally, the dataset is small for deep learning. Although there are thousands of sequences, this is still limited compared with the complexity of financial markets. Deep learning models can easily overfit when the signal is weak and the number of market regimes is limited.

## 15. Limitations for Viva Discussion

### 15.1 Data Leakage Risks

The main leakage risk in the technical model is mostly controlled by chronological splitting and fitting the scaler only on training data. However, the notebook has a labeling inconsistency where it says StandardScaler while using RobustScaler. This is not leakage, but it is important to explain clearly.

The fusion pipeline has a more serious leakage risk. It aligns sentiment by date but does not verify headline release time relative to the trading decision. If a headline was published after market close, using it for that day's prediction would be unrealistic. A production system must use timestamps and only include news available before the prediction time.

Another fusion limitation is that the scaler is fitted on all fusion sequences before model training. Also, the fusion model is trained and evaluated on the same overlapping samples. This means the reported fusion improvement is not a strict out-of-sample result.

### 15.2 Overfitting

Overfitting is a major risk because the LSTM has 216,449 parameters and the dataset is relatively small. The notebook uses dropout, weight decay, gradient clipping, and early stopping to reduce overfitting, but the validation AUC peaks early at epoch 9 and then declines, which suggests overfitting begins quickly.

The fusion model reduces overfitting risk by freezing the LSTM backbone, but it still evaluates on the same data used for training. Therefore, its reported improvement may partly reflect fitting the overlap dataset rather than generalizing to future unseen samples.

### 15.3 No Transaction Cost Modeling

The backtest does not model transaction costs, slippage, spreads, taxes, or market impact. This is important because small predictive edges can disappear after costs. If the model changes position frequently, even low transaction costs can eliminate profitability.

A realistic trading evaluation should include assumptions such as brokerage cost, bid-ask spread, execution delay, and position sizing. It should also compare against simple baselines such as buy-and-hold, moving-average strategies, and always-long strategies.

### 15.4 Weak Generalization

The model is trained on five large-cap technology stocks. These stocks are correlated and influenced by similar sector factors. A model that works on them may not generalize to financial stocks, energy stocks, small-cap stocks, international equities, or different market regimes.

The test period is also limited. A robust financial model should be evaluated across bull markets, bear markets, sideways markets, high-rate regimes, low-rate regimes, crisis periods, and calm periods. Without this, generalization remains uncertain.

### 15.5 Sentiment Ambiguity

FinBERT produces strong sentiment scores, but headlines can be ambiguous. A headline may mention multiple companies. It may describe market-wide news rather than firm-specific news. It may be positive for revenue but negative for margins. A single scalar sentiment score cannot capture all of this nuance.

Also, sentiment does not always translate directly into price movement. If positive news is already expected, the stock may fall after the announcement. This is the classic "buy the rumor, sell the news" effect. Therefore, sentiment should be combined with expectations and surprise, not just polarity.

## 16. Future Improvements

### 16.1 Transformer-Based Time Series Models

A future version could use transformer-based time-series architectures such as Temporal Fusion Transformer, Informer-style models, or patch-based time-series transformers. Attention could help the model identify which past days are most relevant for the current prediction. For example, the model might attend strongly to an earnings gap, a volatility spike, or a previous support-break event.

However, transformers should be introduced carefully. They require more data and stronger validation. A transformer may overfit this dataset unless trained across many more stocks and longer histories.

### 16.2 Attention-Based Fusion

The current fusion model appends one daily sentiment score to the LSTM hidden vector. A stronger approach would use attention over multiple headlines. Instead of averaging sentiment, the model could learn which headlines matter most. For example, an earnings headline should receive more weight than a generic market recap.

Attention fusion could combine text embeddings directly with technical embeddings. Rather than reducing each headline to one sentiment number, the model could use FinBERT's hidden representation, which contains richer semantic information. This would allow the fusion model to learn not only whether news is positive or negative, but what type of news it is.

### 16.3 Real-Time Streaming Pipeline

A production system should process market data and news in real time. It should ingest live prices, live headlines, timestamps, and possibly social media or analyst reports. The model should only use information available before the prediction time.

This would make the evaluation more realistic. For example, the system could generate predictions before market open using overnight news, or during market hours using intraday updates. Real-time alignment would reduce leakage and make the project closer to deployable trading infrastructure.

### 16.4 Macroeconomic and Market Features

The model could be improved by adding index returns, sector ETF returns, VIX, treasury yields, interest-rate expectations, inflation data, and macro announcement flags. These features help distinguish stock-specific movement from market-wide movement.

For technology stocks, NASDAQ index behavior, semiconductor index behavior, and treasury yields may be especially important. Rising yields can pressure growth stocks even when company-specific technical indicators look strong.

### 16.5 Event-Aware Modeling

Future work should include earnings calendars, analyst rating changes, product launches, lawsuits, regulatory announcements, and merger news. Event-aware modeling could help the system understand when normal technical patterns are likely to break.

For example, predicting the day after earnings using the same logic as a normal trading day may be inappropriate. Earnings days have different volatility and sentiment dynamics.

### 16.6 Reinforcement Learning for Trading

The current model predicts direction. A trading system also needs to decide position size, entry timing, exit timing, stop-loss rules, and risk limits. Reinforcement learning could model trading as a sequential decision problem where the agent learns actions such as buy, hold, sell, or reduce exposure.

However, reinforcement learning in finance is risky and easy to overfit. It should only be attempted after building a strong backtesting environment with transaction costs, walk-forward validation, and realistic execution assumptions.

### 16.7 Better Evaluation

Future evaluation should use walk-forward validation. In walk-forward validation, the model trains on an initial period, tests on the next period, then rolls forward and repeats. This better simulates real trading and gives multiple out-of-sample periods.

The fusion model should also be evaluated with a chronological split. The sentiment scaler, model training, threshold tuning, and final testing should all be separated by time. This would make the fusion improvement more credible.

## 17. Strong Conclusion

This project successfully demonstrates an end-to-end stock market prediction pipeline that combines deep learning and financial NLP. The technical branch uses an LSTM to learn from 30-day sequences of engineered stock indicators, while the NLP branch fine-tunes FinBERT to classify financial sentiment and convert real-world headlines into daily sentiment scores. The fusion branch combines both sources by appending sentiment to the LSTM's learned technical representation.

The results are realistic and educational. FinBERT performs very strongly on Financial PhraseBank, reaching about 95 percent validation accuracy. This shows that transformer-based NLP models can understand financial sentiment well when trained on clean domain-specific labels. In contrast, the standalone LSTM achieves only 51.2 percent out-of-sample accuracy and an AUC near 0.5. This shows that next-day stock direction prediction is much harder than sentiment classification because markets are noisy, adaptive, and influenced by many missing variables.

The fusion model improves accuracy from 52.97 percent to 56.05 percent on overlapping news-price samples. This suggests that sentiment contains useful information beyond technical indicators. The improvement is small, but in financial machine learning small improvements can matter if they are stable and tradable. At the same time, the fusion result must be interpreted carefully because the notebook does not perform a strict chronological out-of-sample split for the fusion stage.

The most important takeaway is that the project is not a claim of a finished trading system. It is a strong prototype of a multimodal financial prediction framework. It shows how to engineer technical features, train an LSTM, fine-tune FinBERT, score real news, align sentiment with market data, and evaluate whether sentiment improves prediction. For a viva or interview, the strongest answer is honest: the model is technically sound as a learning pipeline, the sentiment fusion result is promising, but robust financial deployment would require stricter validation, timestamp-safe news alignment, transaction cost modeling, broader datasets, and walk-forward testing.

## Resume Bullet Point

Built an integrated stock-direction prediction system combining a PyTorch stacked LSTM over 18 engineered technical indicators with a fine-tuned FinBERT financial sentiment pipeline, improving overlap-sample direction accuracy from 52.97 percent to 56.05 percent through sentiment-technical fusion while conducting rigorous performance, leakage, and limitation analysis.

## Two-Line Project Summary for Presentation

This project predicts next-day stock direction by combining technical market patterns learned through a stacked LSTM with financial news sentiment extracted using fine-tuned FinBERT.

The standalone LSTM achieved 51.2 percent out-of-sample accuracy, while sentiment fusion improved overlap-sample accuracy to 56.1 percent, showing that financial news sentiment can add incremental predictive signal when carefully aligned with market data.
