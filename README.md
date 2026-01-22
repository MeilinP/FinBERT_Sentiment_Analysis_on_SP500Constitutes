# News Sentiment Analysis for Stock Returns

Does financial news sentiment actually predict stock returns? I wanted to find out.

## What This Is

This project tests whether sentiment extracted from financial news headlines has any predictive power for stock returns. The idea is simple: use FinBERT (a BERT model fine-tuned on financial text) to score ~330K news headlines from 2009-2020, then see if stocks with positive news actually outperform stocks with negative news.

The focus here isn't on building a profitable trading system – it's about isolating sentiment as a signal and measuring its statistical significance.

## The Approach

1. **Data**: Pulled news headlines covering S&P 500 stocks over 11 years
2. **Sentiment Scoring**: Used FinBERT to classify each headline as positive/negative/neutral
3. **Signal Construction**: Aggregated daily sentiment scores for each stock
4. **Testing**: Built long-short portfolios based on sentiment deciles and measured forward returns

The key tests:
- Information Coefficient (IC) analysis to measure rank correlation between sentiment and returns
- Fama-French factor regression to see if sentiment adds explanatory power beyond standard factors
- Multiple time horizons (1-day, 2-day, weekly) to check signal persistence

## Key Findings

Sentiment does show up as a statistically significant factor. Weekly rebalancing with a minimum news volume filter (5+ articles) produced the cleanest signal with a Sharpe of 1.26.

The IC analysis and Fama-French regressions confirmed that sentiment carries incremental information beyond market, size, and value factors.

## Tech Stack

- **FinBERT**: Pre-trained sentiment model from HuggingFace
- **Data Processing**: pandas, numpy
- **Modeling**: PyTorch for transformer inference
- **Environment**: Google Colab with T4 GPU

## Notes

This was a learning project to understand NLP in finance and factor research methodology. The code includes checkpointing for the sentiment scoring (because processing 330K headlines takes a while) and some basic transaction cost modeling.

Feel free to poke around – feedback welcome!
