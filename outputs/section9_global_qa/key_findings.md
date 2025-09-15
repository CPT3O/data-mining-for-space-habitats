# Key Findings (Section 9)
- Missions analyzed: 6

## Coverage
- Average coverage: 99.86%
- Median coverage: 100.0%

## Quality mix
- Original: 62.77% | Interpolated: 37.04% | Missing: 0.19%

## Seasonality
- Series: 42, median strength: 0.049

## Granger causality
- Total significant edges: 50
- By direction: {'radiation→telemetry': 7, 'telemetry→radiation': 6, 'within-domain': 37}

## Forecasting
- Median RMSE by model: {'LSTM': 0.038}
- Best-model wins: {'LSTM': 7}

## Anomalies
- Overall average Z>3 anomalies/day: 15.539

## Omics
- DEG counts (q<0.05): [{'GLDS': 'GLDS-98', 'Total_DEG_q<0.05': 67, 'Up': 26, 'Down': 41}, {'GLDS': 'GLDS-99', 'Total_DEG_q<0.05': 2809, 'Up': 1564, 'Down': 1245}, {'GLDS': 'GLDS-104', 'Total_DEG_q<0.05': 4931, 'Up': 2360, 'Down': 2571}]
