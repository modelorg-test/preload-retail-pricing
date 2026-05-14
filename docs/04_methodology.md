# Methodology & Assumptions

A gradient boosted regression model (LightGBM) trained on historical price-demand pairs. The elasticity coefficient is estimated via a log-log demand model:

$$\ln(Q) = \alpha + \beta \cdot \ln(P) + \gamma \cdot X + \epsilon$$

where Q is quantity demanded, P is price, and X represents exogenous features (seasonality, competitor pricing).
