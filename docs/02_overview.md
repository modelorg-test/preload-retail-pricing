# Overview & Strategy

Regression model estimating price elasticity of demand to maximise gross margin across the ecommerce product catalogue. The model recommends optimal base prices and markdown schedules.

## Inputs

- `price_delta`: Percentage change from current price (float)
- `demand_index`: Normalised demand forecast (float)
- `competitor_price`: Median competitor price for equivalent SKU (float)
- `inventory_days`: Days of remaining inventory at current velocity (int)

## Outputs

- `optimal_price`: Recommended price point (float, AED)
- `elasticity_coefficient`: Estimated demand sensitivity (float)
- `margin_impact`: Projected gross margin change (float, %)
