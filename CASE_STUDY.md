# Case study: Pulse — Customer Intelligence Dashboard

## The problem

Ecommerce stores accumulate order history but have no systematic way to answer "which customers matter right now, and why." Generic approaches either treat every customer identically (a blanket 30-day follow-up email) or borrow a subscription-business framework ("predict churn") that doesn't fit physical-product purchasing, where customers don't cancel — they just quietly stop returning, on their own unpredictable schedule.

## The approach

Pulse combines two ML techniques deliberately chosen to fit the actual shape of the problem, not to showcase complexity:

1. **Unsupervised segmentation (K-Means)** groups customers into behavioral tiers — VIP, regular, at-risk, one-time — using recency, frequency, monetary value, and interpurchase timing. No pre-existing labels are needed or assumed.
2. **Supervised repurchase-window prediction (RandomForestRegressor)** estimates, in days, when each customer is likely to buy again — trained only on customers with enough order history to have a real "next purchase" to learn from, then applied to the full customer base, including one-time buyers.

A business-rule layer translates the model output into one plain-English recommended action and an urgency level, so the deliverable is a decision, not a probability.

## Why synthetic data

No client data exists for this demo yet, and using synthetic data is standard, expected practice for demonstrating methodology before a real engagement begins — this is stated explicitly rather than implied. The synthetic generator (`src/synthetic_data.py`) simulates each customer's orders around a hidden "true repurchase cadence," so the resulting segments and predictions have to be recovered from noisy, realistic behavioral signal — the same challenge a real dataset presents — rather than being trivially separable by construction.

## Results (on the bundled 2,000-customer demo sample)

- 4 behavioral segments recovered with a silhouette score of 0.49
- Repurchase-window predictions accurate to a mean absolute error of ~8.4 days — materially tighter than a generic 30-day rule for most segments
- ~41% of customers had enough order history for a genuine supervised label; the remaining ~59% (mostly one-time buyers) are still scored via generalization, not guesswork

## What this demonstrates for client work

The same pipeline (`src/pipeline.py`) is designed to run unmodified against a real Shopify or CRM export — swapping `DATA_MODE=local` and pointing the loader at a real file requires no changes to the cleaning, feature-engineering, segmentation, scoring, or recommendation logic. That reusability is the actual product: not this specific demo, but the pipeline architecture behind it.
