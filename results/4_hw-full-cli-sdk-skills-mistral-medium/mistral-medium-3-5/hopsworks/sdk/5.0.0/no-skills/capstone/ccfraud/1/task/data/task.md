# Capstone — credit-card fraud detection (classification)

`data/transactions.csv` is a labelled history of card transactions (`transaction_id, cc_num, datetime, amount, merchant, category, lat, long, is_fraud`). `data/score_transactions.csv` holds later transactions WITHOUT the label — you must predict each one's fraud probability.

Build the full pipeline on the platform:
1. Engineer fraud features (e.g. transaction velocity per card, geo distance from the card's usual location, amount/hour signals) into a feature group `cctxnee3558`.
2. Assemble a training dataset `cctdee3558` from it.
3. Train a fraud classifier and register it as `ccmodelee3558` WITH its evaluation metrics.
4. Score every row of `score_transactions.csv` and write the results to a feature table `ccpredee3558` (record key `transaction_id`, column `fraud_probability` in [0,1]).

Target: ROC AUC ≥ 0.749 on the held-out scoring slice (naive base-rate ≈ 0.500).
