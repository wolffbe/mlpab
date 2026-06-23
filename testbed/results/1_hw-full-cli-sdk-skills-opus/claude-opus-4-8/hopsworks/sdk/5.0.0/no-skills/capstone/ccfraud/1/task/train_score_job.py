"""Runs ON THE PLATFORM as a Hopsworks python job: builds training dataset,
trains + registers classifier with metrics, scores transactions to a feature table."""
import os
import joblib
import numpy as np
import pandas as pd
import hopsworks
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

CATS = ['cash_advance','travel','online','electronics','restaurant',
        'health','fuel','grocery','clothing','entertainment']
FEATS = (['amount','log_amount','hour','dow','card_mean_amount','card_std_amount',
          'amount_zscore','dist_from_home','dist_from_prev','speed',
          'time_since_prev_sec','txns_1h','txns_24h'] + ['cat_'+c for c in CATS])


def main():
    proj = hopsworks.login()
    fs = proj.get_feature_store()
    fv = fs.get_feature_view('cctdfe5424', version=1)

    # ---- training dataset (deliverable cctdfe5424) ----
    X_train, X_test, y_train, y_test = fv.train_test_split(test_size=0.2)
    print('train', X_train.shape, 'test', X_test.shape)
    Xtr = X_train[FEATS].astype(float).fillna(0.0)
    Xte = X_test[FEATS].astype(float).fillna(0.0)
    ytr = np.asarray(y_train).ravel().astype(int)
    yte = np.asarray(y_test).ravel().astype(int)

    clf = RandomForestClassifier(n_estimators=300, max_depth=12,
                                 class_weight='balanced', n_jobs=-1,
                                 random_state=42)
    clf.fit(Xtr, ytr)
    p_te = clf.predict_proba(Xte)[:, 1]
    auc = float(roc_auc_score(yte, p_te))
    ap = float(average_precision_score(yte, p_te))
    f1 = float(f1_score(yte, (p_te >= 0.5).astype(int)))
    metrics = {'roc_auc': auc, 'average_precision': ap, 'f1': f1}
    print('METRICS', metrics)

    # ---- register model ----
    mr = proj.get_model_registry()
    model_dir = 'ccmodelfe5424_dir'
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(model_dir, 'model.pkl'))
    with open(os.path.join(model_dir, 'features.txt'), 'w') as fh:
        fh.write('\n'.join(FEATS))
    model = mr.python.create_model(
        name='ccmodelfe5424',
        metrics=metrics,
        description='Credit-card fraud RandomForest classifier')
    model.save(model_dir)
    print('registered model ccmodelfe5424', metrics)

    # ---- score every row of the scoring set ----
    sfg = fs.get_feature_group('ccscorefe5424', version=1)
    sdf = sfg.read()
    print('score rows', len(sdf))
    Xs = sdf[FEATS].astype(float).fillna(0.0)
    proba = clf.predict_proba(Xs)[:, 1].astype(float)
    proba = np.clip(proba, 0.0, 1.0)
    pred = pd.DataFrame({
        'transaction_id': sdf['transaction_id'].values,
        'fraud_probability': proba,
    })
    print('pred sample', pred.head().to_dict('records'),
          'min/max', pred.fraud_probability.min(), pred.fraud_probability.max())

    predfg = fs.get_or_create_feature_group(
        name='ccpredfe5424', version=1,
        description='Fraud probability predictions for scoring transactions',
        primary_key=['transaction_id'],
        online_enabled=True)
    predfg.insert(pred, write_options={'wait_for_job': True})
    print('wrote ccpredfe5424 rows', len(pred))


if __name__ == '__main__':
    main()
