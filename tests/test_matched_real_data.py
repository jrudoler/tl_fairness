"""Check paired estimands against public estimators and fixed-score formulas."""
import numpy as np
import pandas as pd

from analysis.matched_real_data.run import summarize_predictions
from tlfair.metrics import parity, opportunity, prob_parity, prob_opportunity


class FixedClassifier:
    def __init__(self, probabilities: np.ndarray):
        self.probabilities = probabilities

    def fit(self, x, y):
        return self

    def predict_proba(self, x):
        return self.probabilities

    def predict(self, x):
        return (self.probabilities[:,1] >= .5).astype(int)


def test_shared_predictions_reproduce_all_four_public_data_fairness_estimators():
    scores = np.linspace(.08,.92,12)
    decisions = (scores >= .5).astype(int)
    groups = np.array([0]*5+[1]*7)
    outcomes = np.array([0,1,1,0,1,1,0,1,0,1,1,1])
    pi = np.column_stack([np.linspace(.8,.2,12),np.linspace(.2,.8,12)])
    rho = np.tile([.2,.3,.1,.4],(12,1))
    paired = summarize_predictions(scores,decisions,pi,rho,outcomes,groups)
    x = pd.DataFrame({'feature':np.arange(12)})
    for name,estimator,weights in [('parity',parity,pi),('opportunity',opportunity,rho),
                                   ('prob_parity',prob_parity,pi),('prob_opp',prob_opportunity,rho)]:
        expected = estimator(x,x,outcomes,outcomes,groups,groups,
                             FixedClassifier(np.column_stack([1-scores,scores])),FixedClassifier(weights))
        actual = paired['data_fairness'][name]
        np.testing.assert_allclose([actual[0],*actual[1]],[expected[0],*expected[1]])
        prediction = scores if name.startswith('prob_') else decisions
        eligible = outcomes == 1 if name in ['prob_opp','opportunity'] else np.ones(12,dtype=bool)
        samples = [prediction[eligible & (groups == g)] for g in [0,1]]
        mean = samples[1].mean()-samples[0].mean()
        se = np.sqrt(len(groups)/(len(groups)-1) * sum(s.var(ddof=0)/len(s) for s in samples))
        estimate,interval = paired['model_fairness'][name]
        np.testing.assert_allclose([estimate,*interval],[mean,mean-1.96*se,mean+1.96*se])
    for name in ['parity','opportunity']:
        assert paired['data_fairness'][name] == paired['model_fairness'][name]


def test_comparison_table_prioritizes_probabilistic_metrics_and_keeps_cmi_data_only():
    from analysis.table_real_data_inference.run import comparison_rows
    values = {key:(.123456,(.100001,.146911)) for key in ['prob_parity','prob_opp','parity','opportunity']}
    matched = {'data_fairness':values,'model_fairness':values}
    rows = comparison_rows({'cmi_tmle':(.000134,(-.000101,.000373))},matched,'Adult',True)
    assert [row['Metric'] for row in rows] == [
        'Prob. Parity','Prob. Eq. Opp.','Parity (thresholded)','Eq. Opp. (thresholded)','CMI (TMLE)']
    assert rows[0]['Data fairness'] == '0.123 (0.100, 0.147)'
    assert rows[-1]['Data fairness'] == '0.0001 (-0.0001, 0.0004)'
    assert rows[-1]['Model fairness'] == '--'
