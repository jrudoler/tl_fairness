"""Audit refitted controls, allowing negligible numerical rather than byte differences."""
import numpy as np


def audit_unchanged(raw, baseline):
    keys=['scenario','train_size','test_size','case','outcome_model','group_model','method','replicate']
    joined=raw.merge(baseline,on=keys,suffixes=('','_original'),validate='one_to_one')
    assert len(joined)==len(raw)
    assert joined.data_hash.eq(joined.data_hash_original).all(), 'Original draws changed'
    audit={'absolute_tolerance':1e-8,'hash_mismatches':{}}
    boosted=['default','higher_capacity']
    for role in ['outcome','group']:
        col=role+'_hash';oracle=joined[joined[role+'_model']=='oracle']
        assert oracle[col].eq(oracle[col+'_original']).all(), 'Oracle predictions changed'
        fixed=joined[~joined[role+'_model'].isin(boosted)]
        audit['hash_mismatches'][role]=int((fixed[col]!=fixed[col+'_original']).sum())
    controls=joined[(~joined.outcome_model.isin(boosted))&
                    ((~joined.group_model.isin(boosted))|(joined.method=='Model fairness'))]
    audit['checked_method_rows']=len(controls);audit['maximum_absolute_differences']={}
    for col in ['estimate','ci_low','ci_high','se','width']:
        a,b=controls[col],controls[col+'_original']
        np.testing.assert_allclose(a,b,atol=1e-8,rtol=0,err_msg='Unchanged control differs materially')
        audit['maximum_absolute_differences'][col]=float(abs(a-b).max())
    truth=controls['truth_original'] if 'truth_original' in controls else controls['truth']
    old=(controls.ci_low_original<=truth)&(truth<=controls.ci_high_original)
    new=(controls.ci_low<=truth)&(truth<=controls.ci_high)
    audit['coverage_events_changed']=int((old!=new).sum())
    assert audit['coverage_events_changed']==0
    return audit
