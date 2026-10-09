import numpy as np
from src.dsp import regions, inject, repair, detection_counts, restoration_metrics


def test_injection_preserves_exact_ground_truth_and_clean_exterior():
    x=np.sin(np.arange(16000)*.1)*.1
    y,n,m,_=inject(x,np.ones(160),4000,0)
    assert regions(m)==[(4000,4160)]
    assert np.array_equal(y[~m],x[~m])
    assert np.allclose(y-x,n)


def test_local_repair_never_changes_samples_outside_prediction():
    rng=np.random.default_rng(1);x=rng.normal(size=1000)
    m=np.zeros(1000,bool);m[0:10]=True;m[300:400]=True;m[-10:]=True
    for method in ['attenuate','interpolate']:
        y=repair(x,m,method)
        assert np.array_equal(y[~m],x[~m])
        assert np.all(np.isfinite(y))
    assert np.array_equal(repair(x,np.zeros(1000,bool)),x)


def test_matching_does_not_double_credit_one_inserted_event():
    t=np.zeros(100,bool);t[30:60]=True
    p=np.zeros(100,bool);p[28:40]=True;p[50:62]=True
    r=detection_counts(p,t,1000)
    assert (r['tp'],r['fp'],r['fn'])==(1,1,0)
    clean=np.zeros(100,bool)
    assert detection_counts(p,clean)['fp']==2


def test_region_metrics_penalise_collateral_changes():
    x=np.linspace(-.1,.1,1000);m=np.zeros(1000,bool);m[100:150]=True
    y=x.copy();y[~m]+=.01
    r=restoration_metrics(x,y,m,x)
    assert r['event_rmse']==0
    assert np.isclose(r['clean_rmse'],.01)
