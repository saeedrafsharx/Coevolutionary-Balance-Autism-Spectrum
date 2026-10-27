"""Resume missing original classifiers with the frozen Stage 2 site folds.

Original results and the completed logistic/SVM OOF predictions are never edited.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import sys
import traceback
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from abide_stage2 import inferential_features
from abide_stage2_predictive import site_group_splits

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/abide_stage2_repaired'
OUT=ROOT/'results/abide_stage2_classification_extension'
EXPECTED_MODELS=['LogisticRegression','SVM_RBF','GaussianNB','KNN','XGBoost']

def save_json(path,value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')

def model_definitions():
    models={'GaussianNB':(GaussianNB(),{}),
            'KNN':(KNeighborsClassifier(),{'clf__n_neighbors':[3,5,7,9,11], 'clf__weights':['uniform','distance']})}
    unavailable={}
    try:
        from xgboost import XGBClassifier
    except ImportError:
        unavailable['XGBoost']='Dependency missing; no substitute estimator used. Install xgboost-cpu in the project venv.'
    else:
        models['XGBoost']=(XGBClassifier(eval_metric='logloss',random_state=42,verbosity=0,n_jobs=1,tree_method='hist'),
            {'clf__max_depth':[2,3,4],'clf__n_estimators':[100,200,300],'clf__learning_rate':[.05,.1]})
    return models,unavailable

def validate_predictions(pred,metadata):
    expected_ids=set(metadata.SUB_ID)
    for (pipeline,model),block in pred.groupby(['pipeline','model']):
        if len(block)!=140 or set(block.SUB_ID)!=expected_ids or not block.SUB_ID.is_unique:
            raise ValueError(f'Incomplete or duplicate OOF pair {pipeline}/{model}')
        if not block.groupby('site').fold.nunique().eq(1).all():
            raise ValueError('A site crosses outer test folds')
        if not block.groupby('fold').y_true.nunique().eq(2).all():
            raise ValueError('An outer test fold lacks a class')
        truth=block.merge(metadata[['SUB_ID','SITE_ID','DX_GROUP']],on='SUB_ID',validate='one_to_one')
        if not truth.site.eq(truth.SITE_ID).all() or not truth.y_true.eq(truth.DX_GROUP.eq(1).astype(int)).all():
            raise ValueError('Prediction metadata differs from frozen site/diagnosis metadata')
        if not np.isfinite(block.decision_score).all():
            raise ValueError('Nonfinite OOF scores')

def conditional_summary(block,draws=1000):
    # Same RNG, site order and percentile convention as the completed Stage 2 run.
    sites=block.site.unique();rng=np.random.default_rng(303);scores=[]
    # Counts suffice for fixed-prediction balanced accuracy, and avoid repeated concatenation.
    counts=[]
    for site in sites:
        d=block[block.site.eq(site)]
        counts.append([((d.y_true==1)&(d.y_pred==1)).sum(),(d.y_true==1).sum(),
                       ((d.y_true==0)&(d.y_pred==0)).sum(),(d.y_true==0).sum()])
    counts=np.asarray(counts)
    for _ in range(draws):
        chosen=rng.choice(len(sites),len(sites),replace=True)
        tp,p,tn,n=counts[chosen].sum(0)
        if p and n:scores.append(.5*(tp/p+tn/n))
    low,high=np.percentile(scores,[2.5,97.5])
    return {'n_OOF_predictions':len(block),'balanced_accuracy':balanced_accuracy_score(block.y_true,block.y_pred),
            'conditional_site_bootstrap_CI95_low':low,'conditional_site_bootstrap_CI95_high':high,
            'valid_bootstrap_replicates':len(scores),
            'caveat':'fixed OOF predictions; does not include all model-training uncertainty'}

def run(require_xgboost=False):
    stats=OUT/'statistics';audit=OUT/'audit'
    stats.mkdir(parents=True,exist_ok=True);audit.mkdir(exist_ok=True)
    feature_path=SOURCE/'features/all_designs_raw_untrimmed.csv'
    digest=hashlib.sha256(feature_path.read_bytes()).hexdigest()
    base_summary=json.loads((SOURCE/'repair_summary.json').read_text())
    assert base_summary['full_repair_complete'] and digest==base_summary['feature_table_sha256']
    metadata=pd.read_csv(SOURCE/'audit/demographic_manifest.csv')
    all_features=pd.read_csv(feature_path)
    primary=all_features[all_features.design.eq('primary_common_106')]
    base_predictions=pd.read_csv(SOURCE/'statistics/classification_OOF_predictions.csv')
    base_folds=pd.read_csv(SOURCE/'statistics/classification_nested_site_folds.csv')
    validate_predictions(base_predictions,metadata)
    plan={'feature_sha256':digest,'expected_models':EXPECTED_MODELS,'extension_models':['GaussianNB','KNN','XGBoost'],
          'scope':'exploratory restoration of original algorithm types, using frozen revision features and site folds',
          'feature_selection_percentiles':[25,50,100],'outer_folds':5,'inner_folds':3,'inner_scoring':'balanced_accuracy',
          'GaussianNB_grid':{},'KNN_grid':{'n_neighbors':[3,5,7,9,11],'weights':['uniform','distance']},
          'XGBoost_grid':{'max_depth':[2,3,4],'n_estimators':[100,200,300],'learning_rate':[.05,.1]},
          'XGBoost_fixed':{'eval_metric':'logloss','random_state':42,'n_jobs':1,'tree_method':'hist'},
          'ComBat_used':False,'upstream_QC':'cohort-wide QC mask; no fully prospective mask-validation claim'}
    if (audit/'analysis_plan.json').exists():
        if json.loads((audit/'analysis_plan.json').read_text())!=plan:raise ValueError('Existing extension plan differs; use a separate result folder')
    else:save_json(audit/'analysis_plan.json',plan)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [feature_path,SOURCE/'statistics/classification_OOF_predictions.csv',SOURCE/'statistics/classification_conditional_intervals.csv']}
    save_json(audit/'preserved_input_hashes.json',hashes)
    models,unavailable=model_definitions()
    versions={name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','scikit-learn']}
    if 'XGBoost' in models:
        import xgboost
        versions['xgboost']=xgboost.__version__
    save_json(audit/'dependency_versions.json',versions)
    pred_path=stats/'additional_OOF_predictions.csv';fold_path=stats/'additional_nested_site_folds.csv'
    predictions=pd.read_csv(pred_path).to_dict('records') if pred_path.exists() else []
    folds=pd.read_csv(fold_path).to_dict('records') if fold_path.exists() else []
    failures=[]
    for pipeline,table in primary.groupby('pipeline'):
        data=table.merge(metadata[['SUB_ID','DX_GROUP','SITE_ID']],on='SUB_ID',validate='one_to_one').reset_index(drop=True)
        columns=[c for c in inferential_features(data) if np.isfinite(data[c]).all() and data[c].std()>1e-12]
        assert len(columns)==88
        save_json(audit/f'features_{pipeline}.json',columns)
        x=data[columns].to_numpy();y=data.DX_GROUP.eq(1).astype(int).to_numpy();sites=data.SITE_ID.to_numpy()
        outer=site_group_splits(sites,y,5)
        base_map=base_predictions[base_predictions.pipeline.eq(pipeline)&base_predictions.model.eq('LogisticRegression')].set_index('SUB_ID').fold
        for fold,(train,test) in enumerate(outer):
            assert not set(sites[train])&set(sites[test])
            assert base_map.loc[data.SUB_ID.iloc[test]].eq(fold).all()
        for name,(estimator,grid) in models.items():
            for fold,(train,test) in enumerate(outer):
                existing=[r for r in predictions if r['pipeline']==pipeline and r['model']==name and int(r['fold'])==fold]
                existing_fold=[r for r in folds if r['pipeline']==pipeline and r['model']==name and int(r['fold'])==fold]
                expected_ids=set(data.SUB_ID.iloc[test])
                if existing and existing_fold:
                    if {r['SUB_ID'] for r in existing}!=expected_ids or len(existing)!=len(test):raise ValueError('Checkpoint test IDs differ')
                    print(f'Reuse {pipeline}/{name}/fold{fold}',flush=True);continue
                # Recover a checkpoint interrupted between the two CSV writes by rerunning this fold.
                predictions=[r for r in predictions if not(r['pipeline']==pipeline and r['model']==name and int(r['fold'])==fold)]
                folds=[r for r in folds if not(r['pipeline']==pipeline and r['model']==name and int(r['fold'])==fold)]
                try:
                    inner=site_group_splits(sites[train],y[train],3)
                    assert all(not set(sites[train][a])&set(sites[train][b]) for a,b in inner)
                    if name=='KNN':assert min(len(a) for a,_ in inner)>=11
                    pipe=Pipeline([('imputer',SimpleImputer(strategy='median')),('variance',VarianceThreshold()),
                                   ('scaler',StandardScaler()),('select',SelectPercentile(f_classif)),('clf',estimator)])
                    search=GridSearchCV(pipe,{**grid,'select__percentile':[25,50,100]},cv=inner,scoring='balanced_accuracy',n_jobs=1,error_score='raise')
                    print(f'Fit {pipeline}/{name}/fold{fold}: train={len(train)}, test={len(test)}',flush=True)
                    search.fit(x[train],y[train])
                    predicted=search.predict(x[test]);score=search.predict_proba(x[test])[:,1]
                    if not np.isfinite(score).all():raise ValueError('Nonfinite probability scores')
                    majority=int(y[train].mean()>=.5)
                    folds.append({'pipeline':pipeline,'model':name,'fold':fold,'n_train':len(train),'n_test':len(test),
                        'test_sites':'|'.join(sorted(set(sites[test]))),'balanced_accuracy':balanced_accuracy_score(y[test],predicted),
                        'accuracy':accuracy_score(y[test],predicted),'F1_ASD_positive':f1_score(y[test],predicted,zero_division=0),
                        'AUC':roc_auc_score(y[test],score),'majority_baseline_accuracy':accuracy_score(y[test],np.repeat(majority,len(test))),
                        'majority_baseline_balanced_accuracy':.5,'best_inner_params':str(search.best_params_)})
                    for i,j in enumerate(test):
                        predictions.append({'pipeline':pipeline,'model':name,'SUB_ID':int(data.SUB_ID.iloc[j]),'site':sites[j],
                                            'fold':fold,'y_true':int(y[j]),'y_pred':int(predicted[i]),'decision_score':float(score[i])})
                    pd.DataFrame(predictions).to_csv(pred_path,index=False);pd.DataFrame(folds).to_csv(fold_path,index=False)
                except Exception as exc:
                    failures.append({'pipeline':pipeline,'model':name,'fold':fold,'error':str(exc),'traceback':traceback.format_exc()})
                    save_json(audit/'failures.json',failures);raise
    additional=pd.DataFrame(predictions);additional_folds=pd.DataFrame(folds)
    complete=[]
    for key,block in additional.groupby(['pipeline','model']):
        if len(block)==140 and block.SUB_ID.nunique()==140:complete.append(key)
    additional=additional.loc[[tuple(k) in complete for k in additional[['pipeline','model']].to_numpy()]]
    combined=pd.concat([base_predictions,additional],ignore_index=True)
    combined_folds=pd.concat([base_folds,additional_folds],ignore_index=True)
    validate_predictions(combined,metadata)
    combined.to_csv(stats/'classification_OOF_predictions.csv',index=False)
    combined_folds.to_csv(stats/'classification_nested_site_folds.csv',index=False)
    combined_folds.groupby(['pipeline','model'])[['balanced_accuracy','accuracy','F1_ASD_positive','AUC']].agg(['mean','std','count']).to_csv(stats/'classification_fold_summary.csv')
    intervals=[]
    for (pipeline,name),block in combined.groupby(['pipeline','model']):
        intervals.append({'pipeline':pipeline,'model':name,**conditional_summary(block)})
    intervals=pd.DataFrame(intervals)
    # An identical new bootstrap implementation must reproduce the original intervals.
    old=pd.read_csv(SOURCE/'statistics/classification_conditional_intervals.csv')
    check=intervals.merge(old,on=['pipeline','model'],suffixes=('_new','_old'),validate='one_to_one')
    for field in ['balanced_accuracy','conditional_site_bootstrap_CI95_low','conditional_site_bootstrap_CI95_high']:
        assert np.allclose(check[field+'_new'],check[field+'_old'],atol=1e-12),field
    intervals.to_csv(stats/'classification_conditional_intervals.csv',index=False)
    scope=json.loads((SOURCE/'statistics/classification_scope.json').read_text())
    scope.update({'expected_models':EXPECTED_MODELS,'complete_pipeline_model_pairs':len(intervals),
                  'completed_models':sorted(combined.model.unique()),'pending_models':sorted(set(EXPECTED_MODELS)-set(combined.model.unique())),
                  'original_algorithm_types_restored':len(intervals)==10,'new_model_score_type':'positive-class probability',
                  'previous_logistic_SVM_predictions_reused_unchanged':True})
    save_json(stats/'classification_scope.json',scope)
    save_json(audit/'unavailable_models.json',unavailable)
    summary={'finished_utc':datetime.now(timezone.utc).isoformat(),'feature_sha256':digest,'imaging_recalculation':False,'completed_pipeline_model_pairs':len(intervals),
             'expected_pipeline_model_pairs':10,'subjects_per_pair':140,'all_original_classifiers_complete':len(intervals)==10,
             'pending_models':scope['pending_models'],'original_results_preserved':True,'failures':len(failures),
             'ready_for_submission':False,'source_statistics':str(SOURCE.relative_to(ROOT))}
    save_json(OUT/'completion_summary.json',summary)
    print(intervals[['pipeline','model','balanced_accuracy','conditional_site_bootstrap_CI95_low','conditional_site_bootstrap_CI95_high']].to_string(index=False),flush=True)
    if require_xgboost and 'XGBoost' in scope['pending_models']:raise RuntimeError('XGBoost remains unavailable; other completed models are saved and reusable')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--require-xgboost',action='store_true');args=parser.parse_args()
    run(args.require_xgboost)
