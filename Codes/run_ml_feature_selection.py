"""Exploratory nested selection-method comparison; preserves the original five models."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import importlib.metadata
import json
import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from abide_feature_selection import RedundancyPruner, RankedSelector
from abide_stage2 import inferential_features
from abide_stage2_predictive import site_group_splits
from classification_output_source import classification_statistics_directory
from complete_original_classifiers import validate_predictions, conditional_summary

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/abide_stage2_repaired'
OUT=ROOT/'results/abide_ml_feature_selection_review'

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def json_write(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
def models():
 return {
 'LogisticRegression':(LogisticRegression(max_iter=3000,class_weight='balanced'),{'clf__C':[.01,.1,1,10]}),
 'SVM_RBF':(SVC(class_weight='balanced'),{'clf__C':[.1,1,10],'clf__gamma':['scale']}),
 'GaussianNB':(GaussianNB(),{}),
 'KNN':(KNeighborsClassifier(),{'clf__n_neighbors':[3,5,7,9,11],'clf__weights':['uniform','distance']}),
 'XGBoost':(XGBClassifier(eval_metric='logloss',random_state=42,verbosity=0,n_jobs=1,tree_method='hist'),
             {'clf__max_depth':[2,3],'clf__n_estimators':[100,200],'clf__learning_rate':[.05,.1]})}

def selection_records(fit,columns,pipeline,model,fold):
 names=np.array(fit.named_steps['imputer'].get_feature_names_out(columns))
 variable=names[fit.named_steps['variance'].get_support()]
 nonredundant=variable[fit.named_steps['deduplicate'].get_support()]
 selector=fit.named_steps['select'];chosen=nonredundant[selector.get_support()]
 scores=dict(zip(nonredundant,selector.scores_))
 return [{'pipeline':pipeline,'model':model,'fold':fold,'feature':feature,
          'passed_variance':feature in variable,'passed_redundancy':feature in nonredundant,
          'selected':feature in chosen,'selector_method':selector.method,
          'selection_score':float(scores.get(feature,np.nan)),
          'l1_zero_coefficient_selected_count':selector.zero_coefficient_selected_}
          for feature in columns]

def baseline_selection_audit(primary,metadata,stats,out):
 path=out/'baseline_feature_selection.csv'
 if path.exists() and (out/'baseline_selector_reconstruction_checks.csv').exists():
  check=pd.read_csv(out/'baseline_selector_reconstruction_checks.csv')
  if len(check)!=50 or not check.prediction_matches.all():raise RuntimeError('Existing baseline selector audit is incomplete or differs')
  return
 from sklearn.feature_selection import SelectPercentile,f_classif
 baseline=pd.read_csv(stats/'classification_OOF_predictions.csv')
 fold_parameters=pd.read_csv(stats/'classification_nested_site_folds.csv')
 records=[];checks=[]
 for pipeline,table in primary.groupby('pipeline'):
  data=table.merge(metadata[['SUB_ID','DX_GROUP','SITE_ID']],on='SUB_ID',validate='one_to_one').reset_index(drop=True)
  columns=[c for c in inferential_features(data) if np.isfinite(data[c]).all() and data[c].std()>1e-12]
  X=data[columns].to_numpy();y=data.DX_GROUP.eq(1).astype(int).to_numpy();sites=data.SITE_ID.to_numpy()
  for row in fold_parameters[fold_parameters.pipeline.eq(pipeline)].itertuples():
   train,test=site_group_splits(sites,y,5)[int(row.fold)]
   estimator=models()[row.model][0]
   pipe=Pipeline([('imputer',SimpleImputer(strategy='median')),('variance',VarianceThreshold()),
                  ('scaler',StandardScaler()),('select',SelectPercentile(f_classif)),('clf',estimator)])
   # Original XGBoost hyperparameters may extend beyond the compact follow-up grid.
   params=ast.literal_eval(row.best_inner_params);pipe.set_params(**params);pipe.fit(X[train],y[train])
   predicted=pipe.predict(X[test])
   score=pipe.decision_function(X[test]) if hasattr(pipe,'decision_function') else pipe.predict_proba(X[test])[:,1]
   original=baseline[baseline.pipeline.eq(pipeline)&baseline.model.eq(row.model)].set_index('SUB_ID').loc[data.SUB_ID.iloc[test]]
   checks.append({'pipeline':pipeline,'model':row.model,'fold':int(row.fold),'prediction_matches':bool(np.array_equal(predicted,original.y_pred)),
                  'score_max_absolute_difference':float(np.max(np.abs(score-original.decision_score.to_numpy()))),
                  'percentile':int(params['select__percentile'])})
   names=np.array(pipe.named_steps['imputer'].get_feature_names_out(columns))
   variable=names[pipe.named_steps['variance'].get_support()]
   selected=variable[pipe.named_steps['select'].get_support()]
   for feature in columns:
    records.append({'pipeline':pipeline,'model':row.model,'fold':int(row.fold),'feature':feature,
                    'selected':feature in selected,'selection_percentile':int(params['select__percentile'])})
 pd.DataFrame(records).to_csv(path,index=False)
 check=pd.DataFrame(checks);check.to_csv(out/'baseline_selector_reconstruction_checks.csv',index=False)
 if not check.prediction_matches.all():raise RuntimeError('Baseline fitted-prediction reconstruction differs; inspect audit before calling feature names exact')
 print('Baseline selectors reconstructed: all 50 saved outer-fold predictions match.',flush=True)

def paired_site_delta(block,baseline):
 matched=block.merge(baseline[['SUB_ID','y_pred']],on='SUB_ID',suffixes=('','_baseline'),validate='one_to_one')
 sites=matched.site.unique();counts=[]
 for site in sites:
  data=matched[matched.site.eq(site)];positive=data.y_true.eq(1);negative=~positive
  counts.append([int((positive&data.y_pred.eq(1)).sum()),int((negative&data.y_pred.eq(0)).sum()),
                 int((positive&data.y_pred_baseline.eq(1)).sum()),int((negative&data.y_pred_baseline.eq(0)).sum()),int(positive.sum()),int(negative.sum())])
 counts=np.array(counts);rng=np.random.default_rng(303);values=[]
 for _ in range(1000):
  tp,tn,btp,btn,p,n=counts[rng.choice(len(sites),len(sites),replace=True)].sum(0)
  if p and n:values.append(.5*((tp-btp)/p+(tn-btn)/n))
 return np.percentile(values,[2.5,97.5])

def main():
 stats=OUT/'statistics';audit=OUT/'audit';cvout=OUT/'inner_cv'
 for path in [stats,audit,cvout]:path.mkdir(parents=True,exist_ok=True)
 base_stats=classification_statistics_directory()
 scope=json.loads((base_stats/'classification_scope.json').read_text());assert scope['original_algorithm_types_restored']
 feature_path=SOURCE/'features/all_designs_raw_untrimmed.csv';metadata_path=SOURCE/'audit/demographic_manifest.csv'
 metadata=pd.read_csv(metadata_path);all_features=pd.read_csv(feature_path);primary=all_features[all_features.design.eq('primary_common_106')]
 baseline=pd.read_csv(base_stats/'classification_OOF_predictions.csv');validate_predictions(baseline,metadata)
 plan={'feature_sha256':digest(feature_path),'metadata_sha256':digest(metadata_path),
       'baseline_predictions_sha256':digest(base_stats/'classification_OOF_predictions.csv'),
       'specification_sha256':digest(ROOT/'revision/analysis_plan/ML_FEATURE_SELECTION_EXTENSION.md'),
       'feature_selection_configs':10,'selection_methods':['all','anova','mutual_info','l1'],
       'k_values':[5,10,20],'redundancy_threshold':1-1e-10,'outer_results_already_seen':True,
       'validation':'exploratory nested 5 outer/3 inner site folds; same folds as baseline',
       'original_inputs_preserved':True}
 if (audit/'plan.json').exists():
  if json.loads((audit/'plan.json').read_text())!=plan:raise ValueError('Existing follow-up has different frozen inputs/specification')
 else:json_write(audit/'plan.json',plan)
 json_write(audit/'versions.json',{name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','scikit-learn','xgboost-cpu','joblib']})
 baseline_selection_audit(primary,metadata,base_stats,audit)
 cache=Memory(OUT/'cache',verbose=0)
 pred_path=stats/'classification_OOF_predictions.csv';fold_path=stats/'classification_nested_site_folds.csv';selection_path=stats/'feature_selection_by_fold.csv'
 predictions=pd.read_csv(pred_path).to_dict('records') if pred_path.exists() else []
 folds=pd.read_csv(fold_path).to_dict('records') if fold_path.exists() else []
 selections=pd.read_csv(selection_path).to_dict('records') if selection_path.exists() else []
 for pipeline,table in primary.groupby('pipeline'):
  data=table.merge(metadata[['SUB_ID','DX_GROUP','SITE_ID']],on='SUB_ID',validate='one_to_one').reset_index(drop=True)
  columns=[c for c in inferential_features(data) if np.isfinite(data[c]).all() and data[c].std()>1e-12];assert len(columns)==88
  X=data[columns].to_numpy();y=data.DX_GROUP.eq(1).astype(int).to_numpy();sites=data.SITE_ID.to_numpy()
  for name,(estimator,grid) in models().items():
   for fold,(train,test) in enumerate(site_group_splits(sites,y,5)):
    key=lambda row:row['pipeline']==pipeline and row['model']==name and int(row['fold'])==fold
    existing=[r for r in predictions if key(r)]
    if existing and any(key(r) for r in folds) and sum(key(r) for r in selections)==88:
     assert {r['SUB_ID'] for r in existing}==set(data.SUB_ID.iloc[test]) and len(existing)==len(test)
     print(f'Reuse {pipeline}/{name}/fold{fold}',flush=True);continue
    predictions=[r for r in predictions if not key(r)];folds=[r for r in folds if not key(r)];selections=[r for r in selections if not key(r)]
    inner=site_group_splits(sites[train],y[train],3)
    assert not set(sites[train])&set(sites[test])
    assert all(not set(sites[train][a])&set(sites[train][b]) for a,b in inner)
    base_map=baseline[baseline.pipeline.eq(pipeline)&baseline.model.eq(name)].set_index('SUB_ID').fold
    assert base_map.loc[data.SUB_ID.iloc[test]].eq(fold).all()
    pipe=Pipeline([('imputer',SimpleImputer(strategy='median')),('variance',VarianceThreshold()),
        ('deduplicate',RedundancyPruner()),('scaler',StandardScaler()),('select',RankedSelector()),('clf',estimator)],memory=cache)
    configurations=[{**grid,'select__method':['all'],'select__k':[20]}]+[
       {**grid,'select__method':[method],'select__k':[5,10,20]} for method in ['anova','mutual_info','l1']]
    print(f'Fit {pipeline}/{name}/fold{fold}: {len(ParameterGrid(configurations))} inner candidates',flush=True)
    search=GridSearchCV(pipe,configurations,cv=inner,scoring='balanced_accuracy',n_jobs=1,error_score='raise',return_train_score=False)
    search.fit(X[train],y[train]);fit=search.best_estimator_
    pd.DataFrame(search.cv_results_).to_csv(cvout/f'{pipeline}_{name}_fold{fold}.csv',index=False)
    predicted=fit.predict(X[test]);score=fit.decision_function(X[test]) if hasattr(fit,'decision_function') else fit.predict_proba(X[test])[:,1]
    assert np.isfinite(score).all()
    selected=selection_records(fit,columns,pipeline,name,fold);selections.extend(selected)
    method=fit.named_steps['select'].method;count=sum(r['selected'] for r in selected)
    folds.append({'pipeline':pipeline,'model':name,'fold':fold,'n_train':len(train),'n_test':len(test),
                  'test_sites':'|'.join(sorted(set(sites[test]))),'selection_method':method,'n_selected':count,
                  'n_nonredundant':int(fit.named_steps['deduplicate'].get_support().sum()),
                  'inner_best_balanced_accuracy':search.best_score_,'balanced_accuracy':balanced_accuracy_score(y[test],predicted),
                  'AUC':roc_auc_score(y[test],score),'best_inner_params':json.dumps(search.best_params_,sort_keys=True),
                  'l1_zero_coefficient_selected_count':fit.named_steps['select'].zero_coefficient_selected_})
    predictions.extend([{'pipeline':pipeline,'model':name,'fold':fold,'SUB_ID':int(data.SUB_ID.iloc[j]),'site':sites[j],
                         'y_true':int(y[j]),'y_pred':int(predicted[i]),'decision_score':float(score[i])} for i,j in enumerate(test)])
    pd.DataFrame(predictions).to_csv(pred_path,index=False);pd.DataFrame(folds).to_csv(fold_path,index=False);pd.DataFrame(selections).to_csv(selection_path,index=False)
    print(f'Completed {pipeline}/{name}/fold{fold}: {method}, {count} features',flush=True)
 pred=pd.DataFrame(predictions);validate_predictions(pred,metadata);assert len(pred)==1400
 rows=[]
 for (pipeline,name),block in pred.groupby(['pipeline','model']):
  old=baseline[baseline.pipeline.eq(pipeline)&baseline.model.eq(name)]
  b=balanced_accuracy_score(old.y_true,old.y_pred);delta=paired_site_delta(block,old)
  rows.append({'pipeline':pipeline,'model':name,**conditional_summary(block),'accuracy':accuracy_score(block.y_true,block.y_pred),
               'AUC':roc_auc_score(block.y_true,block.decision_score),'F1':f1_score(block.y_true,block.y_pred),
               'baseline_balanced_accuracy':b,'delta_balanced_accuracy':balanced_accuracy_score(block.y_true,block.y_pred)-b,
               'paired_delta_CI95_low':delta[0],'paired_delta_CI95_high':delta[1],
               'scope':'exploratory reuse of known outer folds; no independent validation or model permutation test'})
 summary=pd.DataFrame(rows);summary.to_csv(stats/'classification_conditional_intervals.csv',index=False)
 selected=pd.DataFrame(selections);selected.groupby(['pipeline','model','feature']).selected.agg(['sum','mean']).reset_index().to_csv(stats/'feature_selection_frequencies.csv',index=False)
 pd.DataFrame(folds).groupby(['pipeline','model','selection_method']).size().rename('outer_folds').reset_index().to_csv(stats/'selector_method_counts.csv',index=False)
 json_write(OUT/'completion_summary.json',{'finished_utc':datetime.now(timezone.utc).isoformat(),'complete':True,
        'classifier_pipeline_pairs':10,'OOF_subjects_per_pair':140,'feature_sha256':plan['feature_sha256'],
        'exploratory_post_baseline_review':True,'imaging_recalculation':False,'baseline_outputs_preserved':True,
        'ready_for_submission':False})
 print(summary[['pipeline','model','balanced_accuracy','baseline_balanced_accuracy','delta_balanced_accuracy','AUC']].to_string(index=False),flush=True)

if __name__=='__main__':main()
