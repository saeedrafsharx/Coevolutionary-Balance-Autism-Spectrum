"""Clinical and site-held-out predictive complements for the frozen primary design."""
if 'residual_partial_correlation' not in globals():
    from abide_stage2 import *

BEHAVIOR = {'ADI_R_SOCIAL_TOTAL_A': 'ADI_R_Social', 'ADI_R_VERBAL_TOTAL_BV': 'ADI_R_Verbal',
            'ADI_R_ONSET_TOTAL_D': 'ADI_R_Onset', 'ADOS_TOTAL': 'ADOS_Total', 'ADOS_COMM': 'ADOS_Communication'}


def site_group_splits(groups, y, n_splits):
    """Size-balanced, deterministic site-disjoint folds; reject single-class splits."""
    groups, y = np.asarray(groups), np.asarray(y)
    sites, inverse = np.unique(groups, return_inverse=True)
    if len(sites) < n_splits:
        raise ValueError('Not enough sites for requested folds')
    counts = np.bincount(inverse)
    load, assigned = np.zeros(n_splits, dtype=int), np.empty(len(sites), dtype=int)
    for site_index in np.argsort(-counts, kind='stable'):
        fold = int(np.argmin(load))
        assigned[site_index] = fold
        load[fold] += counts[site_index]
    splits = []
    for fold in range(n_splits):
        test = np.flatnonzero(assigned[inverse] == fold)
        train = np.flatnonzero(assigned[inverse] != fold)
        if len(np.unique(y[train])) != 2 or len(np.unique(y[test])) != 2:
            raise ValueError(f'Site-fold {fold} has only one class; no partial-cohort aggregate is valid')
        splits.append((train, test))
    return splits


def clinical_analyses(primary, metadata, output, n_perm=1000):
    from sklearn.impute import SimpleImputer
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GroupKFold, GridSearchCV
    from sklearn.metrics import r2_score, mean_squared_error
    out = Path(output)
    correlations, partial, predictions, regression, latent, skips = [], [], [], [], [], []
    for pipeline, table in primary.groupby('pipeline'):
        data = table.merge(metadata, on='SUB_ID', validate='one_to_one')
        data = data[data.DX_GROUP.eq(1)].copy()
        feature_cols = [c for c in inferential_features(data) if np.isfinite(data[c]).all() and data[c].std() > 1e-12]
        for source, label in BEHAVIOR.items():
            if source not in data:
                skips.append({'pipeline': pipeline, 'analysis': label, 'reason': 'clinical_column_absent'})
                continue
            values = pd.to_numeric(data[source], errors='coerce')
            data[label] = values.where(values >= 0, np.nan)
            subset = data.loc[data[label].notna()].reset_index(drop=True)
            if len(subset) < 20 or subset[label].std() == 0:
                skips.append({'pipeline': pipeline, 'analysis': label, 'reason': 'n_below_20_or_constant_outcome'})
                continue
            for feature in feature_cols:
                r, p = stats.pearsonr(subset[feature], subset[label])
                rho, sp = stats.spearmanr(subset[feature], subset[label])
                correlations.append({'pipeline': pipeline, 'score': label, 'feature': feature,
                                     'n': len(subset), 'Pearson_r': r, 'Pearson_p': p, 'Spearman_rho': rho, 'Spearman_p': sp})
            try:
                controls = model_design(subset, include_group=False)
                r, p, df = residual_partial_correlation(subset.Whole_Energy_global, subset[label], controls)
                partial.append({'pipeline': pipeline, 'score': label, 'n': len(subset), 'partial_r': r, 'p': p, 'df': df})
            except ValueError as exc:
                skips.append({'pipeline': pipeline, 'analysis': label+'_partial_H', 'reason': str(exc)})
            groups = subset.SITE_ID.to_numpy()
            n_sites = subset.SITE_ID.nunique()
            if n_sites < 4:
                skips.append({'pipeline': pipeline, 'analysis': label+'_PLSR', 'reason': 'fewer_than_4_sites'})
                continue
            x, y = subset[feature_cols].to_numpy(), subset[label].to_numpy()
            outer = GroupKFold(n_splits=min(5, n_sites))
            oof, baseline = np.full(len(y), np.nan), np.full(len(y), np.nan)
            for fold, (train, test) in enumerate(outer.split(x, y, groups)):
                inner = GroupKFold(n_splits=min(3, len(np.unique(groups[train]))))
                splits = list(inner.split(x[train], y[train], groups[train]))
                maximum_components = min(3, x.shape[1], min(len(a)-1 for a, _ in splits))
                if maximum_components < 1 or any(np.std(y[train][a]) == 0 for a, _ in splits):
                    skips.append({'pipeline': pipeline, 'analysis': label+'_PLSR', 'reason': 'inner_training_insufficient_or_constant', 'fold': fold})
                    continue
                pipe = Pipeline([('imputer', SimpleImputer(strategy='median')),
                                 ('variance', VarianceThreshold()), ('pls', PLSRegression(scale=True))])
                search = GridSearchCV(pipe, {'pls__n_components': list(range(1, maximum_components+1))}, cv=splits,
                                      scoring='neg_mean_squared_error', n_jobs=1, error_score='raise')
                search.fit(x[train], y[train], groups=groups[train])
                oof[test] = search.predict(x[test]).ravel()
                baseline[test] = y[train].mean()
                for j in test:
                    predictions.append({'pipeline': pipeline, 'score': label, 'SUB_ID': int(subset.SUB_ID.iloc[j]),
                                        'site': groups[j], 'fold': fold, 'observed': y[j], 'predicted': oof[j],
                                        'fold_training_mean_baseline': baseline[j]})
            if not np.isfinite(oof).all():
                skips.append({'pipeline': pipeline, 'analysis': label+'_PLSR', 'reason': 'incomplete_OOF_predictions_no_aggregate_score'})
                continue
            regression.append({'pipeline': pipeline, 'score': label, 'n': len(y), 'n_sites': n_sites,
                               'OOF_R2': r2_score(y, oof), 'OOF_MSE': mean_squared_error(y, oof),
                               'training_mean_baseline_MSE': mean_squared_error(y, baseline),
                               'validation': 'nested GroupKFold by site; all fitted steps within folds'})
        # Descriptive PLSC with explicit site-constrained permutations.
        for name, scores in [('ADOS_block', ['ADOS_Total', 'ADOS_Communication']), ('all_five_scores', list(BEHAVIOR.values()))]:
            if any(c not in data for c in scores):
                continue
            block = data.dropna(subset=scores).reset_index(drop=True)
            if len(block) < 20:
                skips.append({'pipeline': pipeline, 'analysis': name+'_PLSC', 'reason': 'complete_case_n_below_20'})
                continue
            x = block[feature_cols].to_numpy()
            y = block[scores].to_numpy()
            try:
                controls = model_design(block, include_group=False).to_numpy()
            except ValueError as exc:
                skips.append({'pipeline': pipeline, 'analysis': name+'_PLSC', 'reason': str(exc)})
                continue
            x = x-controls@np.linalg.lstsq(controls, x, rcond=None)[0]
            y = y-controls@np.linalg.lstsq(controls, y, rcond=None)[0]
            x = (x-x.mean(0))/(x.std(0, ddof=1)+1e-12)
            y = (y-y.mean(0))/(y.std(0, ddof=1)+1e-12)
            singular = np.linalg.svd(y.T@x/(len(x)-1), compute_uv=False)
            rng = np.random.default_rng(101)
            ge = np.zeros(len(singular))
            movable = 0
            sites = block.SITE_ID.to_numpy()
            for s in np.unique(sites):
                movable += int(np.sum(sites == s)) if np.sum(sites == s) > 1 else 0
            if movable < 10:
                skips.append({'pipeline': pipeline, 'analysis': name+'_PLSC', 'reason': 'too_few_permutable_within_site_rows'})
                continue
            for _ in range(n_perm):
                indices = np.arange(len(y))
                for site in np.unique(sites):
                    selected = np.flatnonzero(sites == site)
                    indices[selected] = rng.permutation(selected)
                perm = np.linalg.svd(y[indices].T@x/(len(x)-1), compute_uv=False)
                ge += perm >= singular
            for k, value in enumerate(singular):
                latent.append({'pipeline': pipeline, 'block': name, 'n': len(block), 'LV': k+1,
                               'singular_value': value, 'squared_covariance_fraction': value**2/np.sum(singular**2),
                               'permutation_p': (ge[k]+1)/(n_perm+1), 'permutations': n_perm,
                               'method': 'nuisance-residualized PLSC; within-site residual permutations; exploratory exchangeability assumptions'})
    corr = pd.DataFrame(correlations)
    if len(corr):
        corr['Pearson_p_BH'] = adjusted_pvalues(corr.Pearson_p)
        corr['Spearman_p_BH'] = adjusted_pvalues(corr.Spearman_p)
    corr.to_csv(out / 'clinical_all_feature_correlations.csv', index=False)
    part = pd.DataFrame(partial)
    if len(part):
        part['p_BH'] = adjusted_pvalues(part.p)
    part.to_csv(out / 'clinical_H_partial_correlations.csv', index=False)
    pd.DataFrame(regression).to_csv(out / 'clinical_PLSR_site_heldout.csv', index=False)
    pd.DataFrame(predictions).to_csv(out / 'clinical_PLSR_OOF_predictions.csv', index=False)
    table = pd.DataFrame(latent)
    if len(table):
        table['p_BH_exploratory'] = adjusted_pvalues(table.permutation_p)
    table.to_csv(out / 'clinical_PLSC.csv', index=False)
    pd.DataFrame(skips).to_csv(out / 'clinical_unavailable.csv', index=False)


def classifier_analyses(primary, metadata, output):
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import VarianceThreshold, SelectPercentile, f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
    out = Path(output)
    predictions, folds, unavailable, expected_counts = [], [], [], {}
    for pipeline, table in primary.groupby('pipeline'):
        data = table.merge(metadata[['SUB_ID', 'DX_GROUP', 'SITE_ID']], on='SUB_ID', validate='one_to_one').reset_index(drop=True)
        columns = [c for c in inferential_features(data) if np.isfinite(data[c]).all() and data[c].std() > 1e-12]
        x, y, groups = data[columns].to_numpy(), data.DX_GROUP.eq(1).astype(int).to_numpy(), data.SITE_ID.to_numpy()
        outer_splits = site_group_splits(groups, y, 5)
        expected_counts[pipeline] = len(data)
        models = {'LogisticRegression': (LogisticRegression(max_iter=3000, class_weight='balanced'), {'clf__C': [.01, .1, 1, 10]}),
                  'SVM_RBF': (SVC(class_weight='balanced'), {'clf__C': [.1, 1, 10], 'clf__gamma': ['scale']})}
        for model_name, (model, grid) in models.items():
            for fold, (train, test) in enumerate(outer_splits):
                if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
                    unavailable.append({'pipeline': pipeline, 'model': model_name, 'fold': fold, 'reason': 'single_class_train_or_test'})
                    continue
                splits = site_group_splits(groups[train], y[train], 3)
                if any(len(np.unique(y[train][a])) < 2 or len(np.unique(y[train][b])) < 2 for a, b in splits):
                    unavailable.append({'pipeline': pipeline, 'model': model_name, 'fold': fold, 'reason': 'single_class_inner_fold'})
                    continue
                pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('variance', VarianceThreshold()),
                                 ('scaler', StandardScaler()), ('select', SelectPercentile(f_classif)), ('clf', model)])
                search = GridSearchCV(pipe, {**grid, 'select__percentile': [25, 50, 100]}, cv=splits,
                                      scoring='balanced_accuracy', n_jobs=1, error_score='raise')
                search.fit(x[train], y[train])
                pred = search.predict(x[test])
                score = search.decision_function(x[test])
                majority = int(np.mean(y[train]) >= .5)
                folds.append({'pipeline': pipeline, 'model': model_name, 'fold': fold, 'n_train': len(train), 'n_test': len(test),
                    'test_sites': '|'.join(sorted(np.unique(groups[test]))),
                    'balanced_accuracy': balanced_accuracy_score(y[test], pred), 'accuracy': accuracy_score(y[test], pred),
                    'F1_ASD_positive': f1_score(y[test], pred, zero_division=0), 'AUC': roc_auc_score(y[test], score),
                    'majority_baseline_accuracy': accuracy_score(y[test], np.repeat(majority, len(test))),
                    'majority_baseline_balanced_accuracy': .5, 'best_inner_params': str(search.best_params_)})
                for position, j in enumerate(test):
                    predictions.append({'pipeline': pipeline, 'model': model_name, 'SUB_ID': int(data.SUB_ID.iloc[j]),
                                        'site': groups[j], 'fold': fold, 'y_true': int(y[j]),
                                        'y_pred': int(pred[position]), 'decision_score': float(score[position])})
    pd.DataFrame(folds).to_csv(out / 'classification_nested_site_folds.csv', index=False)
    pd.DataFrame(predictions).to_csv(out / 'classification_OOF_predictions.csv', index=False)
    pd.DataFrame(unavailable).to_csv(out / 'classification_unavailable_folds.csv', index=False)
    table = pd.DataFrame(folds)
    pred = pd.DataFrame(predictions)
    complete_keys = set()
    if len(pred):
        for key, block in pred.groupby(['pipeline', 'model']):
            if len(block) == expected_counts[key[0]] and block.SUB_ID.nunique() == expected_counts[key[0]]:
                complete_keys.add(key)
    if len(table):
        complete_table = table.loc[[tuple(k) in complete_keys for k in table[['pipeline', 'model']].to_numpy()]]
        complete_table.groupby(['pipeline', 'model'])[['balanced_accuracy', 'accuracy', 'F1_ASD_positive', 'AUC']].agg(
            ['mean', 'std', 'count']).to_csv(out / 'classification_fold_summary.csv')
    # Conditional site bootstrap of fixed OOF predictions; not a full training-process CI.
    intervals = []
    if len(pred):
        for (pipeline, model_name), block in pred.groupby(['pipeline', 'model']):
            if (pipeline, model_name) not in complete_keys:
                intervals.append({'pipeline': pipeline, 'model': model_name, 'n_OOF_predictions': len(block),
                                  'status': 'incomplete_OOF_no_aggregate_score'})
                continue
            sites = block.site.unique()
            rng = np.random.default_rng(303)
            scores = []
            for _ in range(1000):
                sample = pd.concat([block[block.site.eq(site)] for site in rng.choice(sites, len(sites), replace=True)])
                if sample.y_true.nunique() == 2:
                    scores.append(balanced_accuracy_score(sample.y_true, sample.y_pred))
            low, high = np.percentile(scores, [2.5, 97.5]) if scores else (np.nan, np.nan)
            intervals.append({'pipeline': pipeline, 'model': model_name, 'n_OOF_predictions': len(block),
                              'balanced_accuracy': balanced_accuracy_score(block.y_true, block.y_pred),
                              'conditional_site_bootstrap_CI95_low': low, 'conditional_site_bootstrap_CI95_high': high,
                              'valid_bootstrap_replicates': len(scores),
                              'caveat': 'fixed OOF predictions; does not include all model-training uncertainty'})
    pd.DataFrame(intervals).to_csv(out / 'classification_conditional_intervals.csv', index=False)
    json_save(out / 'classification_scope.json', {
        'training_steps': 'imputation, scaling, feature selection, and tuning nested within site-held-out folds',
        'upstream_QC_mask': '106-node mask selected using QC from the whole eligible cohort before CV',
        'interpretation': 'exploratory prediction conditional on the fixed cohort-wide QC mask; not a fully prospective validation of the entire acquisition/QC workflow',
        'fold_construction': 'deterministic size-balanced site-disjoint K-fold: outer 5, inner 3; both classes required in every split',
        'expected_per_pipeline': expected_counts,
        'complete_pipeline_model_pairs': len(complete_keys),
        'ComBat_used_for_prediction': False})
