"""Training-only redundancy pruning and ranked feature selection for exploratory ML."""
import inspect
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_array, check_is_fitted

class RedundancyPruner(TransformerMixin, BaseEstimator):
    def __init__(self, threshold=1-1e-10):self.threshold=threshold
    def fit(self,X,y=None):
        X=check_array(X);self.n_features_in_=X.shape[1]
        valid=np.flatnonzero(np.std(X,axis=0)>1e-12)
        if not len(valid):raise ValueError('No nonconstant training feature')
        correlation=np.atleast_2d(np.corrcoef(X[:,valid],rowvar=False)) if len(valid)>1 else np.ones((1,1))
        kept=[]
        for i,index in enumerate(valid):
            if not any(abs(correlation[i,j])>=self.threshold for j in kept):kept.append(i)
        self.support_=np.zeros(self.n_features_in_,dtype=bool);self.support_[valid[kept]]=True
        return self
    def transform(self,X):
        check_is_fitted(self,'support_');X=check_array(X)
        if X.shape[1]!=self.n_features_in_:raise ValueError('Feature schema differs from training')
        return X[:,self.support_]
    def get_support(self):check_is_fitted(self,'support_');return self.support_.copy()

class RankedSelector(TransformerMixin, BaseEstimator):
    def __init__(self, method='all',k=20,random_state=101):
        self.method=method;self.k=k;self.random_state=random_state
    def fit(self,X,y):
        X=check_array(X);self.n_features_in_=X.shape[1]
        self.zero_coefficient_selected_=0
        if self.method=='all':
            self.scores_=np.zeros(self.n_features_in_);self.support_=np.ones(self.n_features_in_,dtype=bool);return self
        fscore=np.nan_to_num(f_classif(X,y)[0],nan=-np.inf,posinf=np.finfo(float).max)
        if self.method=='anova':scores=fscore
        elif self.method=='mutual_info':scores=mutual_info_classif(X,y,n_neighbors=3,random_state=self.random_state)
        elif self.method=='l1':
            args={'C':1.,'solver':'liblinear','class_weight':'balanced','max_iter':5000,'random_state':self.random_state}
            # Current sklearn uses l1_ratio; older versions use penalty for liblinear.
            default=inspect.signature(LogisticRegression).parameters['penalty'].default
            args.update({'l1_ratio':1.} if default=='deprecated' else {'penalty':'l1'})
            self.estimator_=LogisticRegression(**args).fit(X,y)
            scores=np.abs(self.estimator_.coef_[0])
        else:raise ValueError(self.method)
        self.scores_=np.asarray(scores)
        order=np.lexsort((np.arange(self.n_features_in_),-fscore,-self.scores_))
        self.support_=np.zeros(self.n_features_in_,dtype=bool)
        self.support_[order[:min(int(self.k),self.n_features_in_)]]=True
        if self.method=='l1':self.zero_coefficient_selected_=int(np.sum((self.scores_<=1e-12)&self.support_))
        return self
    def transform(self,X):
        check_is_fitted(self,'support_');X=check_array(X)
        if X.shape[1]!=self.n_features_in_:raise ValueError('Feature schema differs from training')
        return X[:,self.support_]
    def get_support(self):check_is_fitted(self,'support_');return self.support_.copy()
