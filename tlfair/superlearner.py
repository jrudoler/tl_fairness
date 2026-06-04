import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

#adapted from https://machinelearningmastery.com/super-learner-ensemble-in-python/
class SuperLearnerClassifier():
    def __init__(self, models=None, folds=10, random_state=None) -> None:
        if models is None:
            self.models = [
                make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                HistGradientBoostingClassifier(random_state=random_state),
                make_pipeline(StandardScaler(), KNeighborsClassifier()),
                CalibratedClassifierCV(
                    make_pipeline(StandardScaler(), SVC()),
                    cv=3,
                    ensemble=False
                ),
                RandomForestClassifier(random_state=random_state, n_jobs=-1),
                make_pipeline(
                    StandardScaler(),
                    MLPClassifier(
                        random_state=random_state,
                        max_iter=1000,
                        early_stopping=True
                    )
                )
            ]
        else:
            self.models = models
        self.folds = folds
        self.random_state = random_state
        pass
    
    def fit_base_models(self, x, y):
        for m in self.models:
            m.fit(x,y)

    def _aligned_predict_proba(self, model, x):
        probs = model.predict_proba(x)
        aligned = np.zeros((len(x), len(self.classes_)))
        model_classes = model.classes_
        for model_idx, model_class in enumerate(model_classes):
            class_idx = np.where(self.classes_ == model_class)[0][0]
            aligned[:, class_idx] = probs[:, model_idx]
        return aligned
    
    def kfold_predictions(self, x, y):
        meta_x = []
        meta_y = []
        _, counts = np.unique(y, return_counts=True)
        n_splits = min(self.folds, counts.min())
        if n_splits < 2:
            raise ValueError("SuperLearnerClassifier needs at least two samples per class.")
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        for tr_idx, te_idx in kfold.split(x, y):
            fold_ypreds = []
            xtr, xte = np.array(x)[tr_idx], np.array(x)[te_idx]
            ytr, yte = np.array(y)[tr_idx], np.array(y)[te_idx]
            meta_y.extend(yte)

            for m in self.models:
                m.fit(xtr, ytr)
                ypreds = self._aligned_predict_proba(m, xte)
                fold_ypreds.append(ypreds)
            
            meta_x.append(np.hstack(fold_ypreds))
        return np.vstack(meta_x), np.array(meta_y)
            

    def fit(self, x, y):
        self.classes_ = np.unique(y)
        meta_x, meta_y = self.kfold_predictions(x, y)
        self.lm = LogisticRegression(max_iter=2000).fit(meta_x, meta_y)
        self.fit_base_models(x,y)
        return self

    def predict(self, x):
        meta_x = []
        for m in self.models:
            ypreds = self._aligned_predict_proba(m, x)
            meta_x.append(ypreds)
        meta_x = np.hstack(meta_x)
        return self.lm.predict(meta_x)
    
    def predict_proba(self, x):
        meta_x = []
        for m in self.models:
            ypreds = self._aligned_predict_proba(m, x)
            meta_x.append(ypreds)
        meta_x = np.hstack(meta_x)
        return self.lm.predict_proba(meta_x)
    
