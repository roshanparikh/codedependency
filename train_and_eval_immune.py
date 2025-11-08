import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import warnings
warnings.filterwarnings("error", category=RuntimeWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import xgboost
from xgboost import XGBClassifier

from model_architecture import MLPipeline

# Getting data
df_90 = True
if df_90:
       df = pd.read_csv('../data/90_day_mort.csv')
else:
       df = pd.read_csv('../data/30_day_mort.csv')
y = df.copy()['target']
X = df.copy().drop(['target'], axis=1)
X = X[['SIRI', 'Absolute Monocyte Count', 'Absolute Lymphocyte Count', 'Absolute Neutrophil Count']]

# Organization
names_cat_feats = []
names_cont_feats = ['SIRI', 'Absolute Monocyte Count', 'Absolute Lymphocyte Count', 'Absolute Neutrophil Count']


# Parameter grid
random_state = 42; max_iter = max_iter=1000

models_and_params = {
    'Ridge': {'model': LogisticRegression(penalty = 'l2', C=1, solver='lbfgs', random_state=random_state, max_iter=5000, verbose=0),
              'params': {'model__C': np.logspace(-3, 1, 5),
                         'model__class_weight': ['balanced', None]}
    },
    'KNN': {'model': KNeighborsClassifier(),
            'params': {'model__n_neighbors': [1, 2, 3, 5, 7, 10, 15, 30, 50, 70, 100],
                       'model__weights': ['uniform', 'distance'],
                       'model__p': [1,2]} #1 is Manhattan distance, 2 is Euclidean distance 
    },
    'SVC Linear': {'model': SVC(kernel = 'linear', random_state=random_state),
                   'params': {'model__C': np.logspace(-5, 3, 9),
                              'model__class_weight': ['balanced', None]}
    },
    'SVC RBF': {'model': SVC(kernel = 'rbf', random_state=random_state),
                'params': {'model__C': np.logspace(-5, 3, 9),
                           'model__class_weight': ['balanced', None]}
    }, 
    'SVC Poly': {'model': SVC(kernel='poly', random_state=random_state),
                 'params': {'model__C': np.logspace(-4, 3, 8),
                            'model__degree': [2, 3],                     
                            'model__gamma': ['scale', 'auto', 1e-3, 1e-2, 1e-1],
                            'model__coef0': [0.0, 0.1, 1.0, 10.0],      
                            'model__class_weight': ['balanced', None]}
    },
    'XGB': {'model': XGBClassifier(learning_rate = 0.03, n_estimators = 1000, missing=np.nan, subsample=0.66, verbosity=1),
            'params': {'model__max_depth': [1, 3, 10, 30, 100],  # Depth of the tree
                       'model__colsample_bytree': [0.1, 0.25, 0.5, 0.75, 1.0],  # Fraction of features used for fitting trees
                       'model__scale_pos_weight': [8, 10, 12, 15]}
    },
    'MLP': {'model': MLPClassifier(random_state=random_state, max_iter=300, early_stopping=True, n_iter_no_change=10),
            'params': {'model__hidden_layer_sizes': [(64,), (128,), (128, 64), (256, 128)],
                       'model__activation': ['relu', 'tanh'],
                       'model__alpha': np.logspace(-6, -2, 5),        
                       'model__learning_rate_init': [1e-4, 3e-4, 1e-3],
                       'model__batch_size': ['auto', 64, 128],
                       'model__solver': ['adam'],}
    }
}

model_list = ['XGB', 'KNN', 'SVC Linear', 'SVC RBF', 'SVC Poly'] #Ridge and MLP threw a lot of errors

ml = MLPipeline(X=X, y=y, std_ftrs=names_cont_feats, onehot_ftrs=names_cat_feats)
model_results = ml(model_list=model_list, models_and_params=models_and_params)

import pickle
if df_90:
       with open('model_results_immune_90.pkl', 'wb') as f:
              pickle.dump(model_results, f)
else:
     with open('model_results_immune_30.pkl', 'wb') as f:
              pickle.dump(model_results, f)

