import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, accuracy_score, f1_score, precision_score, recall_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

import warnings
warnings.filterwarnings("error", category=RuntimeWarning)

class MLPipeline:
    '''
    Class that performs kfold cross validation, preprocressing of cleaned data, and gridsearch hyperparameter tuning.
    '''
    def __init__(self, X, y, std_ftrs, onehot_ftrs):
        ''' 
        @params:
            X (pandas df): feature matrix
            y (pandas df): target column
            std_ftrs (list): strings of all continuous features before scaling
            onehot_ftrs (list): strings of all categorical features before saling
        @returns: None
        '''
        self.X = X
        self.y = y
        self.std_ftrs = std_ftrs
        self.onehot_ftrs = onehot_ftrs

        # Categorical scaler
        self.one_hot_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant',fill_value='missing')),
            ('onehot', OneHotEncoder(sparse_output=False,handle_unknown='ignore'))])

        # Standard scaler 
        self.std_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())])
        
        # Iterative imputer
        self.iterative_imputer = IterativeImputer(
            estimator=RandomForestRegressor(
                n_estimators=10,         
                max_depth=5,             
                random_state=42
            ),
            max_iter=10,
            initial_strategy="median",
            random_state=42,
            skip_complete=True          
        )

        # Collect the transformers
        self.preprocessor_xgb = ColumnTransformer(
            transformers=[
                ('std', self.std_transformer, self.std_ftrs),
                ('ohot', self.one_hot_transformer, self.onehot_ftrs)
            ]
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('ohot', self.one_hot_transformer, self.onehot_ftrs),
                ('std', Pipeline(steps=[
                    ('imputer', self.iterative_imputer),
                    ('scaler', StandardScaler())
                ]), self.std_ftrs),
            ]
        )

    def grid_pipeline(self, ML_algo, param_grid):
        ''' 
        Builds the gridsearch pipeline.
        @params:
            ML_algo (str): Specifies which algorithm will be used
            param_grid (dict): Parameters for GridSearchCV for a single model type
        @returns:
            results (dict): metrics, test sets, and best models per validation fold
        '''
        results = {}

        # Split Data
        X_other, X_test, y_other, y_test = train_test_split(self.X, self.y, train_size=0.8, random_state = 42, stratify=self.y)

        # Baselines
        majority_class = np.bincount(y_other).argmax()
        y_base = np.full(shape=len(y_test), fill_value=majority_class)
        base_acc = accuracy_score(y_test, y_base)
        base_f1  = f1_score(y_test, y_base, zero_division=0)

        if ML_algo != 'XGB':
            pipe = Pipeline([
                ('preprocessor', self.preprocessor),
                ('model', ML_algo)
            ])
        else:
            pipe = Pipeline([
                ('preprocessor', self.preprocessor_xgb),
                ('model', ML_algo)
            ])
    
        # CV and prepro
        print(f"\n>>>Running {ML_algo}")
        grid = GridSearchCV(pipe, param_grid=param_grid,scoring = make_scorer(f1_score),
                                cv=None, return_train_score = True, n_jobs=1, verbose=1)

        # Sanity check
        X_prep = self.preprocessor.fit_transform(X_other)

        assert not np.isnan(X_prep).any(), "NaNs found after preprocessing!"
        assert not np.isinf(X_prep).any(), "Infs found after preprocessing!"
        assert np.isfinite(X_prep).all(), "Non-finite values found!"

        grid.fit(X_other, y_other)
        y_pred = grid.best_estimator_.predict(X_test)

        transformer = grid.best_estimator_['columntransformer']
        X_test_prep = transformer.transform(X_test)
        X_test_prep_df = pd.DataFrame(X_test_prep, columns = transformer.get_feature_names_out())

        mod_acc = accuracy_score(y_test, y_pred)
        mod_f1  = f1_score(y_test, y_pred, zero_division=0)

        results = {
            'X_test_raw': X_test,
            'X_test_preprocessed': X_test_prep_df,
            'y_test': y_test,
            'y_pred': y_pred,
            'metrics': {
                'accuracy': mod_acc,
                'f1': mod_f1,
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'rel_impr_accuracy': (mod_acc - base_acc)/max(base_acc, 1e-12),
                'rel_impr_f1': (mod_f1 - base_f1)/max(base_f1, 1e-12),
            },
            'best_model': grid.best_estimator_,
            'best_params': grid.best_params_,
            'baseline': {
                'accuracy': base_acc,
                'f1': base_f1,
            }
        }

        return results

    def __call__(self, model_list, models_and_params):
        '''
        Performs grid search over all models.
        @params: 
            model_list (list): list of strings with models to be trained
            models_and_params (dict): full dictionary of all models to be trains with hyperparameters to adjust
        @returns:
            model_results (dict): full dictionary of data sets, best metrics, and best trained models for each model type
        '''
        model_results = {}

        for model in model_list:
            ML_algo = models_and_params[model]['model']
            params = models_and_params[model]['params']
            
            results = self.grid_pipeline(ML_algo=ML_algo, param_grid=params)
            
            print(f"Results for {model}:")
            print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
            print(f"Accuracy improvement over baseline: {results['metrics']['rel_impr_accuracy']:.2%}")
            print(f"F1 Score: {results['metrics']['f1']:.4f}")
            print(f"F1 improvement over baseline: {results['metrics']['rel_impr_f1']:.2%}")

            model_results[model] = results

        return model_results
        

