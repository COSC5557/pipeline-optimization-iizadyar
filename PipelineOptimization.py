import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from scipy.stats import randint, uniform
import matplotlib.pyplot as plt

file_path = r"C:\Users\irani\Downloads\ML Dr. Lars\winequality-red.csv"
df = pd.read_csv(file_path, sep=';')
X = df.drop('quality', axis=1)
y = df['quality']
# Split data into training and test sets globally
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# transformations
def log_transform(X):
    return np.log1p(X)


class OptionalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformer, active=True):
        self.transformer = transformer
        self.active = active

    def fit(self, X, y=None):
        if self.active:
            self.transformer.fit(X, y)
        return self

    def transform(self, X):
        if self.active:
            return self.transformer.transform(X)
        return X


numeric_features = X.select_dtypes(include=np.number).columns

numeric_transformer = Pipeline(steps=[
    ('log_transform', OptionalTransformer(FunctionTransformer(log_transform, validate=False))),
    ('scaler', OptionalTransformer(StandardScaler()))
])



preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features)
])


pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('feature_selector', SelectKBest(score_func=f_classif, k=10)),
    ('classifier', None)  # Placeholder for dynamic classifier selection
])


classifiers = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'KNN': KNeighborsClassifier()
}



search_spaces = {
    'RandomForest': {
        'classifier__n_estimators': Integer(100, 1000),
        'classifier__max_depth': Integer(10, 100),
        'classifier__min_samples_leaf': Integer(1, 20),
        'preprocessor__num__log_transform__active': Categorical([True, False]),
        'preprocessor__num__scaler__active': Categorical([True, False]),

    },
    'GradientBoosting': {
        'classifier__n_estimators': Integer(50, 300),
        'classifier__learning_rate': Real(0.01, 0.2),
        'classifier__max_depth': Integer(1, 10),
        'classifier__min_samples_leaf': Integer(1, 10),
        'preprocessor__num__log_transform__active': Categorical([True, False]),
        'preprocessor__num__scaler__active': Categorical([True, False]),

    },
    'KNN': {
        'classifier__n_neighbors': Integer(1, 30),
        'classifier__weights': Categorical(['uniform', 'distance']),
        'classifier__algorithm': Categorical(['auto', 'ball_tree', 'kd_tree', 'brute']),
        'preprocessor__num__log_transform__active': Categorical([True, False]),
        'preprocessor__num__scaler__active': Categorical([True, False]),

    },

}



random_search_spaces = {
    'RandomForest': {
        'classifier__n_estimators': randint(100, 1000),
        'classifier__max_depth': randint(10, 100),
        'classifier__min_samples_leaf': randint(1, 20),
        'preprocessor__num__log_transform__active': [True, False],
        'preprocessor__num__scaler__active': [True, False],

    },

    'GradientBoosting': {
        'classifier__n_estimators': randint(50, 500),
        'classifier__learning_rate': uniform(0.01, 0.2),
        'classifier__max_depth': randint(1, 10),
        'classifier__min_samples_leaf': randint(2, 10),
        'preprocessor__num__log_transform__active': [True, False],
        'preprocessor__num__scaler__active': [True, False],

    },
    'KNN': {
        'classifier__n_neighbors': randint(1, 30),
        'classifier__weights': ['uniform', 'distance'],
        'classifier__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'preprocessor__num__log_transform__active': [True, False],
        'preprocessor__num__scaler__active': [True, False],

    }
}



def evaluate_default_settings(model_name):
    pipeline.set_params(classifier=classifiers[model_name])
    pipeline.fit(X_train, y_train)
    y_test_pred = pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    print(f"{model_name} - Default Test Accuracy: {test_accuracy:.4f}")
    print(f"{model_name} - Cross-validated Accuracy: {np.mean(cv_scores):.4f}")
    return test_accuracy, np.mean(cv_scores)

def optimize_classifier_with_cv(model_name):
    accuracies = {}
    # Random Search
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=random_search_spaces[model_name],
        n_iter=30, 
        cv=5,
        scoring='accuracy',
        random_state=42
    )
    random_search.fit(X_train, y_train)
    random_best_estimator = random_search.best_estimator_
    random_test_accuracy = accuracy_score(y_test, random_best_estimator.predict(X_test))
    random_cv_scores = cross_val_score(random_best_estimator, X_train, y_train, cv=5, scoring='accuracy')
    print(f"{model_name} - Random Search - Best CV Accuracy: {random_search.best_score_:.4f}")
    print(f"{model_name} - Random Search - Test Accuracy: {random_test_accuracy:.4f}")
    print(f"{model_name} - Random Search - Best Params: {random_search.best_params_}")
    accuracies['random_search'] = (random_test_accuracy, np.mean(random_cv_scores))

    # Bayesian Search
    bayes_search = BayesSearchCV(
        estimator=pipeline,
        search_spaces=search_spaces[model_name],
        n_iter=30,
        cv=5,
        scoring='accuracy',
        random_state=42
    )
    bayes_search.fit(X_train, y_train)
    bayes_best_estimator = bayes_search.best_estimator_
    bayes_test_accuracy = accuracy_score(y_test, bayes_best_estimator.predict(X_test))
    bayes_cv_scores = cross_val_score(bayes_best_estimator, X_train, y_train, cv=5, scoring='accuracy')
    print(f"{model_name} - Bayesian Search - Best CV Accuracy: {bayes_search.best_score_:.4f}")
    print(f"{model_name} - Bayesian Search - Test Accuracy: {bayes_test_accuracy:.4f}")
    print(f"{model_name} - Bayesian Search - Best Params: {bayes_search.best_params_}")
    accuracies['bayes_search'] = (bayes_test_accuracy, np.mean(bayes_cv_scores))

    return accuracies


results = {}
for model in classifiers:
    default_test_accuracy, default_cv_accuracy = evaluate_default_settings(model)
    accuracies = optimize_classifier_with_cv(model)
    results[model] = {
        'default': (default_test_accuracy, default_cv_accuracy),
        'random_search': accuracies['random_search'],
        'bayes_search': accuracies['bayes_search']
    }

fig, axs = plt.subplots(len(classifiers), 1, figsize=(8, 6))
for i, model in enumerate(classifiers):
    axs[i].bar(['Default', 'Random Search', 'Bayesian Search'],
               [results[model]['default'][0], results[model]['random_search'][0], results[model]['bayes_search'][0]],
               color='gray', label='Test Accuracy', width=.3)
    axs[i].set_title(f"{model} Test Accuracies")
    axs[i].set_ylim([0.5, 1])
    axs[i].legend()

plt.tight_layout()
plt.show()
