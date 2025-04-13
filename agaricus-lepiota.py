'''
Kevin Ke
IDSN 542, Fall 2024
kevinke@usc.edu
Final Project Part 3
'''

import pandas as pd
from pandas.core.common import random_state
from sklearn.linear_model import LogisticRegression, SGDClassifier
# import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_curve, accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import validation_curve
import xgboost as xgb
import numpy as np
from xgboost import XGBClassifier
from xgboost.core import XGBoostError
from sklearn.model_selection import GridSearchCV
from neuralnetmlp import *

def minibatch_generator(X, y, minibatch_size):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    for start_idx in range(0, indices.shape[0] - minibatch_size
                              + 1, minibatch_size):
        batch_idx = indices[start_idx:start_idx + minibatch_size]

        yield X[batch_idx], y[batch_idx]


def compute_mse_and_acc(nnet, X, y, num_labels=2, minibatch_size=100):
    mse, correct_pred, num_examples = 0., 0, 0
    minibatch_gen = minibatch_generator(X, y, minibatch_size)

    for i, (features, targets) in enumerate(minibatch_gen):
        _, probas = nnet.forward(features)
        predicted_labels = np.argmax(probas, axis=1)

        onehot_targets = int_to_onehot(targets, num_labels=num_labels)
        loss = np.mean((onehot_targets - probas) ** 2)
        correct_pred += (predicted_labels == targets).sum()

        num_examples += targets.shape[0]
        mse += loss

    mse = mse / (i + 1)
    acc = correct_pred / num_examples
    return mse, acc


def train(model, X_train, y_train, X_valid, y_valid, num_epochs,
          learning_rate=0.1):

    for e in range(num_epochs):

        # iterate over minibatches
        minibatch_gen = minibatch_generator(
            X_train, y_train, 100)

        for X_train_mini, y_train_mini in minibatch_gen:
            #### Compute outputs ####
            a_h, a_out = model.forward(X_train_mini)

            #### Compute gradients ####
            d_loss__d_w_out, d_loss__d_b_out, d_loss__d_w_h, d_loss__d_b_h = model.backward(X_train_mini, a_h, a_out, y_train_mini)

            #### Update weights ####
            model.weight_h -= learning_rate * d_loss__d_w_h
            model.bias_h -= learning_rate * d_loss__d_b_h
            model.weight_out -= learning_rate * d_loss__d_w_out
            model.bias_out -= learning_rate * d_loss__d_b_out


# Function: main
# Purpose: Prints data's info, corr matrix, and model scores
# Parameters: None
# Side effects: None
# Returns: Nothing
def main():
    # Load Data
    da = pd.read_csv("data.csv")
    print('\n------------------\nData Info\n------------------\n')
    print(da.info())

    # Clean Data
    da = da.dropna(subset=["stalk-root"])
    le = LabelEncoder()
    for column in da.columns:
        da[column] = le.fit_transform(da[column])

    X = da.drop('poisonous', axis=1)
    y = da['poisonous']
    corr_matrix = da.corr(numeric_only=True)
    print('\n------------------\nCorrelation Matrix\n------------------\n')
    print(corr_matrix["poisonous"].sort_values(ascending=False))

    # Create training and test sets
    print('\n------------------\nPlease wait for results...\n------------------\n')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42, stratify=y)

    # Tuning parameters and creating pipelines
    classifier = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=10000)
    param_grid = {'C': [1, 2, 3, 4, 5]}
    gs = GridSearchCV(estimator=classifier, param_grid=param_grid, scoring='accuracy', refit=True, cv=3)
    gs.fit(X_train, y_train)
    lr_pipe = make_pipeline(StandardScaler(), gs.best_estimator_)

    classifier = KNeighborsClassifier()
    param_grid = {'n_neighbors': [1, 2, 3, 4, 5], 'p': [1, 2, 3, 4, 5]}
    gs = GridSearchCV(estimator=classifier, param_grid=param_grid, scoring='accuracy', refit=True, cv=3)
    gs.fit(X_train, y_train)
    kn_pipe = make_pipeline(StandardScaler(), gs.best_estimator_)

    classifier = SVC(kernel = 'poly')
    param_grid = {'C': [1, 2, 3, 4, 5], 'degree': [1, 2, 3, 4, 5]}
    gs = GridSearchCV(estimator=classifier, param_grid=param_grid, scoring='accuracy', refit=True, cv=3)
    gs.fit(X_train, y_train)
    sv_pipe = make_pipeline(StandardScaler(), gs.best_estimator_)

    classifier = RandomForestClassifier()
    param_grid = {'n_estimators': [10, 20, 30, 40, 50], 'max_depth': [10, 20, 30, 40, 50]}
    gs = GridSearchCV(estimator=classifier, param_grid=param_grid, scoring='accuracy', refit=True, cv=3)
    gs.fit(X_train, y_train)
    rf_pipe = make_pipeline(StandardScaler(), gs.best_estimator_)

    classifier = GradientBoostingClassifier(criterion = 'squared_error')
    param_grid = {'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5], 'max_depth': [10, 20, 30, 40, 50]}
    gs = GridSearchCV(estimator=classifier, param_grid=param_grid, scoring='accuracy', refit=True, cv=3)
    gs.fit(X_train, y_train)
    gb_pipe = make_pipeline(StandardScaler(), gs.best_estimator_)

    classifier = AdaBoostClassifier(random_state = 42)
    param_grid = {'learning_rate': [1, 2, 3, 4, 5], 'n_estimators': [10, 20, 30, 40, 50]}
    gs = GridSearchCV(estimator=classifier, param_grid=param_grid, scoring='accuracy', refit=True, cv=3)
    gs.fit(X_train, y_train)
    ad_pipe = make_pipeline(StandardScaler(), gs.best_estimator_)

    # 10-fold cross validation
    labels = ['Logistic Regression', 'K-Nearest Neighbor', 'Support Vector', 'Random Forest', 'Gradient Boosting', 'Ada Boost']
    for classifier, label in zip([lr_pipe, kn_pipe, sv_pipe, rf_pipe, gb_pipe, ad_pipe], labels):
        scores = cross_val_score(estimator=classifier, X=X_test, y=y_test, cv=10, scoring='accuracy')
        print(f'{label} ACC: {scores.mean() * 100 : .2f} '
              f'(+/- {scores.std():.2f})')

    # Converting X and y into numpy arrays
    columns = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size',
               'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
               'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type',
               'spore-print-color', 'population', 'habitat']

    X = np.zeros((5644, 22), dtype=int)
    y = np.zeros((5644,), dtype=int)

    for i in range(da.shape[0]):
        temp = []
        for col in columns:
            temp.append(da[col][i])
        vector = np.array(temp)
        X[i] = vector
        y[i] = da['poisonous'][i]

    # Split into training, validation, and test set:
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=3900, random_state=123, stratify=y)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp, y_temp, test_size=600, random_state=123, stratify=y_temp)

    # optional to free up some memory by deleting non-used arrays:
    del X_temp, y_temp

    model = NeuralNetMLP(num_features=22,
                         num_hidden=50,
                         num_classes=2)

    num_epochs = 300

    np.random.seed(123)  # for the training set shuffling

    train(model, X_train, y_train, X_valid, y_valid, num_epochs, learning_rate=0.4)

    test_mse, test_acc = compute_mse_and_acc(model, X_test, y_test)
    print(f'Multi-layered Perceptron ACC: {test_acc * 100 : .2f} '
          f'(+/- {test_mse:.2f})')

main()