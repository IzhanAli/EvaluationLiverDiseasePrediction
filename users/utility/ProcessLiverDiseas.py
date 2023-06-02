# Ignore Warnings
import warnings

# for numerical computing
import numpy as np
# for dataframes
import pandas as pd
# for easier visualization
import seaborn as sns
from django.conf import settings
# for visualization and to display plots
from matplotlib import pyplot as plt

# import color maps

warnings.filterwarnings("ignore")

# to split train and test set

# to perform hyperparameter tuning
from sklearn.model_selection import GridSearchCV

# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix,accuracy_score, precision_score

from sklearn.model_selection import train_test_split
# import xgboost
import os

path = os.path.join(settings.MEDIA_ROOT, 'indian_liver_patient.csv')
df = pd.read_csv(path)


## if score==negative, mark 0 ;else 1
def partition(x):
    if x == 2:
        return 0
    return 1


df['Dataset'] = df['Dataset'].map(partition)

df.describe(include=['object'])
df[df['Gender'] == 'Male'][['Dataset', 'Gender']].head()


## if score==negative, mark 0 ;else 1
def partition(x):
    if x == 'Male':
        return 0
    return 1


df['Gender'] = df['Gender'].map(partition)
sns.set_style('whitegrid')  ## Background Grid
sns.FacetGrid(df, hue='Dataset').map(plt.scatter, 'Total_Bilirubin', 'Direct_Bilirubin').add_legend()
# plt.show()

## Data Cleaning
df = df.drop_duplicates()
print(df.shape)
df.Aspartate_Aminotransferase.sort_values(ascending=False).head()
df = df[df.Aspartate_Aminotransferase <= 3000]
df.shape
df.Aspartate_Aminotransferase.sort_values(ascending=False).head()
df = df[df.Aspartate_Aminotransferase <= 2500]
df.shape
df.isnull().values.any()
df = df.dropna(how='any')
# Create separate object for target variable
y = df.Dataset

# Create separate object for input features
X = df.drop('Dataset', axis=1)

# Split X and y into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=1234,
                                                    stratify=df.Dataset)
# Print number of observations in X_train, X_test, y_train, and y_test
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
train_mean = X_train.mean()
train_std = X_train.std()
## Standardize the train data set
X_train = (X_train - train_mean) / train_std
## Check for mean and std dev.
X_train.describe()
## Note: We use train_mean and train_std_dev to standardize test data set
X_test = (X_test - train_mean) / train_std
## Check for mean and std dev. - not exactly 0 and 1
X_test.describe()


def start_logistic_regression():
    tuned_params = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], 'penalty': ['l1', 'l2']}
    model = GridSearchCV(LogisticRegression(), tuned_params, scoring='roc_auc', n_jobs=-1)
    model.fit(X_train, y_train)
    model.best_estimator_
    ## Predict Train set results
    y_train_pred = model.predict(X_train)
    ## Predict Test set results
    y_pred = model.predict(X_test)
    # Get just the prediction for the positive class (1)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    # Display first 10 predictions
    y_pred_proba[:10]
    i = 28  ## Change the value of i to get the details of any point (56, 213, etc.)
    print('For test point {}, actual class = {}, precited class = {}, predicted probability = {}'.
          format(i, y_test.iloc[i], y_pred[i], y_pred_proba[i]))
    confusion_matrix(y_test, y_pred).T
    # Calculate ROC curve from y_test and pred
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    # Plot the ROC curve
    fig = plt.figure(figsize=(8, 8))
    plt.title('Receiver Operating Characteristic')

    # Plot ROC curve
    plt.plot(fpr, tpr, label='l1')
    plt.legend(loc='lower right')

    # Diagonal 45 degree line
    plt.plot([0, 1], [0, 1], 'k--')

    # Axes limits and labels
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    # plt.show()
    # Calculate AUC for Train set
    roc = roc_auc_score(y_train, y_train_pred)
    # Calculate AUC for Test set
    print(auc(fpr, tpr))
    ## Building the model again with the best hyperparameters
    model = LogisticRegression(C=1, penalty='l2')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    lg_accuracy = accuracy_score(y_pred, y_test)
    lg_precision = precision_score(y_pred, y_test)
    cm1 = confusion_matrix(y_pred, y_test)
    lg_specificity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    lg_sensitivity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    # indices = np.argsort(-abs(model.coef_[0, :]))
    # print("The features in order of importance are:")
    # print(50 * '-')
    # # for feature in X.columns[indices]:
    # #     print(feature)
    return lg_accuracy, lg_precision,lg_sensitivity, lg_specificity


def start_svm():
    from sklearn import svm
    def svc_param_selection(X, y, nfolds):
        Cs = [0.001, 0.01, 0.1, 1, 10]
        gammas = [0.001, 0.01, 0.1, 1]
        param_grid = {'C': Cs, 'gamma': gammas}
        grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
        grid_search.fit(X_train, y_train)
        grid_search.best_params_
        return grid_search.best_params_

    # svClassifier = SVC(kernel='rbf')
    # svClassifier.fit(X_train, y_train)
    # svc_param_selection(X_train, y_train, 5)
    ###### Building the model again with the best hyperparameters
    model = SVC(C=1, gamma=1)
    model.fit(X_train, y_train)
    ## Predict Train results
    y_train_pred = model.predict(X_train)
    ## Predict Test results
    y_pred = model.predict(X_test)
    confusion_matrix(y_test, y_pred).T
    #y_pred_proba = model.predict_proba(X_test)[:, 1]
    # Calculate ROC curve from y_test and pred
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    # Plot the ROC curve
    fig = plt.figure(figsize=(8, 8))
    plt.title('Receiver Operating Characteristic')

    # Plot ROC curve
    plt.plot(fpr, tpr, label='l1')
    plt.legend(loc='lower right')

    # Diagonal 45 degree line
    plt.plot([0, 1], [0, 1], 'k--')

    # Axes limits and labels
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    # plt.show()
    # Calculate AUC for Train
    roc_auc_score(y_train, y_train_pred)
    print(auc(fpr, tpr))
    y_pred = model.predict(X_test)
    svm_accuracy = accuracy_score(y_pred, y_test)
    svm_precision = precision_score(y_pred, y_test)
    cm1 = confusion_matrix(y_pred, y_test)
    svm_specificity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    svm_sensitivity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    return svm_accuracy, svm_precision, svm_sensitivity, svm_specificity