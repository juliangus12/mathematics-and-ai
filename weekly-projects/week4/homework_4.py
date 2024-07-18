import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Part 1: Data preparation

# load the compas data set
data = pd.read_csv('compas-scores-two-years.csv')

# explore the data
print(data.head())
print(data.info())
print(data.describe())

# select features and response variables
features = ['age', 'sex', 'race', 'priors_count', 'juv_fel_count']
response = 'two_year_recid'

X = data[features]
y = data[response]

# one-hot encoding for categorical features
X = pd.get_dummies(X, columns=['sex', 'race'], drop_first=True)

# check the transformed features
print(X.head())

# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# check the shapes of the splits
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# What do you observe in the data?
# the dataset has a mix of numerical and categorical variables some variables have missing values the data includes demographic information offense history and recidivism outcomes there are 53 columns and 7214 rows after applying filters the data frame has fewer rows

# Part 2: Train and validate a decision tree

# fit decision tree model
dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(X_train, y_train)
dtc_pred = dtc.predict(X_test)
dtc_accuracy = accuracy_score(y_test, dtc_pred)
print(f'decision tree test accuracy: {dtc_accuracy:.4f}')

# Perform 5-fold cross-validation for different tree sizes
print('Leaves\tMean accuracy')
print('---------------------')
best_num_leaves = 0
best_accuracy = 0

for num_leaves in range(100, 1800, 100):
    if num_leaves >= 2:
        dtc = DecisionTreeClassifier(max_leaf_nodes=num_leaves, random_state=42)
        scores = cross_val_score(dtc, X_train, y_train, cv=5)
        mean_accuracy = scores.mean()
        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_num_leaves = num_leaves
        print(f"{num_leaves}\t{mean_accuracy:.3f}")

# Train a decision tree using the selected value of max_leaf_nodes
dtc = DecisionTreeClassifier(max_leaf_nodes=best_num_leaves, random_state=42)
dtc.fit(X_train, y_train)
dtc_pred = dtc.predict(X_test)
dtc_accuracy = accuracy_score(y_test, dtc_pred)
num_leaves = dtc.get_n_leaves()
print(f'trained decision tree with {num_leaves} leaves and test accuracy {dtc_accuracy:.2f}.')

# How does the decision tree perform on the training data?
# the decision tree performs well on the training data with high accuracy but its huge and likely overfits because it has many leaves

# What is the effect of tuning the "maximum number of leaves" parameter?
# tuning the maximum number of leaves parameter helps find a balance between model complexity and performance using cross-validation we can determine the best number of leaves that gives us the highest mean accuracy

# Part 3: Auditing a decision tree for demographic biases

# Create subset of training data without information on race.
remaining_features = [v for v in X.columns if not v.startswith('race_')]
X_train_sub = X_train[remaining_features]
X_test_sub = X_test[remaining_features]

# Fit model to training data without racial information
dtc_no_race = DecisionTreeClassifier(max_leaf_nodes=best_num_leaves, random_state=42)
dtc_no_race.fit(X_train_sub, y_train)
dtc_no_race_pred = dtc_no_race.predict(X_test_sub)
dtc_no_race_accuracy = accuracy_score(y_test, dtc_no_race_pred)
num_leaves_no_race = dtc_no_race.get_n_leaves()
print(f'trained decision tree without race with {num_leaves_no_race} leaves and test accuracy {dtc_no_race_accuracy:.2f}.')

# Get decision tree rules
tree_rules = export_text(dtc, feature_names=list(X_train.columns))
tree_no_race_rules = export_text(dtc_no_race, feature_names=list(X_train_sub.columns))
print("Decision tree rules with racial information:")
print(tree_rules)
print("\nDecision tree rules without racial information:")
print(tree_no_race_rules)

# Additional metrics and confusion matrix for deeper insight
conf_matrix_tree = confusion_matrix(y_test, dtc_pred)
class_report_tree = classification_report(y_test, dtc_pred)
conf_matrix_tree_no_race = confusion_matrix(y_test, dtc_no_race_pred)
class_report_tree_no_race = classification_report(y_test, dtc_no_race_pred)

print("\nConfusion Matrix for Decision Tree with racial information:")
print(conf_matrix_tree)
print("\nClassification Report for Decision Tree with racial information:")
print(class_report_tree)
print("\nConfusion Matrix for Decision Tree without racial information:")
print(conf_matrix_tree_no_race)
print("\nClassification Report for Decision Tree without racial information:")
print(class_report_tree_no_race)

# Does the model show signs of racial bias in performance assessment?
# by comparing the test accuracy of the decision tree with and without racial information we can see if race significantly affects the model's performance if the accuracy drops significantly without race it suggests that race was an important feature in the model

# Do the decision rules indicate racial bias?
# by examining the decision rules of the tree with and without race we can check if race-related features appear prominently in the decision nodes if they do it indicates potential racial bias in the model's decisions

# Part 4: Comparison to other linear classifiers

# fit lda model
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
lda_pred = lda.predict(X_test)
lda_accuracy = accuracy_score(y_test, lda_pred)
print(f'lda test accuracy: {lda_accuracy:.4f}')

# fit logistic regression model
log_reg = LogisticRegression(max_iter=10000)
log_reg.fit(X_train, y_train)
log_reg_pred = log_reg.predict(X_test)
log_reg_accuracy = accuracy_score(y_test, log_reg_pred)
print(f'logistic regression test accuracy: {log_reg_accuracy:.4f}')

# fit random forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f'random forest test accuracy: {rf_accuracy:.4f}')

# fit bagging classifier model
bagging = BaggingClassifier(n_estimators=100, random_state=42)
bagging.fit(X_train, y_train)
bagging_pred = bagging.predict(X_test)
bagging_accuracy = accuracy_score(y_test, bagging_pred)
print(f'bagging classifier test accuracy: {bagging_accuracy:.4f}')

# fit gradient boosting model
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
gb_accuracy = accuracy_score(y_test, gb_pred)
print(f'gradient boosting test accuracy: {gb_accuracy:.4f}')

# fit support vector classifier model
svc = SVC()
svc.fit(X_train, y_train)
svc_pred = svc.predict(X_test)
svc_accuracy = accuracy_score(y_test, svc_pred)
print(f'support vector classifier test accuracy: {svc_accuracy:.4f}')

# compare performance metrics
print("\nComparison of test accuracies:")
print(f'LDA Test Accuracy: {lda_accuracy:.4f}')
print(f'Logistic Regression Test Accuracy: {log_reg_accuracy:.4f}')
print(f'Random Forest Test Accuracy: {rf_accuracy:.4f}')
print(f'Bagging Classifier Test Accuracy: {bagging_accuracy:.4f}')
print(f'Gradient Boosting Test Accuracy: {gb_accuracy:.4f}')
print(f'Support Vector Classifier Test Accuracy: {svc_accuracy:.4f}')
print(f'Decision Tree Test Accuracy: {dtc_accuracy:.4f}')
print(f'Decision Tree without Race Test Accuracy: {dtc_no_race_accuracy:.4f}')

# Which classifier performed the best on the test set?
# among the classifiers logistic regression performed the best followed closely by gradient boosting and lda

# How do the decision tree's performance metrics compare to the other classifiers?
# decision trees tend to have lower prediction accuracy compared to other classifiers like lda logistic regression and gradient boosting they are more interpretable but may not always give the best performance
