import os
from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
control_1 = pd.read_csv('C:\Users\SWAPNIL\Desktop\Datasets\data\condition/control_1.csv')
condition_1 = pd.read_csv('C:\Users\SWAPNIL\Desktop\Datasets\data\condition/condition_1.csv')
print(f'control_1.shape: {control_1.shape}')
print(f'condition_1.shape: {condition_1.shape}')
control_1.head()
condition_1.head()
condition_1.describe()
sns.histplot(x='activity', data=condition_1)
condition_1['activity'].skew()
condition_1['log_activity'] = np.log(condition_1['activity'] + 1) # add + 1 because log(0) is infinity
condition_1['log_activity'].skew()
sns.displot(x='log_activity', data=condition_1, kind='kde', fill=True)
condition_1['sqrt_activity'] = np.sqrt(condition_1['activity'])
condition_1['sqrt_activity'].skew()
sns.displot(x='sqrt_activity', data=condition_1, kind='kde', fill=True)
condition_1.describe()
control_1['activity'].skew()
control_1.describe()
control_1['log_activity'] = np.log(control_1['activity'] + 1)
control_1['log_activity'].skew()
sns.displot(x='log_activity', data=control_1, kind='kde', fill=True)
condition_1.head()
df = condition_1.groupby('date')['log_activity'].mean().reset_index()
df.head()


def combine_data(path):
    dirs = os.listdir(path)
    combine_df = []

    for filepath in dirs:
        source = filepath.split('.')[0]
        if filepath.endswith('.csv'):
            X = pd.read_csv(path + filepath, parse_dates=['timestamp'], index_col='timestamp')
            X['source'] = source
            combine_df.append(X)

    return combine_df
combine_df = combine_data('/kaggle/input/the-depression-dataset/data/condition/')conditions = []
for condition in combine_df:
    condition_df = pd.DataFrame(columns=['mean_activity', 'std_activity', 'zero_activity_proportion', 'source'])
    condition_df['mean_activity'] = condition.activity.resample('H').mean()
    condition_df['std_activity'] = condition.activity.resample('H').std()
    condition_df['zero_activity_proportion'] = [data[1].tolist().count(0) for data in condition.activity.resample('H')]
    condition_df['source'] = condition.source
    conditions.append(condition_df)
    combine_df = combine_data('/kaggle/input/the-depression-dataset/data/control/')
    controls = []
    for control in combine_df:
        control_df = pd.DataFrame(columns=['mean_activity', 'std_activity', 'zero_activity_proportion', 'source'])
        control_df['mean_activity'] = control.activity.resample('H').mean()
        control_df['std_activity'] = control.activity.resample('H').std()
        control_df['zero_activity_proportion'] = [data[1].tolist().count(0) for data in control.activity.resample('H')]
        control_df['source'] = control.source
        controls.append(control_df)
        fig, axes = plt.subplots(23, 1, figsize=(23, 30))
        cnt = 0
        for i in range(23):
            condition = conditions[cnt]
            axes[i].plot(condition.index, condition.mean_activity, color='r')
            axes[i].set_title(f'Mean activity for {condition.source[1]}', fontsize=18)
            cnt += 1

        plt.xlabel('Date', fontsize=14)
        fig.tight_layout(pad=1.0)
        fig.savefig('Mean activity of condition group.jpg', dpi=100)
        plt.show()
        fig, axes = plt.subplots(32, 1, figsize=(23, 40))
        cnt = 0
        for i in range(32):
            control = controls[cnt]
            axes[i].plot(control.index, control.mean_activity, color='g')
            axes[i].set_title(f'Mean activity for {control.source[1]}', fontsize=18)
            cnt += 1

        plt.xlabel('Date', fontsize=14)
        fig.tight_layout(pad=1.0)
        fig.savefig('Mean activity of control group.jpg', dpi=100)
        plt.show()
        # Draw Plot
        fig, axes = plt.subplots(23, 1, figsize=(23, 40))

        cnt = 0
        for i in range(23):
            df = conditions[i].reset_index()

            # Prepare data
            df['hour'] = [d.hour for d in df.timestamp]
            df = df.sort_values('hour')
            df['clock_hour'] = df['hour'].apply(lambda x: to_clock(x))
            sns.boxplot(x='clock_hour', y='mean_activity', data=df, ax=axes[i])
            axes[i].set_title(f'Box Plot of mean activity for {df.source[1]}', fontsize=18)
            cnt += 1

        plt.xlabel('Date', fontsize=14)
        fig.tight_layout(pad=1.0)
        plt.show()# Draw Plot
fig, axes = plt.subplots(32, 1, figsize=(23, 50))

cnt = 0
for i in range(32):
    df = controls[i].reset_index()

    # Prepare data
    df['hour'] = [d.hour for d in df.timestamp]
    df = df.sort_values('hour')
    df['clock_hour'] = df['hour'].apply(lambda x: to_clock(x))
    sns.boxplot(x='clock_hour', y='mean_activity', data=df, ax=axes[i])
    axes[i].set_title(f'Box Plot of mean activity for {df.source[1]}', fontsize=18)
    cnt += 1

plt.xlabel('Date', fontsize=14)
fig.tight_layout(pad=1.0)
plt.show()
fig, axes = plt.subplots(2, 1, figsize=(24, 10))
df = conditions[12].reset_index()
df['hour'] = [d.hour for d in df.timestamp]
df = df.sort_values('hour')
df['clock_hour'] = df['hour'].apply(lambda x: to_clock(x))
sns.boxplot(x='clock_hour', y='zero_activity_proportion', data=df, ax=axes[0])
axes[0].set_title('Zero Activity Count of a Depressed Patient', fontsize=18)

df = controls[2].reset_index()
df['hour'] = [d.hour for d in df.timestamp]
df = df.sort_values('hour')
df['clock_hour'] = df['hour'].apply(lambda x: to_clock(x))
sns.boxplot(x='clock_hour', y='zero_activity_proportion', data=df, ax=axes[1])
axes[1].set_title('Zero Activity Count of a Non-Depressed Patient', fontsize=18)

fig.tight_layout(pad=1.0)
plt.show()
def nextday(dates):
    for date in dates:
        yield date
        def zero_count(series):
    return list(series).count(0)
def extractfeatures(X, date):
    mask = X['date'] == date
    d = {
        'mean_log_activity': X[mask]['log_activity'].mean(),
        'std_log_activity': X[mask]['log_activity'].std(),
        'min_log_activity': X[mask]['log_activity'].min(),
        'max_log_activity': X[mask]['log_activity'].max(),
        'zero_proportion_activity': zero_count(X[mask]['log_activity'])
    }
    return d


class ExtractData(BaseEstimator, TransformerMixin):

    def __init__(self, path):
        self.path = path
        self.X = []

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        dirs = os.listdir(self.path)

        for filepath in sorted(dirs, key=lambda x: x.split('_')[0]):
            condition = filepath.split('.')[0]
            if filepath.endswith('.csv'):
                X = pd.read_csv(self.path + filepath)
                X['log_activity'] = np.log(X['activity'] + 1)
                dates = X.date.unique()

                for date in nextday(dates):
                    d = extractfeatures(X, date)
                    d['source'] = condition
                    self.X.append(d)

        return pd.DataFrame(self.X)

    e = ExtractData(path='/kaggle/input/the-depression-dataset/data/condition/')
    conditions = e.fit_transform(X=None, y=None)
    conditions['state'] = 1
    conditions.tail()
    e = ExtractData(path='/kaggle/input/the-depression-dataset/data/control/')
    controls = e.fit_transform(X=None, y=None)
    controls['state'] = 0
    full_df = controls.append(conditions, ignore_index=True)
    full_df.head()
    full_df.shape
    full_df = full_df.sample(frac=1)  # reshufle the dataset

    def custom_train_test_split(train_set, test_set):
        X_train = train_set.drop('label', axis=1)
        y_train = train_set.label
        X_test = test_set.drop('label', axis=1)
        y_test = test_set.label

        return X_train, X_test, y_train, y_test

    class CustomClassifierCV(BaseEstimator, TransformerMixin):

        def __init__(self, base_clf):
            self.base_clf = base_clf

        def fit(self, X, y=None):
            X['label'] = y
            participants = X.source.unique()
            folds = []

            predictions = []  # predicted labels
            actuals = []  # actual labels

            for p in participants:
                folds.append(X[X['source'] == p])

            for i in range(len(folds)):
                test_set = folds[i]
                train_fold = [elem for idx, elem in enumerate(folds) if idx != i]

                train_set = pd.concat(train_fold)
                X_train, X_test, y_train, y_test = custom_train_test_split(train_set.drop(['source'], axis=1),
                                                                           test_set.drop(['source'], axis=1))

                self.base_clf.fit(X_train, y_train)
                predictions.append(self.predict(X_test))
                actuals.append(test_set.label.iloc[0])

            self.score(predictions, actuals)

        def predict(self, X):
            predictions = self.base_clf.predict(X)
            ones = predictions.tolist().count(1)
            zeroes = predictions.tolist().count(0)

            return 1 if ones > zeroes else 0

        def score(self, predictions, actuals):
            print(classification_report(predictions, actuals))
            X = full_df.drop(['state'], axis=1)
            y = full_df.state
            forest = RandomForestClassifier(n_estimators=100)
            custom_clfCV = CustomClassifierCV(forest)
            custom_clfCV.fit(X, y)