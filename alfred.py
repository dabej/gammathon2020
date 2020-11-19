import pandas as pd
#from numpy import loadtxt
#from xgboost import XGBClassifier
from matplotlib import pyplot
import os
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, f1_score, plot_confusion_matrix
import pandas as pd
import random
from xgboost import plot_importance

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def data_handling_preprocessing_1():
    df = pd.read_csv('data/customer_data.csv')
    months = df.month.unique()
    dict = {}
    for i in range(len(months)):
        dict[months[i]] = i
    df['cluster'].replace({'A': 0, 'B': 1, 'C': 2}, inplace=True)
    df['month'].replace(dict, inplace=True)
    df = df[df['month'] > 35]

    customers = df.customer_id.unique()
    customers = random.sample(customers.tolist(), 50000)
    df = df[df['customer_id'].isin(customers)]

    print(len(df.index))

    data = pd.DataFrame()
    for index, row in df.iterrows():
        print(index)
        temp = df[(df['month'] == row['month'] - 1) & (df['customer_id'] == row['customer_id'])]
        temp['target'] = row['churned']
        data = data.append(temp)

    print(data.columns)

    #del data['Unnamed: 0']
    del data['churned']
    data.to_csv('train_sample.csv')

    data2 = df[(df['month'] == 44) & (df['churned'] == False)]
    #del data2['Unnamed: 0']
    del data2['churned']
    data2.to_csv('test_sample.csv')


def create_new_pivot_dataset():
    df = pd.read_csv('data/customer_data.csv')
    df2 = pd.read_csv('data/internet_explorers.csv')

    df = df.append(df2)

    months = df.month.unique()
    dict = {}
    for i in range(len(months)):
        dict[months[i]] = i

    df['cluster'].replace({'A': 0, 'B': 1, 'C': 2}, inplace=True)
    df['month'].replace(dict, inplace=True)
    table = df.pivot_table(columns='month', index='customer_id')

    table.to_csv('new_pivot.csv')


def infer():
    print("infer")
    #rf = RandomForestClassifier()
    #rf.fit(x_train, y_train)
    #print "Features sorted by their score:"

    #sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), x_train), reverse=True)


def shift_values_in_pivoted_dataset():

    dftest = pd.read_csv('data/pivot.csv')

    for i in list(dftest.index):
        dftest.loc[i] = dftest.loc[i].shift(int(sum(dftest.loc[i].isnull()) / 8))

    print(dftest)


def simple_model():
    df = pd.read_csv("data/customer_data.csv")

    df_out = df.groupby('customer_id').tail(2)
    df = df_out.groupby('customer_id').head(1)
    temp = df_out.groupby('customer_id').tail(1)
    target = temp["churned"]

    df = df.drop(columns=["customer_id", "month", "churned", "cluster", "transaction_value", "earned_reward_points"])

    data_train, data_test, churn_train, churn_test = train_test_split(df, target, test_size=0.2, random_state=1)

    cls = MultinomialNB()
    cls.fit(data_train, churn_train)
    prediction = cls.predict(data_test)
    print(prediction)

    classification_report(churn_test, prediction, output_dict=True)


def xgboost_feature_importance():

    use_xgboost = True

    if (use_xgboost):
        # split data into X and y
        X = dataset[dataset.columns[1:-1]]
        y = dataset['churned']

        # fit model no training data
        model = XGBClassifier()
        model.fit(X, y)

        plot_importance(model)

        # feature importance
        print(model.feature_importances_)
        # plot
        pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
        pyplot.show()

#main()
#fr_jakob_2()