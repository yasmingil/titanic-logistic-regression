# Yasmin Gil
# yasmingi@usc.edu
# Titanic Logistic Regression Predictor
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def AnalyzeTitanic():
    # 1. Read data set into frame
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    titanic_raw = pd.read_csv("titanic.csv")

    # target variable is survived
    # drop useless info
    titanic = titanic_raw.drop(titanic_raw.columns[[2, 5, 6, 7, 8, 9, 10]], axis=1)

    # make sure there are no missing values
    titanic.dropna(axis=0, inplace=True)

    # plot histogram of all variables in 2x2 fig
    fig, ax = plt.subplots(2, 2)

    # survived histogram
    ax[0, 0].hist(titanic['survived'], bins=3, color='gold')
    ax[0, 0].set(xticks=[0,1], xlabel='Survived', ylabel='Count')
    ax[0, 0].set_xticklabels(['no', 'yes'])
    # Pclass histogram
    ax[0, 1].hist(titanic['pclass'], bins=5, color='maroon')
    ax[0, 1].set(xticks=[1, 2, 3], xlabel='Pclass', ylabel='Count')

    # sex histogram
    ax[1, 0].hist(titanic['sex'], bins=3, color='purple')
    ax[1, 0].set(xlabel='Sex', ylabel='Count')

    # Age histogram
    ax[1, 1].hist(titanic['age'], bins=10, color='navy')
    ax[1, 1].set(xlabel='Age', ylabel='Count')

    fig.tight_layout()
    plt.subplots_adjust(top=0.91)
    fig.suptitle('Titanic Data: Histograms of Input Variables')

    # convert all categorical into dummy
    # convert female & male to 1 and 0
    titanic_dummies = pd.get_dummies(titanic, columns=['sex'], prefix='', prefix_sep='')

    X = titanic_dummies.iloc[:, 1:]
    y = titanic_dummies.iloc[:, 0]

    # fit into log reg model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)
    titanic_log_reg = LogisticRegression(max_iter=1000)
    titanic_log_reg.fit(X_train, y_train)

    # calculate accuracy of predictions of model
    accuracy = titanic_log_reg.score(X_test, y_test)
    print('Accuracy:', accuracy)

    # plot confusion matrix
    labels = ['no', 'yes']
    metrics.plot_confusion_matrix(titanic_log_reg, X_test, y_test, display_labels=labels)
    plt.title('Titanic Dataset Survivability\n(Model accuracy: '+ str(accuracy * 100) + '%)')

    # print survivability of 3rd class, 33, male
    print('survived?: ', titanic_log_reg.predict([[3, 33.0, 0, 1]]))
    print(titanic_raw.head())
    # plt.show()

def main():
    AnalyzeTitanic()

if __name__ == '__main__':
    main()


