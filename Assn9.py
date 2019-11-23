from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import tree, ensemble
import matplotlib.pyplot as plot
import random


class Evaluation:
    def __init__(self):
        self.cancer = load_breast_cancer()

        self.X_train, self.X_test, self.y_train, self.y_test \
            = train_test_split(self.cancer.data, self.cancer.target, test_size=0.9)

        self.tree_score = 0
        self.bagging_score = {}
        self.boost_score = {}
        self.forest_score = {}

    def decision_tree(self):
        tree1 = tree.DecisionTreeClassifier(criterion='gini').fit(self.X_train, self.y_train)
        # print(tree1.predict(self.X_test))
        self.tree_score = tree1.score(self.X_test, self.y_test)
        print(tree1.score(self.X_test, self.y_test))

    def bagging(self):
        n_estimators = 1
        for i in range(20):
            n_estimators += i
            bagging1 = ensemble.BaggingClassifier(tree.DecisionTreeClassifier(), n_estimators=n_estimators).fit(self.X_train, self.y_train)
            score = bagging1.score(self.X_test, self.y_test)
            self.bagging_score[n_estimators] = score
        self.drawChart(self.bagging_score, "Bagging")

    def forest(self):
        pass

    def boost(self):
        pass

    def summary(self):
        pass

    def drawChart(self, dictionary, chartTitle):
        items = sorted(dictionary.items())
        (x, y) = zip(*items)
        plot.title(chartTitle)
        plot.plot(x, y)
        plot.show()

if __name__ == '__main__':
    exp = Evaluation()
    exp.decision_tree()
    exp.bagging()
    exp.forest()
    exp.boost()
    exp.summary()
