from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import tree, ensemble
import matplotlib.pyplot as plot
import random
from mpl_toolkits.mplot3d import Axes3D


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
        for i in range(40):
            n_estimators = i + 1
            bagging1 = ensemble.BaggingClassifier(tree.DecisionTreeClassifier(), n_estimators).fit(self.X_train, self.y_train)
            score = bagging1.score(self.X_test, self.y_test)
            self.bagging_score[n_estimators] = score
        self.drawChart(self.bagging_score, "Bagging")

    def forest(self):
        for i in range(10):
            n_estimators = i + 1
            for j in range(10):
                max_features = j * 2 + 1
                forest1 = ensemble.RandomForestClassifier(n_estimators, max_features=max_features).fit(self.X_train, self.y_train)
                score = forest1.score(self.X_test, self.y_test)
                self.forest_score.update({(n_estimators, max_features): score})

        (xy, scores) = zip(*self.forest_score.items())
        (x, y) = zip(*xy)
        fig = plot.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel("N Estimators")
        ax.set_ylabel("Max Features")
        ax.set_zlabel("Scores")
        ax.scatter(x, y, scores)
        plot.title("Forest")
        plot.show()

    def boost(self):
        for i in range(40):
            n_estimators = i + 1
            ada_boost = ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(), n_estimators).fit(self.X_train, self.y_train)
            score = ada_boost.score(self.X_test, self.y_test)
            self.boost_score[n_estimators] = score
        self.drawChart(self.boost_score, "AdaBoost")

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
