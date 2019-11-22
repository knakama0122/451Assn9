from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import tree, ensemble


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
        t1 = tree.DecisionTreeClassifier(criterion="gini").fit(self.X_train, self.y_train)
        self.tree_score = t1.score(self.X_test, self.y_test)

    def bagging(self):
        for int i * 10 in range(20)
        bagging = ensemble.BaggingClassifier(tree.DecisionTreeClassifier(),
                                             n_estimators=i).fit(self.X_train, self.y_train)
        self.bagging_score = bagging.score(self.X_test, self.y_test)
        # create a chart use map to do key for # of estimator, and the value to be score

    # def forest(self):


    # def boost(self):
    


    # def summary(self):


if __name__ == '__main__':
    exp = Evaluation()
    # exp.decision_tree()
    # exp.bagging()
    # exp.forest()
    # exp.boost()
    # exp.summary()
