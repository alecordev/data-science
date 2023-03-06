import pathlib

import numpy as np
import pandas as pd

from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


def load_iris():
    url = pathlib.Path(__file__).parent.parent.parent.joinpath("data", "iris.csv")
    names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
    dataset = pd.read_csv(url, names=names)
    return dataset


def example_analysis_one():
    dataset = load_iris()

    print(dataset.shape)
    print(dataset.head(20))
    print(dataset.describe())
    print(dataset.groupby("species").size())

    dataset.plot(kind="box", subplots=True, layout=(2, 2), sharex=False, sharey=False)

    dataset.hist()

    scatter_matrix(dataset)

    # Split-out validation dataset
    array = dataset.values
    X = array[:, 0:4]
    y = array[:, 4]
    X_train, X_validation, Y_train, Y_validation = train_test_split(
        X, y, test_size=0.20, random_state=1
    )

    models = [
        ("LR", LogisticRegression(solver="liblinear", multi_class="ovr")),
        ("LDA", LinearDiscriminantAnalysis()),
        ("KNN", KNeighborsClassifier()),
        ("CART", DecisionTreeClassifier()),
        ("NB", GaussianNB()),
        ("SVM", SVC(gamma="auto")),
    ]

    results = []
    names = []
    for name, model in models:
        kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        cv_results = cross_val_score(
            model, X_train, Y_train, cv=kfold, scoring="accuracy"
        )
        results.append(cv_results)
        names.append(name)
        print("%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()))

    # Compare Algorithms
    plt.boxplot(results, labels=names)
    plt.title("Algorithm Comparison")
    plt.show()

    model = SVC(gamma="auto")
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)

    # Evaluate predictions
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))


def example_analysis_two():
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    from sklearn.model_selection import train_test_split
    from pandas.plotting import parallel_coordinates
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn import metrics
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import (
        LinearDiscriminantAnalysis,
        QuadraticDiscriminantAnalysis,
    )
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression

    dataset = load_iris()
    train, test = train_test_split(
        dataset, test_size=0.4, stratify=dataset["species"], random_state=42
    )

    n_bins = 10
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].hist(train["sepal_length"], bins=n_bins)
    axs[0, 0].set_title("Sepal Length")
    axs[0, 1].hist(train["sepal_width"], bins=n_bins)
    axs[0, 1].set_title("Sepal Width")
    axs[1, 0].hist(train["petal_length"], bins=n_bins)
    axs[1, 0].set_title("Petal Length")
    axs[1, 1].hist(train["petal_width"], bins=n_bins)
    axs[1, 1].set_title("Petal Width")
    # add some spacing between subplots
    fig.tight_layout(pad=1.0)

    fig, axs = plt.subplots(2, 2)
    fn = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    cn = ["setosa", "versicolor", "virginica"]
    sns.boxplot(x="species", y="sepal_length", data=train, order=cn, ax=axs[0, 0])
    sns.boxplot(x="species", y="sepal_width", data=train, order=cn, ax=axs[0, 1])
    sns.boxplot(x="species", y="petal_length", data=train, order=cn, ax=axs[1, 0])
    sns.boxplot(x="species", y="petal_width", data=train, order=cn, ax=axs[1, 1])
    # add some spacing between subplots
    fig.tight_layout(pad=1.0)

    sns.violinplot(
        x="species",
        y="petal_length",
        data=train,
        size=5,
        order=cn,
        palette="colorblind",
    )

    sns.pairplot(train, hue="species", height=2, palette="colorblind")

    corrmat = train.corr()
    sns.heatmap(corrmat, annot=True, square=True)

    parallel_coordinates(train, "species", color=["blue", "red", "green"])

    X_train = train[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y_train = train.species
    X_test = test[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y_test = test.species

    mod_dt = DecisionTreeClassifier(max_depth=3, random_state=1)
    mod_dt.fit(X_train, y_train)
    prediction = mod_dt.predict(X_test)
    print(metrics.accuracy_score(prediction, y_test))
    print(mod_dt.feature_importances_)

    plt.figure(figsize=(10, 8))
    plot_tree(mod_dt, feature_names=fn, class_names=cn, filled=True)

    disp = metrics.plot_confusion_matrix(
        mod_dt, X_test, y_test, display_labels=cn, cmap=plt.cm.Blues, normalize=None
    )
    disp.ax_.set_title("Decision Tree Confusion matrix, without normalization")

    plt.show()

    mod_dt = GaussianNB()
    mod_dt.fit(X_train, y_train)
    prediction = mod_dt.predict(X_test)
    print(f"GaussianNB: {metrics.accuracy_score(prediction, y_test)}")
    # print(mod_dt.feature_importances_)

    model = SVC()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    print(f"SVC: {metrics.accuracy_score(prediction, y_test)}")


def example_analysis3():
    import seaborn as sns

    dataset = load_iris()
    # on top of the previous axes
    ax = sns.boxplot(x="species", y="petal_length", data=dataset)
    ax = sns.stripplot(
        x="species", y="petal_length", data=dataset, jitter=True, edgecolor="gray"
    )

    # A final seaborn plot useful for looking at univariate relations is the kdeplot,
    # which creates and visualizes a kernel density estimate of the underlying feature
    sns.FacetGrid(dataset, hue="species", height=6).map(
        sns.kdeplot, "petal_length"
    ).add_legend()

    sns.pairplot(dataset, hue="species", height=3)

    # plt.show()

    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # LogisticRegression
    # from sklearn.linear_model import LogisticRegression
    # classifier = LogisticRegression()
    # classifier.fit(X_train, y_train)
    #
    # y_pred = classifier.predict(X_test)
    #
    # # Summary of the predictions made by the classifier
    # print(classification_report(y_test, y_pred))
    # print(confusion_matrix(y_test, y_pred))
    # # Accuracy score
    # from sklearn.metrics import accuracy_score
    # print('accuracy is', accuracy_score(y_pred, y_test))

    from sklearn.neighbors import KNeighborsClassifier

    classifier = KNeighborsClassifier(n_neighbors=8)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    # Summary of the predictions made by the classifier
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    # Accuracy score
    from sklearn.metrics import accuracy_score

    print("accuracy is", accuracy_score(y_pred, y_test))

    # Support Vector Machine's
    from sklearn.svm import SVC

    classifier = SVC()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    # Summary of the predictions made by the classifier
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    # Accuracy score
    from sklearn.metrics import accuracy_score

    print("accuracy is", accuracy_score(y_pred, y_test))

    # Decision Tree's
    from sklearn.tree import DecisionTreeClassifier

    classifier = DecisionTreeClassifier()

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    # Summary of the predictions made by the classifier
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    # Accuracy score
    from sklearn.metrics import accuracy_score

    print("accuracy is", accuracy_score(y_pred, y_test))

    # Gaussian Naive Bayes
    from sklearn.naive_bayes import GaussianNB

    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    # Summary of the predictions made by the classifier
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    # Accuracy score
    from sklearn.metrics import accuracy_score

    print("accuracy is", accuracy_score(y_pred, y_test))

    # Multinomial Naive Bayes
    from sklearn.naive_bayes import MultinomialNB

    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    # Summary of the predictions made by the classifier
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    # Accuracy score
    from sklearn.metrics import accuracy_score

    print("accuracy is", accuracy_score(y_pred, y_test))

    # Bernoulli Naive Bayes
    from sklearn.naive_bayes import BernoulliNB

    classifier = BernoulliNB()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    # Summary of the predictions made by the classifier
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    # Accuracy score
    from sklearn.metrics import accuracy_score

    print("accuracy is", accuracy_score(y_pred, y_test))

    # Complement Naive Bayes
    from sklearn.naive_bayes import ComplementNB

    classifier = ComplementNB()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    # Summary of the predictions made by the classifier
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    # Accuracy score
    from sklearn.metrics import accuracy_score

    print("accuracy is", accuracy_score(y_pred, y_test))

    from sklearn.metrics import accuracy_score, log_loss

    classifiers = [
        GaussianNB(),
        MultinomialNB(),
        BernoulliNB(),
        ComplementNB(),
    ]

    # Logging for Visual Comparison
    log_cols = ["Classifier", "Accuracy", "Log Loss"]
    log = pd.DataFrame(columns=log_cols)

    for clf in classifiers:
        clf.fit(X_train, y_train)
        name = clf.__class__.__name__

        print("=" * 30)
        print(name)

        print("****Results****")
        train_predictions = clf.predict(X_test)
        acc = accuracy_score(y_test, train_predictions)
        print("Accuracy: {:.4%}".format(acc))

        log_entry = pd.DataFrame([[name, acc * 100, 11]], columns=log_cols)
        log = log.append(log_entry)

        print("=" * 30)

    fig, ax = plt.subplots()
    sns.set_color_codes("muted")
    sns.barplot(x="Accuracy", y="Classifier", data=log, color="b")

    plt.xlabel("Accuracy %")
    plt.title("Classifier Accuracy")
    plt.show()


if __name__ == "__main__":
    example_analysis3()
