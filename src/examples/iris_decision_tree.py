from sklearn import datasets, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# load the iris datasets
dataset = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    dataset["data"], dataset["target"], random_state=0
)

# fit a CART model to the data
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
print(model)

# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)

# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
print("Test score: {:.2f}".format(model.score(X_test, y_test)))

y_pred = model.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


def save_tree_graph():
    import io
    from sklearn.tree import export_graphviz
    import pydotplus

    dot_data = io.StringIO()
    export_graphviz(
        model,
        out_file=dot_data,
        filled=True,
        rounded=True,
        special_characters=True,
        feature_names=dataset['feature_names'],
        class_names=['0', '1', '2'],
    )
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('iris_tree.png')
