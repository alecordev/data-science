import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import shap


corpus, y = shap.datasets.imdb()
corpus_train, corpus_test, y_train, y_test = train_test_split(
    corpus, y, test_size=0.2, random_state=7
)

vectorizer = TfidfVectorizer(min_df=10)
X_train = vectorizer.fit_transform(corpus_train)
X_test = vectorizer.transform(corpus_test)

model = sklearn.linear_model.LogisticRegression(penalty="l2", C=0.1)
model.fit(X_train, y_train)

explainer = shap.LinearExplainer(model, X_train)
shap_values = explainer.shap_values(X_test)
X_test_array = (
    X_test.toarray()
)  # we need to pass a dense version for the plotting functions


shap.summary_plot(
    shap_values, X_test_array, feature_names=vectorizer.get_feature_names_out()
)

ind = 0
shap.force_plot(
    explainer.expected_value,
    shap_values[ind, :],
    X_test_array[ind, :],
    feature_names=vectorizer.get_feature_names_out(),
)

print("Positive" if y_test[ind] else "Negative", "Review:")
print(corpus_test[ind])
