import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, silhouette_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier


#Lib implementations of 3 selections
def embedded_feature_selection(X_train, y_train, k=30):
    model = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
    model.fit(X_train, y_train)
    coef = np.abs(model.coef_[0])
    return np.argsort(coef)[-k:]


def wrapper_feature_selection(X_train, y_train, model, k=30):
    model.fit(X_train, y_train)
    importances = (np.sum(np.abs(model.feature_log_prob_), axis=0)
                   if isinstance(model, MultinomialNB) else np.abs(model.coef_[0]))
    return np.argsort(importances)[-k:]


def filter_feature_selection(X_train, y_train, k=30):
    chi2_scores, _ = chi2(X_train, y_train)
    return np.argsort(chi2_scores)[-k:]

#Classifiers evaluation funcs
def evaluate_classifier(X_train, X_test, y_train, y_test, classifier):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    return accuracy_score(y_test, y_pred)


classifiers = {
    "MultinomialNB": MultinomialNB(),
    "LogisticRegression": LogisticRegression(max_iter=500, random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42)
}


def evaluate_all_classifiers(X_train, X_test, y_train, y_test):
    scores = {}
    for name, clf in classifiers.items():
        score = evaluate_classifier(X_train, X_test, y_train, y_test, clf)
        scores[name] = score
    return scores


# 30 top, clustering, pcs tsne
def print_top_features(vectorizer, feature_indices, method_name):
    feature_names = np.array(vectorizer.get_feature_names_out())
    top_features = feature_names[feature_indices]
    print(f"Top 30 features ({method_name}):")
    print(", ".join(top_features))
    print()


def perform_clustering(X, y, method_name):
    kmeans = KMeans(n_clusters=2, random_state=42)
    y_pred = kmeans.fit_predict(X)
    silhouette = silhouette_score(X, y_pred)
    nmi = normalized_mutual_info_score(y, y_pred)
    print(f"Clustering results ({method_name}):")
    print(f"Silhouette Score: {silhouette}")
    print(f"NMI: {nmi}")
    print()


def plot_pca_tsne(X_train_vect, X_test_vect, y_train, y_test, method_name):
    def pca_tsne(data, method):
        return method.fit_transform(data.toarray())

    X_train_pca = pca_tsne(X_train_vect, PCA(n_components=2, random_state=42))
    X_test_pca = pca_tsne(X_test_vect, PCA(n_components=2, random_state=42))

    X_train_tsne = pca_tsne(X_train_vect, TSNE(n_components=2, random_state=42, perplexity=min(30, len(y_train) - 1)))
    X_test_tsne = pca_tsne(X_test_vect, TSNE(n_components=2, random_state=42, perplexity=min(30, len(y_test) - 1)))

    plt.figure(figsize=(12, 5))

    sns.scatterplot(x=X_train_pca[:, 0], y=X_train_pca[:, 1], hue=y_train, ax=plt.subplot(2, 2, 1)).set_title(f"PCA train - {method_name}")
    sns.scatterplot(x=X_train_tsne[:, 0], y=X_train_tsne[:, 1], hue=y_train, ax=plt.subplot(2, 2, 2)).set_title(f"t-SNE train - {method_name}")
    sns.scatterplot(x=X_test_pca[:, 0], y=X_test_pca[:, 1], hue=y_test, ax=plt.subplot(2, 2, 3)).set_title(f"PCA test - {method_name}")
    sns.scatterplot(x=X_test_tsne[:, 0], y=X_test_tsne[:, 1], hue=y_test, ax=plt.subplot(2, 2, 4)).set_title(f"t-SNE test - {method_name}")

    plt.tight_layout()
    plt.show()


# Read and vectorize
#######################################################################################################################
data = pd.read_csv("SMS.tsv", sep='\t')[['class', 'text']].rename(columns={'class': 'label'})
data['label'] = data['label'].map({'spam': 0, 'ham': 1})


X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

vectorizer = CountVectorizer(max_features=1000)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)


# Evaluating classifiers and printing top30
#######################################################################################################################
print_top_features(vectorizer, embedded_feature_selection(X_train_vect, y_train), "L1")
print_top_features(vectorizer, wrapper_feature_selection(X_train_vect, y_train, MultinomialNB()), "Wrapper")
print_top_features(vectorizer, filter_feature_selection(X_train_vect, y_train), "Chi-squared")


print("Evaluating classifiers on original data:")
original_scores = evaluate_all_classifiers(X_train_vect, X_test_vect, y_train, y_test)
for name, score in original_scores.items():
    print(f"{name}: {score}")
print("")

print("Evaluating classifiers after feature selection:")
for method_name, feature_selector in {
    "L1": lambda: embedded_feature_selection(X_train_vect, y_train),
    "Wrapper": lambda: wrapper_feature_selection(X_train_vect, y_train, MultinomialNB()),
    "Chi-squared": lambda: filter_feature_selection(X_train_vect, y_train)
}.items():
    selected_features = feature_selector()
    X_train_fs = X_train_vect[:, selected_features]
    X_test_fs = X_test_vect[:, selected_features]
    scores = evaluate_all_classifiers(X_train_fs, X_test_fs, y_train, y_test)
    print(f"Feature Selection ({method_name}):")
    for name, score in scores.items():
        print(f"{name}: {score}")
    print("")


# Clustering
#######################################################################################################################
print("\nClustering results:")
perform_clustering(X_train_vect, y_train, "Original")
perform_clustering(X_train_vect[:, embedded_feature_selection(X_train_vect, y_train)], y_train, "L1")
perform_clustering(X_train_vect[:, wrapper_feature_selection(X_train_vect, y_train, MultinomialNB())], y_train, "Wrapper")
perform_clustering(X_train_vect[:, filter_feature_selection(X_train_vect, y_train)], y_train, "Chi-squared")


# PCA/TSNE
#######################################################################################################################
plot_pca_tsne(X_train_vect, X_test_vect, y_train, y_test, 'Original')

plot_pca_tsne(X_train_vect[:, embedded_feature_selection(X_train_vect, y_train)],
              X_test_vect[:, embedded_feature_selection(X_train_vect, y_train)],
              y_train, y_test, 'L1 Feature Selection')

plot_pca_tsne(X_train_vect[:, wrapper_feature_selection(X_train_vect, y_train, MultinomialNB())],
              X_test_vect[:, wrapper_feature_selection(X_train_vect, y_train, MultinomialNB())],
              y_train, y_test, 'Wrapper Feature Selection')

plot_pca_tsne(X_train_vect[:, filter_feature_selection(X_train_vect, y_train)],
              X_test_vect[:, filter_feature_selection(X_train_vect, y_train)],
              y_train, y_test, 'Filter Feature Selection')
