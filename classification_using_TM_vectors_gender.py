from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from topic_modeling_predict_gender import run_TM
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

for n_topics in range(2, 20):
    print('\n\nNEW REPORT\n\n')

    TM_results = run_TM(n_topics, 0, 1)

    X = list()
    y = list()  # MALE 0, FEMALE 1
    for entry in TM_results:
        if entry.startswith('MALE'):
            y.append(0)
            X.append(TM_results[entry])
        if entry.startswith('FEMALE'):
            y.append(1)
            X.append(TM_results[entry])

    print('Number of entries in the SVM model:', len(X))
    print('Number or topics (features):', n_topics)
    clf1 = SVC()
    scores = cross_val_score(clf1, X, y, cv=20)
    print(scores)
    print("Accuracy of cross-validation: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2), '\n')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf2 = SVC()
    print('Size of the train set:', len(X_train))
    print('Size of the test set:', len(X_test))
    clf2.fit(X_train, y_train)
    y_pred = clf2.predict(X_test)
    print(set(y_pred))
    print('Accuracy of classificator:', metrics.accuracy_score(y_test, y_pred), '\n')
    print('Classification report:', classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))