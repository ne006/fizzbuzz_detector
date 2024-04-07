from sklearn.naive_bayes import MultinomialNB
import pandas
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import glob
import os

all_files = glob.glob(os.path.join("./data/*.csv"))

X_train, X_test, y_train, y_test = [None, None, None, None]

for filename in all_files:
    df = pandas.read_csv(filename, index_col=None, header=0, names=['text', 'labels'])
    df['labels'] = df['labels'].apply(lambda s : s.split(';') if isinstance(s, str) else [])

    pX_train, pX_test, py_train, py_test = train_test_split(df['text'], df['labels'])
    
    if X_train is None:
        X_train, X_test, y_train, y_test = pX_train, pX_test, py_train, py_test
    else:
        X_train = pandas.concat([X_train, pX_train])
        X_test = pandas.concat([X_test, pX_test])
        y_train = pandas.concat([y_train, py_train])
        y_test = pandas.concat([y_test, py_test])

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)

tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)

mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(y_train.to_list())

clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])
clf = MultiOutputClassifier(clf)
clf.fit(X_train, labels)

print(classification_report(mlb.fit_transform(y_test.to_list()), clf.predict(X_test)))

for word in ['fizz', 'buzz', 'teapot']:
    prediction = [
        clf.predict([word])[0].tolist(),
        clf.predict_proba([word])
    ]

    prediction[0] = [(mlb.classes_[i] if value == 1 else None) for i, value in enumerate(prediction[0])]
    prediction[0] = [x for x in prediction[0] if x is not None]

    print(f'{word}: {prediction}')