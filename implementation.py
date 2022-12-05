from Multinomial_Naive_Bayes_Class import MNaiveBayes
classifier = MNaiveBayes()

#Splitting the dataset
x_train, y_train, x_test, y_test = classifier.train_test_split(filename, ratio)

#Training the model
classifier.fit(x_train, y_train)

#Predicting the labels
predicts = classifier.predict_label(x_test)

#Accuracy score and confusion matrix
ac = classifier.accuracy_score(y_test, predicts)
cm = classifier.confusion_matrix(y_test, predicts)