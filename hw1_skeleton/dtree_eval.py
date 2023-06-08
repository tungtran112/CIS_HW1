# '''
#     TEMPLATE FOR MACHINE LEARNING HOMEWORK
#     AUTHOR Eric Eaton, Chris Clingerman
# '''

# import numpy as np
# import matplotlib.pyplot as plt

# from sklearn import tree
# from sklearn.metrics import accuracy_score



# def evaluatePerformance():
#     '''
#     Evaluate the performance of decision trees,
#     averaged over 1,000 trials of 10-fold cross validation
    
#     Return:
#       a matrix giving the performance that will contain the following entries:
#       stats[0,0] = mean accuracy of decision tree
#       stats[0,1] = std deviation of decision tree accuracy
#       stats[1,0] = mean accuracy of decision stump
#       stats[1,1] = std deviation of decision stump
#       stats[2,0] = mean accuracy of 3-level decision tree
#       stats[2,1] = std deviation of 3-level decision tree
      
#     ** Note that your implementation must follow this API**
#     '''
    
#     # Load Data
#     filename = 'data/SPECTF.dat'
#     data = np.loadtxt(filename, delimiter=',')
#     X = data[:, 1:]
#     y = np.array([data[:, 0]]).T
#     n,d = X.shape

#     # shuffle the data
#     idx = np.arange(n)
#     np.random.seed(13)
#     np.random.shuffle(idx)
#     X = X[idx]
#     y = y[idx]
    
#     # split the data
#     Xtrain = X[1:101,:]  # train on first 100 instances
#     Xtest = X[101:,:]
#     ytrain = y[1:101,:]  # test on remaining instances
#     ytest = y[101:,:]

#     # train the decision tree
#     clf = tree.DecisionTreeClassifier()
#     clf = clf.fit(Xtrain,ytrain)

#     # output predictions on the remaining data
#     y_pred = clf.predict(Xtest)

#     # compute the training accuracy of the model
#     meanDecisionTreeAccuracy = accuracy_score(ytest, y_pred)
    
    
#     # TODO: update these statistics based on the results of your experiment
#     stddevDecisionTreeAccuracy = 0
#     meanDecisionStumpAccuracy = 0
#     stddevDecisionStumpAccuracy = 0
#     meanDT3Accuracy = 0
#     stddevDT3Accuracy = 0

#     # make certain that the return value matches the API specification
#     stats = np.zeros((3,2))
#     stats[0,0] = meanDecisionTreeAccuracy
#     stats[0,1] = stddevDecisionTreeAccuracy
#     stats[1,0] = meanDecisionStumpAccuracy
#     stats[1,1] = stddevDecisionStumpAccuracy
#     stats[2,0] = meanDT3Accuracy
#     stats[2,1] = stddevDT3Accuracy
#     return stats



# # Do not modify from HERE...
# if __name__ == "__main__":
    
#     stats = evaluatePerformance()
#     print("Decision Tree Accuracy =", stats[0,0], "(", stats[0,1], ")")
#     print("Decision Stump Accuracy =", stats[1,0], "(", stats[1,1], ")")
#     print("3-level Decision Tree =", stats[2,0], "(", stats[2,1], ")")

# # ...to HERE.
import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

def evaluatePerformance():
    '''
    Evaluate the performance of decision trees,
    averaged over 1,000 trials of 10-fold cross validation
    
    Return:
      a matrix giving the performance that will contain the following entries:
      stats[0,0] = mean accuracy of decision tree
      stats[0,1] = std deviation of decision tree accuracy
      stats[1,0] = mean accuracy of decision stump
      stats[1,1] = std deviation of decision stump
      stats[2,0] = mean accuracy of 3-level decision tree
      stats[2,1] = std deviation of 3-level decision tree
      
    ** Note that your implementation must follow this API**
    '''
    
    # Load Data
    filename = 'data/SPECTF.dat'
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, 1:]
    y = np.array([data[:, 0]]).T
    n,d = X.shape

    # shuffle the data
    idx = np.arange(n)
    np.random.seed(13)
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    # Initialize arrays to store accuracy values
    dt_accuracies = []
    ds_accuracies = []
    dt3_accuracies = []

    # Perform 1,000 trials of 10-fold cross-validation
    for _ in range(1000):
        kf = KFold(n_splits=10, shuffle=True)

        # Train and test decision tree on each fold
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Train the decision tree classifier
            dt_clf = tree.DecisionTreeClassifier()
            dt_clf.fit(X_train, y_train)

            # Train the decision stump classifier
            ds_clf = tree.DecisionTreeClassifier(max_depth=1)
            ds_clf.fit(X_train, y_train)

            # Train the 3-level decision tree classifier
            dt3_clf = tree.DecisionTreeClassifier(max_depth=3)
            dt3_clf.fit(X_train, y_train)

            # Make predictions on the testing set
            dt_y_pred = dt_clf.predict(X_test)
            ds_y_pred = ds_clf.predict(X_test)
            dt3_y_pred = dt3_clf.predict(X_test)

            # Compute accuracies and store them
            dt_accuracy = accuracy_score(y_test, dt_y_pred)
            ds_accuracy = accuracy_score(y_test, ds_y_pred)
            dt3_accuracy = accuracy_score(y_test, dt3_y_pred)

            dt_accuracies.append(dt_accuracy)
            ds_accuracies.append(ds_accuracy)
            dt3_accuracies.append(dt3_accuracy)

    # Compute mean and standard deviation of accuracy values
    meanDecisionTreeAccuracy = np.mean(dt_accuracies)
    stddevDecisionTreeAccuracy = np.std(dt_accuracies)
    meanDecisionStumpAccuracy = np.mean(ds_accuracies)
    stddevDecisionStumpAccuracy = np.std(ds_accuracies)
    meanDT3Accuracy = np.mean(dt3_accuracies)
    stddevDT3Accuracy = np.std(dt3_accuracies)

    # make certain that the return value matches the API specification
    stats = np.zeros((3,2))
    stats[0,0] = meanDecisionTreeAccuracy
    stats[0,1] = stddevDecisionTreeAccuracy
    stats[1,0] = meanDecisionStumpAccuracy
    stats[1,1] = stddevDecisionStumpAccuracy
    stats[2,0] = meanDT3Accuracy
    stats[2,1] = stddevDT3Accuracy
    return stats

# Do not modify from HERE...
if __name__ == "__main__":
    stats = evaluatePerformance()
    print("Decision Tree Accuracy =", stats[0,0], "(", stats[0,1], ")")
    print("Decision Stump Accuracy =", stats[1,0], "(", stats[1,1], ")")
    print("3-level Decision Tree =", stats[2,0], "(", stats[2,1], ")")
# ...to HERE.
