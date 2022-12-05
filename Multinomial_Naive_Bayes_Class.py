import numpy as np
import pandas as pd

class MNaiveBayes:
    def load_dataset(self, file):
        dataset = pd.read_csv(file).values
        np.random.shuffle(dataset)
        x = dataset[:, :-1]
        y = dataset[:, -1]
        return x, y

    def train_test_split(self, file, ratio):
        x, y = self.load_dataset(file)
        x_test_size = int(ratio * len(x))
        x_train_size = len(x) - x_test_size
        y_test_size = int(ratio * len(y))
        y_train_size = len(y) - y_test_size

        x_train = x[:x_train_size]
        x_test = x[x_train_size:]
        y_train = y[:y_train_size]
        y_test = y[y_train_size:]

        return x_train, y_train, x_test, y_test
    def est_prior(self, y):
        prior = np.zeros(shape = (5), dtype= float)
        list = []
        instance_count = {}
        for element in y:
            if element in instance_count:
                instance_count[element] += 1
            else:
                instance_count[element] = 1
        total = sum(instance_count.values())
        for i in range(len(instance_count)):
                prior[i] = (instance_count[i] / total)
        prior = np.log(prior)
        return prior

    def est_map(self, X, y, alpha=1):
        article, word = X.shape
        t = np.zeros(shape = (word,5), dtype=int)
        for i in range(article):  
            if y[i] == 0:
                for j in range(word):
                    t[j,0] += X[i,j]
            elif y[i] == 1:
                for j in range(word):
                    t[j,1] += X[i,j]
            elif y[i] == 2:
                for j in range(word):
                    t[j,2] += X[i,j]
            elif y[i] == 3:
                for j in range(word):
                    t[j,3] += X[i,j]
            elif y[i] == 4:
                for j in range(word):
                    t[j,4] += X[i,j]

                    
        sum_0 = sum(t[:, 0]) + (alpha * word)
        sum_1 = sum(t[:, 1]) + (alpha * word)
        sum_2 = sum(t[:, 2]) + (alpha * word)
        sum_3 = sum(t[:, 3]) + (alpha * word)
        sum_4 = sum(t[:, 4]) + (alpha * word)
        sum_list = [sum_0, sum_1, sum_2, sum_3, sum_4]
        
        mle = np.zeros(shape = (word, 5), dtype = float)
        for j in range(word):
            for i in range(5):
                mle[j,i] = (t[j,i] + 1) / sum_list[i]
                
        mle= np.log(mle)
        mle = np.nan_to_num(mle, neginf=np.nan_to_num(-np.inf))
        return mle

    def est_prob(self, X):
        article, words = X.shape
        prob = np.zeros(shape = (article,5), dtype=float)
        for i in range(5):
            for j in range(article):
                prob[j,i] = self.prior[i]
                for k in range(words):
                        prob[j,i] += self.mle[k,i] * X[j,k]
        
        y_pred = np.zeros(shape=(article), dtype=float)
        for i in range(article):
            y_pred[i] = np.argmax(prob[i], axis=0)

        return y_pred
    
    def fit(self,x, y):
        prior = self.est_prior(y)
        mle = self.est_map(x, y)
        self.prior = prior
        self.mle = mle

    def predict_label(self, X):
        c_pred = self.est_prob(X)
        return c_pred
    
    def confusion_matrix(self, true, predicted):
        labels = np.unique(predicted)
        matrix = np.zeros(shape=(len(labels),len(labels)))
        for i in range(len(labels)):
            for j in range(len(labels)):
                matrix[j,i] = np.sum((true == labels[i]) & (predicted == labels[j]))
        matrix = matrix.astype('int')
        return matrix

    def accuracy_score(self, true, predicted):
        accuracy = np.sum(np.equal(true, predicted)) / len(true)
        return accuracy
