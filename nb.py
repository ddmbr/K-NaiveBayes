import scipy.stats
import numpy as np
import utils

class NB:
    def __init__(self):
        self.train_set = np.array([])
        self.classes = np.array([])
        self.mem_size = 500

    def fit(self, train_set, classes):
        if self.train_set.size == 0:
            self.train_set = np.array(train_set)
        else:
            self.train_set = np.append(self.train_set, train_set, axis=0)
        self.classes = np.append(self.classes, classes, axis=0)
        self.all_class = np.unique(self.classes)

        if self.train_set.shape[0] > 500:
            start = self.train_set.shape[0] - 500
            self.train_set = self.train_set[start:, :]
            self.classes = self.classes[start:, :]

    def class_prob(self, cls):
        n = len([item for item in self.classes if item == cls])
        d = len(self.classes)
        return n * 1.0 /d

    def _joint_log_likelihood(self, X):
        X = utils.array2d(X)
        jll = np.zeros((X.shape[0], np.size(self.all_class)))
        for i in range(np.size(self.all_class)):
            jll[:,i] += np.log(self.class_prob(self.all_class[i]))
        #print jll
        for i, x in enumerate(X):
            for cls in self.all_class:
                #prob = 1e-31
                prob = 0
                for j, f in enumerate(self.train_set):
                    if self.classes[j] != cls: continue
                    prob += self.kernel_g(x - f)
                    #print x, f, prob, self.kernel_g(x - f)
                c = np.where(self.all_class == cls)
                #print prob
                jll[i, c] += np.log(prob)
        #print jll
        return jll.T

    def predict(self, X):
        jll = self._joint_log_likelihood(X)
        return self.all_class[np.argmax(jll, axis=0)]

    def kernel_g(self, x):
        x = utils.array1d(x)
        #print x
        res = 1
        for col in x:
            res *= scipy.stats.norm(0, 1).pdf(col)
        return res
