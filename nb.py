import scipy.stats
import numpy as np

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

    def predict(self, data):
        score = map(self.class_prob, self.all_class)
        for i in range(len(self.all_class)):
            for col in range(self.train_set.shape[1]):
                den = 0
                for row in range(self.train_set.shape[0]):
                    if self.classes[row] != self.all_class[i]:
                        continue
                    den += self.kernel_g(self.train_set[row, col], data[col])
                score[i] *= den
        return self.all_class[np.argmax(score)]

    def kernel_g(self, a, b):
        return scipy.stats.norm(0, 1).pdf(a - b)
