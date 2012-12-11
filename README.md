Naive Bayes with KDE(Kernel Density Estimation)
===============================================

This method mainly has two feature, I'll introduce them non-technically:

1. There's not assumption about the distribution of the data, and the probabilities are derived by KDE so the result should be more reliable. Some test show that this method outperform the Gaussian Naive Bayes provided by scikit-learn(I'll upload the test details afterwards).
2. It has a memory to 'remember' the things it learn. This also allow it to, first, learn while working, and second, forget things that are too old.

More details to be added
