Naive Bayes with KDE(Kernel Density Estimation)
===============================================

This method mainly has two feature, I'll introduce them non-technically:

1. There's not assumption about the distribution of the data, and the probabilities are derived by KDE so the result should be more reliable. Some test show that this method outperform the Gaussian Naive Bayes provided by [scikit-learn](http://scikit-learn.org)(I'll upload the test details afterwards).
2. It has a memory to 'remember' the things it learn. This also allow it to, first, learn while working, and second, forget things that are too old.

Quick Start
------------

For example, if I have some data about programmers' heights and the level of their programming skill, and now I want to use heights to predict if one is a good programmer(I'm just kidding).

    >>> import nb
    >>> clf = nb.NB()
    >>> X = [[169], [172], [185], [182], [162], [160], [190]]
    >>> y = ['guru', 'guru', 'beginner', 'beginner', 'ok', 'ok']
    >>> clf.fit(X, y)

This builds the classifier. And then I enter my height.

    >>> clf.predict([171])
    'guru'

It can continue to learn after. You can continue to use:

    >>> clf.fit([[200]], ['super'])

, if you have new train case.

More details to be added
