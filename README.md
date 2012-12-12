Naive Bayes with KDE(Kernel Density Estimation)
===============================================

This method mainly has two features, I'll introduce them non-technically:

1. There's not assumption about the distribution of the data, and the probabilities are derived by KDE so the result should be more reliable. Some test shows that this method outperforms the Gaussian/Multinomial Naive Bayes provided by [scikit-learn](http://scikit-learn.org)(I'll upload the test details afterwards).
2. It has a memory to 'remember' the things it learns. This also allow it to, first, learn while working, and second, forget things that are too old.

Quick Start
------------

For example, if I have some data about programmers' heights and the level of their programming skill, and now I want to use heights to predict if one is a good programmer(I'm just kidding).

    >>> import nb
    >>> clf = nb.NB()
    >>> X = [[169], [172], [185], [182], [162], [160], [190], [192]]
    >>> y = ['guru', 'guru', 'beginner', 'beginner', 'ok', 'guru', 'guru']
    >>> clf.fit(X, y)

This builds the classifier. And then I enter my height.

    >>> clf.predict([171])
    'guru'

It can continue to learn after this. You can continue to use:

    >>> clf.fit([[200]], ['super'])

, if you have new train case.

More details to be added

TODOs
-------

1. There's no bandwidth selection in the KDE currently. I'll fix it ASAP.
2. It's slow. More optimization needed.
