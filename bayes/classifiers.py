import cPickle as pickle

class NaiveBayesClassifier(dict):
    """
    Namespaces classifiers so that we can refer to them individually

    Simple wrapper around the dict object
    """
    def __init__(self, *args, **kwargs):
        super(NaiveBayesClassifier, self).__init__(*args, **kwargs)

    def _getident(self):
        return self._ident
    ident = property(_getident)

    @classmethod
    def load(cls, file_path):
        return cls(pickle.load(open(file_path, 'rb')))

class ExampleClassifier(NaiveBayesClassifier):
    """
    Example Classifier
    """
    def __init__(self, *args, **kwargs):
        self._ident = 'example'
        super(NaiveBayesClassifier, self).__init__(*args, **kwargs)

