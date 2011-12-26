import os
import unittest2 as unittest
from disqus.analytics.bayes.bayes import NaiveBayes
from disqus.analytics.bayes.backends import RedisBackend
from disqus.analytics.bayes.classifiers import FMClassifier
from django.conf import settings
from redis import Redis


PORT = 6379


class NaiveBayesianTest(unittest.TestCase):

    def setUp(self):
        self.b = NaiveBayes(backend=RedisBackend(host=HOST, port=PORT))
        self.clsfr = FMClassifier.load(os.path.join(settings.DISQUS_PATH, 'analytics', 'hadoop', 'thread_views', 'classifiers', 'fm_cmap.b'))

    def tearDown(self):
        pass

    def _clear_redis(self):
        # clear out any existing redis data
        conn = Redis(host=HOST, port=PORT)
        conn.flushdb()

    def _train_bayesian(self):
        v0 = ('fm', 'politics', 'cnn')
        v1 = ('fm', 'tech news', 'avc')
        v2 = ('fm', 'music', 'likefm')
        self.b.train([v0] * 10)
        self.b.train([v1] * 11)
        self.b.train([v2] * 12)

    def test_initial_training(self):
        self._clear_redis()
        self._train_bayesian()
        self.assertEquals('politics', self.b.classify(self.clsfr, 'politics', 'cnn')[0])
        self.assertEquals('politics', self.b.classify(self.clsfr, 'summer', 'cnn')[0])
        self.assertEquals('tech news', self.b.classify(self.clsfr, 'tech news', 'avc')[0])
        self.assertEquals('tech news', self.b.classify(self.clsfr, 'summer', 'avc')[0])
        self.assertEquals('tech news', self.b.classify(self.clsfr, 'techsummerpolitics', 'avc')[0])
        self.assertEquals('music', self.b.classify(self.clsfr, 'music', 'likefm')[0])
        self.assertEquals('music', self.b.classify(self.clsfr, 'relationships & dating', 'likefm')[0])

