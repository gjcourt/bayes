#!/usr/bin/env python
import cPickle as pickle
from backends import RedisBackend
from bayes import NaiveBayes
from classifiers import FMClassifier

training_data = (
    ('fm', 'politics', 'cnn'),
    ('fm', 'politics', 'cnn'),
    ('fm', 'politics', 'cnn'),
    ('fm', 'politics', 'cnn'),
    ('fm', 'politics', 'cnn'),
    ('fm', 'politics', 'cnn'),
    ('fm', 'politics', 'cnn'),
    ('fm', 'politics', 'cnn'),
    ('fm', 'politics', 'cnn'),
    ('fm', 'politics', 'cnn'),
    ('fm', 'politics', 'cnn'),
    ('fm', 'politics', 'cnn'),
    ('fm', 'politics', 'cnn'),
    ('fm', 'politics', 'cnn'),
    ('fm', 'politics', 'cnn'),
    ('fm', 'politics', 'cnn'),
    ('fm', 'politics', 'cnn'),
    ('fm', 'politics', 'cnn'),
    ('fm', 'business', 'cnn'),
    ('fm', 'business', 'cnn'),
    ('fm', 'business', 'cnn'),
    ('fm', 'business', 'cnn'),
    ('fm', 'business', 'cnn'),
    ('fm', 'business', 'cnn'),
    ('fm', 'business', 'cnn'),
    ('fm', 'business', 'cnn'),
    ('fm', 'business', 'cnn'),
    ('fm', 'business', 'cnn'),
    ('fm', 'business', 'cnn'),
    ('fm', 'music', 'cnn'),
    ('fm', 'music', 'cnn'),
    ('fm', 'music', 'cnn'),
    ('fm', 'music', 'cnn'),
    ('fm', 'music', 'cnn'),
    ('fm', 'music', 'cnn'),
    ('fm', 'music', 'cnn'),
    ('fm', 'music', 'cnn'),
    ('fm', 'tech news', 'cnn'),
    ('fm', 'tech news', 'cnn'),
    ('fm', 'tech news', 'cnn'),
    ('fm', 'tech news', 'cnn'),
    ('fm', 'tech news', 'cnn'),
    ('fm', 'tech news', 'cnn'),
    ('fm', 'tech news', 'cnn'),
    ('fm', 'tech news', 'cnn'),
    ('fm', 'tech news', 'cnn'),
    ('fm', 'tech news', 'cnn'),
)

cmap = pickle.load(open('../hadoop/thread_views/var/cmap.b',
        'rb'))
clsfr = FMClassifier(cmap)
backend = RedisBackend()
bayes = NaiveBayes(backend=backend)
bayes.train(training_data)
# no we are ready to test the bayes filter

# TODO add support for subfeatures in features

import time
_start = time.time()
# bayes.classify(clsfr, ('aldfksjalskdjfasdflapoliticsadlskfajsldfj',), 'cnn')
# bayes.classify(clsfr, ('politics', 'aldfksjalspoliticskdjfasdflbusinessapmusicadlhomeskfajsldfj', 'music'), 'cnn')
# bayes.classify(clsfr, ('business', 'music', 'love', 'living', 'politics', 'music'), 'cnn', linear_weight_vector=True)
# bayes.classify(clsfr, ('tech', 'computers', 'news'), 'cnn')
line = 'gaming.www.myvidster.com/video/2797926/PornoTubecom_-_Keymon_Phoenix_Mister_Buck_Dee_Truth_Intrigue_and_Jermany_-_Browsin'
bayes.classify(clsfr, line.split('/'), 'myvidster.com', linear_weight_vector=True)
print (time.time() - _start), 'seconds'

