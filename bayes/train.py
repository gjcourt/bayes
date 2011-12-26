#!/usr/bin/env python
import cPickle as pickle
import json
import time
from backends import RedisBackend
from bayes import NaiveBayes
from classifiers import FMClassifier
from optparse import OptionParser
parser = OptionParser(conflict_handler='resolve')
parser.add_option('-h', dest='host')
parser.add_option('-p', '--port', dest='port')
options, args = parser.parse_args()

clsfr = FMClassifier(pickle.load(open('/Users/georgecourtsunis/projects/disqus/disqus/analytics/hadoop/thread_views/var/cmap.b', 'rb')))
backend = RedisBackend(host=options.host, port=options.port)
bayes = NaiveBayes(backend=backend)

_start = time.time()
for file_name in args:
    print 'Training file %s' % file_name
    fd = open(file_name, 'r')
    _counter = 0
    for line in fd:
        _counter += 1
        if _counter % 100000 == 0:
            print _counter, (time.time() - _start)
        # if _counter % 1000000 == 0:
        #     break
        # grab args
        vector, count = line.split('\t')
        vector = json.loads(vector)
        count = int(count)
        bayes.train([vector] * count)

print 'Trained in %0.2f seconds' % (time.time() - _start)

