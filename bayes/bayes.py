# import sys
# import time
from collections import defaultdict as dd

mean = lambda x: sum(x) / len(x)

exact_match = lambda w, c: w == c
starts_match = lambda w, c: w.startswith(c) and len(c) > (len(w) / 2)
word_in_cat = lambda w, c: w in c
cat_in_word = lambda w, c: c in w

stop_words = set([
    'or', 'of', 'and', 'to', '&', 'back', 'new', 'personal',
    'power', 'care', 'back', 'loss', 'day', 'video',
])

class NaiveBayes(object):
    """
    Simple implementation of a naive Bayes classifier.
    """
    def __init__(self, backend=None):
        if backend is None:
            # TODO fillin with a default connection or something
            raise Exception, 'Hey asshole, supply a backend'
        self.b = backend

    def _calc_feature_score(self, classifier, feature, scale=1):
        """
        Given a feature and classifier, determine the score
        """
        r = 0
        # normalize the feature
        feature = feature.lower()
        conds = (
            exact_match,
            starts_match,
            word_in_cat,
            cat_in_word,
        )
        for cond in conds:
            # shortcut exact to hit sets
            if cond == exact_match:
                if feature == classifier['name']:
                    r += 1 * scale
                elif feature in classifier['parts_s'].difference(stop_words):
                    r += 1 * scale * 0.95
                elif feature in classifier['hyponyms_s']:
                    r += 1 * scale * 0.90
                elif feature in classifier['hypernyms_s']:
                    r += 1 * scale * 0.90
                else:
                    continue
            # normal comparisons
            else:
                if cond(feature, classifier['name']):
                    r += 1 * scale
                # check parts
                elif any(cond(feature, c_p) for c_p in classifier['parts'] if c_p not in stop_words):
                    r += 1 * scale
                # TODO each of these checks will slow down classifier
                # # check hyponyms
                # elif any(cond(feature, c_ho) for c_ho in classifier['hyponyms'] if c_ho not in stop_words):
                elif any(cond(feature, c_ho) for c_ho in classifier['hyponyms']):
                    r += 1 * scale * 0.90
                # check hypernyms
                elif any(cond(feature, c_h) for c_h in classifier['hypernyms']):
                    r += 1 * scale * 0.90
                # # check part hyponyms
                # elif any(cond(feature, c) for c in classifier['phyponyms']):
                #     r = 1 * scale * 0.001
                # # check part hypernyms
                # elif any(cond(feature, c) for c in classifier['phypernyms']):
                #     r = 1 * scale * 0.002
                else:
                    continue
        return r + 1e-10 # small K for > 0 multiplication

    def _calc_scores(self, Classifier, Feature, key, **kwargs):
        """
        p(C) = p(f1|c1) * p(f2|c1), etc for all f in F, for all c in C
        """
        # build up a map of all cats/features/scores
        cat_feature_score_map = dd(dict)
        for classifier, meta in Classifier.iteritems():
            meta['name'] = classifier
            # TODO figure out if we want to do something more interesting here
            for i, feature in enumerate(Feature):
                # feature not in cat map
                if feature not in cat_feature_score_map[classifier]:
                    if kwargs.get('linear_weight_vector'):
                        cat_feature_score_map[classifier][feature] = self._calc_feature_score(meta, feature, scale=i+1)
                    else:
                        cat_feature_score_map[classifier][feature] = self._calc_feature_score(meta, feature)
                # feature in cat map
                else:
                    if kwargs.get('linear_weight_vector'):
                        cat_feature_score_map[classifier][feature] += self._calc_feature_score(meta, feature, scale=i+1)
                    else:
                        cat_feature_score_map[classifier][feature] += self._calc_feature_score(meta, feature)
                # train the bayesian based on the situational names
                if cat_feature_score_map[classifier][feature] > 0.50:
                    pass
                    # self.b.update_classification(Classifier.ident, classifier, key)

        # calculate the probabilty distribution
        cat_score_map = dd(float)
        for classifier, features in cat_feature_score_map.iteritems():
            # grab currently probability
            p_cat = self.b.get_classification(Classifier.ident, classifier, key)
            # print classifier, p_cat
            # score = p(c) * p(f1|c1) * p(f2|c1), etc for all f in F, for all c in C
            scores = [f[1] for f in features.iteritems()]
            cat_score_map[classifier] = (p_cat * reduce(lambda x, y: x * y, scores, 1))
        # return a list of probabilities for each category (between 0 and 1)
        return sorted(cat_score_map.iteritems(), key=lambda x:x[1], reverse=True)

    def _calc_classification(self, Classifier, Feature, key, **kwargs):
        scores = self._calc_scores(Classifier, Feature, key, **kwargs)
        return scores[0] # most relevant classification

    def train(self, vectors):
        # vector -> Classifier_id, classifier, key
        for v in vectors:
            self.b.batch_update(*v)
        # persist
        self.b.flush_pipe()

    def classify(self, Classifier, Feature, key, **kwargs):
        classifier, prob = self._calc_classification(Classifier, Feature, key, **kwargs)
        self.b.update_classification(Classifier, classifier, key)
        # print 'Classified as', classifier, prob
        return classifier, prob

