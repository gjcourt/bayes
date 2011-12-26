from redis import Redis

class NaiveBayesBackend(object):

    def __init__(self, host, port):
        if not host:
            host = 'localhost'
        if not port:
            port = 6379
        self.conn = Redis(host=host, port=port)

    def get_classification(self, C, c, k):
        """
        p(c) for the given key
        for a given key and classifier, get the probability
        """
        raise NotImplementedError

    def update_classification(self, C, c, k):
        raise NotImplementedError


class RedisBackend(NaiveBayesBackend):

    def __init__(self, host=None, port=None):
        # call up to init the redis connection
        super(RedisBackend, self).__init__(host, port)
        self._count = 0
        self.pipe = self.conn.pipeline()

    def _get_name(self, C, k):
        return '%s.%s' % (C, k)

    def _get_keys(self, C, c, k):
        return (
            '%s' % c,
            'T',
        )

    def flush_pipe(self):
        self.pipe.execute()
        self.pipe = self.conn.pipeline()

    def get_classification(self, C, c, k):
        name = self._get_name(C, k)
        keys = self._get_keys(C, c, k)
        c_count = float(self.conn.hget(name, keys[0]) or 0)
        C_count = float(self.conn.hget(name, keys[1]) or 0) + 1e-10 # correct for /0 errors
        return c_count / C_count

    def update_classification(self, C, c, k):
        name = self._get_name(C, k)
        keys = self._get_keys(C, c, k)
        self.conn.hincrby(name, keys[0], 1)
        self.conn.hincrby(name, keys[1], 1)

    def batch_update(self, C, c, k):
        name = self._get_name(C, k)
        keys = self._get_keys(C, c, k)
        self.pipe.hincrby(name, keys[0], 1)
        self.pipe.hincrby(name, keys[1], 1)
        self._count += 1
        if self._count % 100000 == 0:
            self.flush_pipe()
            self._count = 0

