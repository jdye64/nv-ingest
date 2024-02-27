import logging
import time

import redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)


class RedisClient:
    def __init__(self, host, port, db=0, max_retries=-1, max_backoff=32, connection_timeout=10, use_ssl=False):
        self.host = host
        self.port = port
        self.db = db
        self.max_retries = max_retries
        self.max_backoff = max_backoff
        self.connection_timeout = connection_timeout
        self.use_ssl = use_ssl
        self.client = None
        self.retries = 0

    def get_client(self):
        if self.client is None or not self.ping():
            self.connect()

        return self.client

    def connect(self):
        while self.client is None:
            backoff_delay = min(2 ** self.retries, self.max_backoff)
            try:
                self.client = redis.Redis(host=self.host, port=self.port, db=self.db,
                                          socket_connect_timeout=self.connection_timeout, ssl=self.use_ssl)
                self.client.ping()
                logger.info("Successfully connected to Redis")
                self.retries = 0
            except RedisError as e:
                self.retries += 1
                if self.max_retries == 0 or self.retries < self.max_retries:
                    logger.error(f"Failed to connect to Redis: {e}, retrying in {backoff_delay}s...")
                    time.sleep(backoff_delay)
                    self.connect()
                else:
                    logger.error(
                        f"Failed to connect to Redis after {self.max_retries if self.max_retries >= 0 else 'infinite'} attempts: {e}")
                    raise

    def ping(self):
        try:
            self.client.ping()
            return True
        except (RedisError, AttributeError):
            return False

    def fetch_message(self, task_queue):
        try:
            _, job_payload = self.get_client().blpop([task_queue])
            return job_payload
        except RedisError as err:
            logger.error(f"Redis error during fetch: {err}")
            self.client = None  # Force a reconnection attempt
            raise
