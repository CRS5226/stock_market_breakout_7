import redis

# Connect to Redis
r = redis.Redis(host="localhost", port=6379, db=0)

# Clear all keys from all databases
r.flushall()

# Or, to clear only the current database
# r.flushdb()
