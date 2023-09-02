import redis

redis_host = redis.Redis(
	host="127.0.0.1", port=6379, db=1, decode_responses=True
)

redis_host.set("message_review_version",1)

print("Done")

