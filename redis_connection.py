import os
import redis
import json
import numpy as np
from time import sleep


# Replace values below with your own if using Redis Cloud instance
REDIS_HOST = os.getenv("REDIS_HOST", "localhost") # ex: "redis-18374.c253.us-central1-1.gce.cloud.redislabs.com"
REDIS_PORT = os.getenv("REDIS_PORT", "6379")      # ex: 18374
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")  # ex: "1TNxTEdYRDgIDKM2gDfasupCADXXXX"

# If SSL is enabled on the endpoint, use rediss:// as the URL prefix
REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}"
print(f"Connecting to Redis at: {REDIS_URL}")

# Connect with the Redis Python Client
client = redis.Redis.from_url(REDIS_URL)

client.ping()
print("✅ Successfully connected to Redis!")

print(client.dbsize())

client.set("hello", "world")
print(client.get("hello"))

print(client.dbsize())

obj = {
    "user": "john",
    "age": 45,
    "job": "dentist",
    "bio": "long form text of john's bio",
    "user_embedding": np.array([0.3, 0.4, -0.8], dtype=np.float32).tobytes() # cast vectors to bytes string
}

obj = {
    "user": "john",
    "age": 45,
    "job": "dentist",
    "bio": "long form text of john's bio",
    "user_embedding": np.array([0.3, 0.4, -0.8], dtype=np.float32).tobytes() # cast vectors to bytes string
}

client.hset("user:1", mapping=obj)
print(client.hgetall("user:1"))
print(client.dbsize())

# set a JSON obj
obj = {
    "user": "john",
    "metadata": {
        "age": 45,
        "job": "dentist",
    },
    "user_embedding": [0.3, 0.4, -0.8]
}

client.json().set("user:john", "$", obj)
print(client.json().get("user:john"))