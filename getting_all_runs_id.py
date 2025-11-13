from apify_client import ApifyClient
import os

# Initialize the client
client = ApifyClient(os.getenv("APIFY_API_TOKEN"))

# Replace with your actor ID or name (e.g. "username~actor-name")
ACTOR_ID = "aYG0l9s7dbB7j3gbS"

actor_client = client.actor(ACTOR_ID)
runs_client = actor_client.runs()

# List runs (returns up to 1000 by default)
actor_runs = runs_client.list(limit=10, desc=True).items


print(actor_runs)