from nv_ingest.api.v1.tasks import celery_app

# Only needed so celery autodiscovers tasks
# Run using:
#   celery -A app.celery_worker worker --loglevel=info
