COMPOSE_BASELINE_DEV_FILES = -f docker-compose.yaml -f docker-compose.dev.yaml
COMPOSE_PROD_FILES = -f docker-compose.yaml -f docker-compose.prod.yaml
COMPOSE_SIMPLE_FILES = -f docker-compose.yaml -f docker-compose.simple.yaml

.PHONY: dev

# Create and start the dev environment
dev:
	echo "Starting dev nv-ingest environment"
	docker compose down \
		&& docker compose $(COMPOSE_BASELINE_DEV_FILES) up --build --wait --remove-orphans \
		&& docker compose -f logs nv-ingest-ms-runtime

# Create and start a production environment
prod:
	echo "Starting production nv-ingest environment"
	docker compose down \
		&& docker compose $(COMPOSE_PROD_FILES) up --wait --remove-orphans \
		&& docker compose -f logs nv-ingest-ms-runtime

# Create and start a "simple" environment
simple:
	echo "Starting 'simple' nv-ingest environment"
	docker compose down \
		&& docker compose $(COMPOSE_SIMPLE_FILES) up -- wait --remove-orphans \
		&& docker compose -f logs nv-ingest-ms-runtime
