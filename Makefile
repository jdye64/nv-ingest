# Makefile for managing Docker Compose with override

COMPOSE_FILES = -f docker-compose.yaml -f docker-compose.dev.yaml

.PHONY: up down restart

# Bring the stack down
down:
	docker-compose $(COMPOSE_FILES) down

# Bring the stack up with the override file
up:
	docker-compose $(COMPOSE_FILES) up -d

# Restart the stack
restart: down up
