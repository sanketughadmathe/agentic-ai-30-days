# Makefile
.PHONY: help build up down logs test clean

help:
	@echo "RAG API - Development Commands"
	@echo "==============================="
	@echo "make build    - Build Docker image"
	@echo "make up       - Start services"
	@echo "make down     - Stop services"
	@echo "make logs     - View logs"
	@echo "make restart  - Restart services"
	@echo "make test     - Run API tests"
	@echo "make clean    - Remove containers and images"

build:
	docker-compose build

up:
	docker-compose up -d
	@echo "API running at http://localhost:8000"
	@echo "Docs at http://localhost:8000/docs"

down:
	docker-compose down

logs:
	docker-compose logs -f

restart:
	docker-compose restart

test:
	@echo "Testing health endpoint..."
	@curl -s http://localhost:8000/health | python -m json.tool
	@echo "\nTesting query endpoint..."
	@curl -s -X POST http://localhost:8000/query \
		-H "Content-Type: application/json" \
		-d '{"question": "What is ReAct?"}' | python -m json.tool

clean:
	docker-compose down -v --rmi all
