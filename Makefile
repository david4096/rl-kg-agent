# Makefile for RL-KG-Agent development

.PHONY: install install-dev clean test lint format type-check quality help run-interactive run-train run-eval

# Colors for output
BLUE := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m

help: ## Show this help message
	@echo "$(BLUE)RL-KG-Agent Development Commands$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'

install: ## Install dependencies using uv
	@echo "$(BLUE)Installing dependencies...$(RESET)"
	uv sync

install-dev: ## Install all dependencies including development tools
	@echo "$(BLUE)Installing development dependencies...$(RESET)"
	uv sync --dev

clean: ## Clean build artifacts and caches
	@echo "$(BLUE)Cleaning build artifacts...$(RESET)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -delete
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete

test: ## Run tests with pytest
	@echo "$(BLUE)Running tests...$(RESET)"
	uv run pytest tests/ -v

test-cov: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(RESET)"
	uv run pytest tests/ --cov=src/rl_kg_agent --cov-report=term-missing

lint: ## Run linting with ruff
	@echo "$(BLUE)Running ruff linting...$(RESET)"
	uv run ruff check src/

lint-fix: ## Run linting with automatic fixes
	@echo "$(BLUE)Running ruff with automatic fixes...$(RESET)"
	uv run ruff check src/ --fix

format: ## Format code with ruff
	@echo "$(BLUE)Formatting code...$(RESET)"
	uv run ruff format src/

format-check: ## Check code formatting without making changes
	@echo "$(BLUE)Checking code formatting...$(RESET)"
	uv run ruff format --check src/

type-check: ## Run type checking with mypy
	@echo "$(BLUE)Running type checking...$(RESET)"
	uv run mypy src/

quality: lint format-check type-check ## Run all quality checks

quality-fix: lint-fix format type-check ## Run all quality checks with automatic fixes

# Development workflow commands
dev-setup: install-dev ## Complete development setup
	@echo "$(GREEN)Development setup complete!$(RESET)"
	@echo "Run '$(YELLOW)make help$(RESET)' to see available commands"

# Example run commands (requires TTL file)
run-interactive: ## Run interactive mode (requires TTL_FILE environment variable)
	@if [ -z "$(TTL_FILE)" ]; then \
		echo "$(RED)Error: TTL_FILE environment variable not set$(RESET)"; \
		echo "Usage: make run-interactive TTL_FILE=path/to/your/file.ttl"; \
		exit 1; \
	fi
	@echo "$(BLUE)Starting interactive mode...$(RESET)"
	uv run rl-kg-agent interactive --ttl-file $(TTL_FILE)

run-train: ## Run training (requires TTL_FILE environment variable)
	@if [ -z "$(TTL_FILE)" ]; then \
		echo "$(RED)Error: TTL_FILE environment variable not set$(RESET)"; \
		echo "Usage: make run-train TTL_FILE=path/to/your/file.ttl"; \
		exit 1; \
	fi
	@echo "$(BLUE)Starting training...$(RESET)"
	uv run rl-kg-agent train --ttl-file $(TTL_FILE) --dataset squad --episodes 1000

run-eval: ## Run evaluation (requires TTL_FILE and MODEL_PATH environment variables)
	@if [ -z "$(TTL_FILE)" ] || [ -z "$(MODEL_PATH)" ]; then \
		echo "$(RED)Error: TTL_FILE and MODEL_PATH environment variables required$(RESET)"; \
		echo "Usage: make run-eval TTL_FILE=path/to/file.ttl MODEL_PATH=path/to/model"; \
		exit 1; \
	fi
	@echo "$(BLUE)Starting evaluation...$(RESET)"
	uv run rl-kg-agent evaluate --ttl-file $(TTL_FILE) --model-path $(MODEL_PATH)

# Build and distribution
build: clean ## Build the package
	@echo "$(BLUE)Building package...$(RESET)"
	uv build

# CI/CD workflow
ci: quality test ## Run full CI pipeline (quality checks + tests)
	@echo "$(GREEN)CI pipeline completed successfully!$(RESET)"

# Shell access
shell: ## Activate uv shell environment
	@echo "$(BLUE)Activating uv shell...$(RESET)"
	uv shell