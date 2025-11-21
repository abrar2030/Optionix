# Optionix Automation Scripts

This package contains a collection of automation scripts for the Optionix project. Below is an overview of each script, its purpose, usage, and key features.

---

## Scripts Overview

### 1. `comprehensive_test.sh`

- **Purpose:** Runs comprehensive tests across all components of Optionix
- **Usage:**

  ```bash
  ./comprehensive_test.sh
  ```

- **Features:**
  - Tests backend, frontend, mobile, blockchain, and AI components
  - Generates detailed test reports in HTML format
  - Provides test summary with pass/fail statistics

---

### 2. `ci_cd_pipeline.sh`

- **Purpose:** Automates the CI/CD process for Optionix
- **Usage:**

  ```bash
  ./ci_cd_pipeline.sh [--environment=ENV] [--skip-tests] [--skip-lint] [--skip-build] [--skip-deploy]
  ```

- **Features:**
  - Checks out code from repository
  - Runs linting and tests
  - Builds application and creates Docker containers
  - Deploys to specified environment (development, staging, production)

---

### 3. `env_validator.sh`

- **Purpose:** Validates and fixes the development environment setup
- **Usage:**

  ```bash
  ./env_validator.sh
  ```

- **Features:**
  - Checks for required tools and dependencies
  - Validates project structure
  - Creates missing scripts and configurations
  - Sets up Docker Compose configuration

---

### 4. `db_manager.sh`

- **Purpose:** Manages database operations for Optionix
- **Usage:**

  ```bash
  ./db_manager.sh [command] [options]
  ```

- **Commands:**
  - `setup` – Sets up database schema
  - `migrate` – Runs migrations
  - `seed` – Seeds database with test data
  - `backup` – Backs up the database
  - `restore` – Restores database from backup
  - `reset` – Drops and recreates the database

---

### 5. `performance_monitor.sh`

- **Purpose:** Monitors and reports on system performance
- **Usage:**

  ```bash
  ./performance_monitor.sh [command] [options]
  ```

- **Commands:**
  - `monitor` – Monitors CPU, memory, and disk usage
  - `benchmark` – Runs API benchmarks
  - `loadtest` – Performs load testing
  - `report` – Generates performance reports

---

### 6. `doc_generator.sh`

- **Purpose:** Generates documentation from code and project structure
- **Usage:**

  ```bash
  ./doc_generator.sh [command] [options]
  ```

- **Commands:**
  - `api` – Generates API documentation
  - `components` – Creates component documentation
  - `project` – Documents project structure and architecture
  - `changelog` – Generates changelog from git history
  - `all` – Runs all documentation commands

---

## Installation Instructions

1. Extract the zip file into your Optionix project root directory.
2. Make all scripts executable:

   ```bash
   chmod +x *.sh
   ```

3. Run the scripts as needed based on the usage instructions above.

> For more detailed information about each script, run the script with the `--help` flag.
