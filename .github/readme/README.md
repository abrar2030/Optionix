# GitHub Workflows Directory

## Overview

The `.github` directory contains GitHub-specific configurations and workflows that automate various development processes for the Optionix project. This directory is essential for maintaining continuous integration and continuous deployment (CI/CD) pipelines, ensuring code quality, and streamlining the development workflow.

## Directory Structure

```
.github/
└── workflows/
    └── ci-cd.yml
```

## Workflows

### CI/CD Pipeline (`workflows/ci-cd.yml`)

The CI/CD pipeline is configured to automatically run on push events to the `main`, `master`, and `develop` branches, as well as on pull requests targeting these branches. This workflow ensures that code changes are properly tested and validated before being merged or deployed.

The pipeline consists of the following jobs:

#### Backend Testing

- **Environment**: Runs on Ubuntu latest
- **Setup**: Uses Python 3.10
- **Process**:
  1. Installs dependencies from `code/backend/requirements.txt`
  2. Runs pytest on the backend code

#### Frontend Testing

- **Environment**: Runs on Ubuntu latest
- **Setup**: Uses Node.js 18 with npm caching
- **Process**:
  1. Installs dependencies using `npm ci` in the `code/frontend` directory
  2. Runs the frontend test suite

#### Docker Build

- **Environment**: Runs on Ubuntu latest
- **Setup**: Uses Docker Buildx
- **Process**:
  1. Builds a Docker image from the `infrastructure` directory
  2. Tags the image as `optionix:latest`

#### Build and Deploy

- **Trigger**: Only runs on push events to `main` or `master` branches
- **Dependencies**: Requires successful completion of backend tests, frontend tests, and Docker build
- **Environment**: Runs on Ubuntu latest
- **Process**:
  1. Sets up Node.js 18
  2. Builds the frontend by installing dependencies and running the build script
  3. Sets up Python 3.10
  4. Installs backend dependencies
  5. Includes placeholder for deployment steps (to be customized based on deployment strategy)

## Usage

The workflows in this directory run automatically based on their configured triggers. No manual intervention is required for normal operation. However, developers should be aware of these workflows to understand how their code is being tested and deployed.

## Best Practices

1. **Workflow Modifications**: When modifying the CI/CD workflow, ensure that all necessary tests are still being run and that the deployment process remains secure and reliable.

2. **Secrets Management**: Any sensitive information required by the workflows (such as API keys or deployment credentials) should be stored as GitHub Secrets and referenced in the workflow files.

3. **Branch Protection**: Consider enabling branch protection rules for the main branches to ensure that code cannot be merged without passing the CI/CD pipeline.

## Contributing

When contributing to the Optionix project, be aware that your code will be automatically tested by these workflows. Ensure that your changes pass all tests locally before pushing to avoid unnecessary workflow failures.

## Troubleshooting

If a workflow fails:

1. Check the workflow logs in the GitHub Actions tab of the repository
2. Verify that all dependencies are correctly specified
3. Ensure that tests are properly configured and passing locally
4. Check for any environment-specific issues that might be causing failures in the CI environment but not locally
