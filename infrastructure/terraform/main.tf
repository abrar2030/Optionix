# Optionix Infrastructure - Main Terraform Configuration
# Financial Grade Security and Compliance

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
  }

  # Backend configuration should be provided via backend config file or CLI
  # For local development, use local backend (comment out for remote):
  # backend "local" {
  #   path = "terraform.tfstate"
  # }
  
  # For production, use S3 backend (configure via backend.hcl):
  # terraform init -backend-config=backend.hcl
  # 
  # backend.hcl example:
  # bucket         = "optionix-terraform-state-ACCOUNTID"
  # key            = "optionix/terraform.tfstate"
  # region         = "us-west-2"
  # encrypt        = true
  # dynamodb_table = "optionix-terraform-locks"
  # kms_key_id     = "arn:aws:kms:REGION:ACCOUNT:key/KEY_ID"
}

# Configure AWS Provider with enhanced security
provider "aws" {
  region = var.aws_region

  # Security best practices
  skip_credentials_validation = false
  skip_metadata_api_check     = false
  skip_region_validation      = false

  default_tags {
    tags = {
      Project     = "Optionix"
      Environment = var.environment
      ManagedBy   = "Terraform"
      Owner       = var.owner
      CostCenter  = var.cost_center
      Compliance  = "Financial"
      DataClass   = "Sensitive"
    }
  }
}

# Data sources for existing resources
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}
data "aws_availability_zones" "available" {
  state = "available"
}

# Random password generation for secrets
resource "random_password" "db_password" {
  length  = 32
  special = true
  upper   = true
  lower   = true
  numeric = true
}

resource "random_password" "redis_password" {
  length  = 32
  special = false
  upper   = true
  lower   = true
  numeric = true
}

# Random ID for unique resource naming
resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# KMS Key for encryption
resource "aws_kms_key" "optionix_key" {
  description             = "Optionix encryption key for ${var.environment}"
  deletion_window_in_days = 7
  enable_key_rotation     = true

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "Enable IAM User Permissions"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action   = "kms:*"
        Resource = "*"
      }
    ]
  })

  tags = {
    Name = "optionix-${var.environment}-key"
  }
}

resource "aws_kms_alias" "optionix_key_alias" {
  name          = "alias/optionix-${var.environment}"
  target_key_id = aws_kms_key.optionix_key.key_id
}

# NOTE: The modules below require proper variable mappings
# Uncomment and configure as per your actual module implementations

# Network Module
module "network" {
  source = "./modules/network"

  environment = var.environment
  vpc_cidr    = var.vpc_cidr
}

# Security Module
module "security" {
  source = "./modules/security"

  environment = var.environment
  vpc_id      = module.network.vpc_id
}

# Compute Module
module "compute" {
  source = "./modules/compute"

  environment        = var.environment
  vpc_id             = module.network.vpc_id
  private_subnet_ids = module.network.private_subnet_ids
  app_name           = var.app_name
  security_group_ids = [module.security.compute_security_group_id]
}

# Database Module
module "database" {
  source = "./modules/database"

  environment = var.environment
  vpc_id      = module.network.vpc_id
  subnet_ids  = module.network.database_subnet_ids

  # Database configuration
  db_name  = var.db_name
  username = var.db_username
  password = random_password.db_password.result
}

# Storage Module
module "storage" {
  source = "./modules/storage"

  environment = var.environment
}
