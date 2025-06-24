# Optionix Infrastructure - Main Terraform Configuration
# Financial Grade Security and Compliance

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.10"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }
  
  backend "s3" {
    bucket         = var.terraform_state_bucket
    key            = "optionix/terraform.tfstate"
    region         = var.aws_region
    encrypt        = true
    dynamodb_table = var.terraform_lock_table
    
    # Enhanced security for state management
    kms_key_id                = var.terraform_state_kms_key
    skip_credentials_validation = false
    skip_metadata_api_check     = false
    skip_region_validation      = false
  }
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
      },
      {
        Sid    = "Allow use of the key for Optionix services"
        Effect = "Allow"
        Principal = {
          AWS = [
            module.security.optionix_role_arn,
            module.compute.eks_node_role_arn
          ]
        }
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:ReEncrypt*",
          "kms:GenerateDataKey*",
          "kms:DescribeKey"
        ]
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

# Network Module
module "network" {
  source = "./modules/network"
  
  environment         = var.environment
  vpc_cidr           = var.vpc_cidr
  availability_zones = data.aws_availability_zones.available.names
  
  # Enhanced security settings
  enable_dns_hostnames = true
  enable_dns_support   = true
  enable_nat_gateway   = true
  single_nat_gateway   = var.environment == "dev" ? true : false
  enable_vpn_gateway   = false
  
  # Flow logs for compliance
  enable_flow_log                      = true
  flow_log_destination_type           = "cloud-watch-logs"
  create_flow_log_cloudwatch_log_group = true
  create_flow_log_cloudwatch_iam_role  = true
  
  # Network ACLs for additional security
  manage_default_network_acl = true
  default_network_acl_ingress = [
    {
      rule_no    = 100
      protocol   = "-1"
      rule_action = "allow"
      cidr_block = "10.0.0.0/8"
    }
  ]
  default_network_acl_egress = [
    {
      rule_no    = 100
      protocol   = "-1"
      rule_action = "allow"
      cidr_block = "0.0.0.0/0"
    }
  ]
  
  tags = {
    Terraform = "true"
    Environment = var.environment
  }
}

# Security Module
module "security" {
  source = "./modules/security"
  
  environment = var.environment
  vpc_id      = module.network.vpc_id
  
  # Security group configurations
  allowed_cidr_blocks = var.allowed_cidr_blocks
  
  # WAF configuration
  enable_waf = true
  waf_rules = [
    "AWSManagedRulesCommonRuleSet",
    "AWSManagedRulesKnownBadInputsRuleSet",
    "AWSManagedRulesSQLiRuleSet",
    "AWSManagedRulesLinuxRuleSet",
    "AWSManagedRulesUnixRuleSet"
  ]
  
  # GuardDuty
  enable_guardduty = true
  
  # Config
  enable_config = true
  
  # CloudTrail
  enable_cloudtrail = true
  cloudtrail_s3_bucket_name = "${var.project_name}-${var.environment}-cloudtrail-${random_id.bucket_suffix.hex}"
  
  # Secrets Manager
  kms_key_id = aws_kms_key.optionix_key.arn
  
  tags = {
    Terraform = "true"
    Environment = var.environment
  }
}

# Compute Module (EKS)
module "compute" {
  source = "./modules/compute"
  
  environment = var.environment
  vpc_id      = module.network.vpc_id
  subnet_ids  = module.network.private_subnets
  
  # EKS Configuration
  cluster_version = var.eks_cluster_version
  
  # Node groups configuration
  node_groups = {
    main = {
      desired_capacity = var.node_group_desired_capacity
      max_capacity     = var.node_group_max_capacity
      min_capacity     = var.node_group_min_capacity
      instance_types   = var.node_group_instance_types
      
      # Security settings
      ami_type        = "AL2_x86_64"
      capacity_type   = "ON_DEMAND"
      disk_size       = 50
      
      # Taints for financial workloads
      taints = [
        {
          key    = "financial-workload"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
      
      labels = {
        Environment = var.environment
        NodeGroup   = "main"
        Workload    = "financial"
      }
    }
  }
  
  # Enhanced security
  cluster_encryption_config = [
    {
      provider_key_arn = aws_kms_key.optionix_key.arn
      resources        = ["secrets"]
    }
  ]
  
  cluster_endpoint_private_access = true
  cluster_endpoint_public_access  = var.environment == "dev" ? true : false
  cluster_endpoint_public_access_cidrs = var.environment == "dev" ? ["0.0.0.0/0"] : var.allowed_cidr_blocks
  
  # Logging
  cluster_enabled_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]
  
  # IRSA (IAM Roles for Service Accounts)
  enable_irsa = true
  
  tags = {
    Terraform = "true"
    Environment = var.environment
  }
}

# Database Module
module "database" {
  source = "./modules/database"
  
  environment = var.environment
  vpc_id      = module.network.vpc_id
  subnet_ids  = module.network.database_subnets
  
  # RDS Configuration
  engine         = "mysql"
  engine_version = "8.0"
  instance_class = var.db_instance_class
  
  allocated_storage     = var.db_allocated_storage
  max_allocated_storage = var.db_max_allocated_storage
  storage_encrypted     = true
  kms_key_id           = aws_kms_key.optionix_key.arn
  
  # Database credentials
  db_name  = var.db_name
  username = var.db_username
  password = random_password.db_password.result
  
  # Security settings
  vpc_security_group_ids = [module.security.database_security_group_id]
  
  # Backup and maintenance
  backup_retention_period = 30
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  # Enhanced security
  deletion_protection = var.environment == "prod" ? true : false
  skip_final_snapshot = var.environment == "dev" ? true : false
  
  # Monitoring
  monitoring_interval = 60
  monitoring_role_arn = module.security.rds_monitoring_role_arn
  
  # Performance Insights
  performance_insights_enabled = true
  performance_insights_kms_key_id = aws_kms_key.optionix_key.arn
  
  tags = {
    Terraform = "true"
    Environment = var.environment
  }
}

# Storage Module
module "storage" {
  source = "./modules/storage"
  
  environment = var.environment
  
  # S3 buckets for different purposes
  buckets = {
    application_data = {
      versioning = true
      encryption = true
      kms_key_id = aws_kms_key.optionix_key.arn
      lifecycle_rules = [
        {
          id     = "transition_to_ia"
          status = "Enabled"
          transition = [
            {
              days          = 30
              storage_class = "STANDARD_IA"
            },
            {
              days          = 90
              storage_class = "GLACIER"
            }
          ]
        }
      ]
    }
    
    backup_data = {
      versioning = true
      encryption = true
      kms_key_id = aws_kms_key.optionix_key.arn
      lifecycle_rules = [
        {
          id     = "backup_lifecycle"
          status = "Enabled"
          transition = [
            {
              days          = 7
              storage_class = "GLACIER"
            },
            {
              days          = 365
              storage_class = "DEEP_ARCHIVE"
            }
          ]
        }
      ]
    }
    
    audit_logs = {
      versioning = true
      encryption = true
      kms_key_id = aws_kms_key.optionix_key.arn
      lifecycle_rules = [
        {
          id     = "audit_retention"
          status = "Enabled"
          expiration = [
            {
              days = 2555  # 7 years for financial compliance
            }
          ]
        }
      ]
    }
  }
  
  tags = {
    Terraform = "true"
    Environment = var.environment
  }
}

# Random ID for unique resource naming
resource "random_id" "bucket_suffix" {
  byte_length = 4
}

