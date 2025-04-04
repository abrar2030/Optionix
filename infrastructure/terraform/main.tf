provider "aws" {  
    region = "us-west-2"  
}  

resource "aws_eks_cluster" "derivatives_cluster" {  
    name     = "derivatives-trading-cluster"  
    role_arn = aws_iam_role.eks_cluster.arn  

    vpc_config {  
        subnet_ids = [aws_subnet.derivatives_subnet.id]  
    }  
}  

resource "aws_db_instance" "derivatives_db" {  
    allocated_storage    = 100  
    engine               = "postgres"  
    instance_class       = "db.m5.large"  
    username             = "quantadmin"  
    password             = var.db_password  
    publicly_accessible  = false  
}  