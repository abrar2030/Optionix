# Optionix Deployment Guide

## Overview
This guide provides instructions for deploying the Optionix platform in various environments, from development to production.

## Prerequisites

### System Requirements
- Linux-based server (Ubuntu 20.04 LTS recommended)
- Docker and Docker Compose
- Nginx
- SSL certificates
- Domain name
- Sufficient disk space and memory

### Required Accounts
- Docker Hub account
- Cloud provider account (if deploying to cloud)
- Domain registrar
- SSL certificate provider

## Deployment Environments

### Development Environment
1. Clone the repository:
```bash
git clone https://github.com/your-org/optionix.git
cd optionix
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with development settings
```

3. Start services:
```bash
docker-compose -f docker-compose.dev.yml up -d
```

### Staging Environment
1. Set up staging server
2. Configure environment variables
3. Deploy using:
```bash
docker-compose -f docker-compose.staging.yml up -d
```

### Production Environment
1. Set up production server
2. Configure environment variables
3. Deploy using:
```bash
docker-compose -f docker-compose.prod.yml up -d
```

## Configuration

### Environment Variables
Required environment variables:
```
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=optionix
DB_USER=optionix
DB_PASSWORD=your_password

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# JWT
JWT_SECRET=your_secret
JWT_ALGORITHM=HS256

# Blockchain
BLOCKCHAIN_NETWORK=mainnet
CONTRACT_ADDRESS=your_address

# API
API_HOST=0.0.0.0
API_PORT=8000
```

### Nginx Configuration
Example Nginx configuration:
```nginx
server {
    listen 80;
    server_name optionix.com;

    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    location /api {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

### SSL Configuration
1. Obtain SSL certificates
2. Configure Nginx for HTTPS
3. Set up automatic certificate renewal

## Deployment Process

### Frontend Deployment
1. Build the application:
```bash
cd code/frontend
npm run build
```

2. Deploy to web server:
```bash
# Copy build files to web server
scp -r build/* user@server:/var/www/optionix
```

### Backend Deployment
1. Build Docker image:
```bash
cd code/backend
docker build -t optionix-backend .
```

2. Push to Docker Hub:
```bash
docker push your-org/optionix-backend
```

3. Deploy to server:
```bash
docker-compose up -d
```

### Database Deployment
1. Create database:
```bash
createdb optionix
```

2. Run migrations:
```bash
cd code/backend
alembic upgrade head
```

## Monitoring and Maintenance

### Logging
- Configure log rotation
- Set up log aggregation
- Monitor error logs

### Backup
1. Database backup:
```bash
pg_dump -U optionix optionix > backup.sql
```

2. Regular backup schedule:
```bash
# Add to crontab
0 0 * * * /path/to/backup.sh
```

### Updates
1. Pull latest changes:
```bash
git pull
```

2. Update dependencies:
```bash
npm install
pip install -r requirements.txt
```

3. Restart services:
```bash
docker-compose restart
```

## Troubleshooting

### Common Issues

1. Service not starting:
- Check logs: `docker-compose logs`
- Verify environment variables
- Check port conflicts

2. Database connection issues:
- Verify database is running
- Check credentials
- Test network connectivity

3. SSL certificate issues:
- Verify certificate installation
- Check certificate expiration
- Test HTTPS connection

## Security Considerations

### Network Security
- Configure firewall rules
- Enable HTTPS
- Set up rate limiting
- Implement IP whitelisting

### Application Security
- Regular security updates
- Vulnerability scanning
- Penetration testing
- Security audits

### Data Security
- Encrypt sensitive data
- Regular backups
- Access control
- Audit logging

## Scaling

### Horizontal Scaling
1. Add more servers
2. Configure load balancing
3. Set up database replication

### Vertical Scaling
1. Increase server resources
2. Optimize database
3. Implement caching

## Support

For deployment support:
- Check documentation
- Review deployment logs
- Contact operations team
- Submit support ticket 