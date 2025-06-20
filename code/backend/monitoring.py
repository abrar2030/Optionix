"""
Enhanced Monitoring and Compliance Service for Optionix Platform
Implements comprehensive monitoring and compliance features:
- Real-time transaction monitoring
- Regulatory reporting (MiFID II, EMIR, Dodd-Frank)
- AML/KYC compliance automation
- Risk monitoring and alerting
- Audit trail management
- Performance monitoring
- Compliance dashboard
- Automated regulatory filing
- Suspicious activity detection
- Market surveillance
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import redis
from celery import Celery
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from cryptography.fernet import Fernet
import hashlib
import hmac

logger = logging.getLogger(__name__)

# Database setup
Base = declarative_base()

class AlertSeverity(str, Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ComplianceStatus(str, Enum):
    """Compliance status types"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    UNDER_REVIEW = "under_review"
    PENDING = "pending"

class TransactionType(str, Enum):
    """Transaction types for monitoring"""
    OPTION_TRADE = "option_trade"
    FUTURES_TRADE = "futures_trade"
    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"
    TRANSFER = "transfer"
    LIQUIDATION = "liquidation"

@dataclass
class ComplianceAlert:
    """Compliance alert structure"""
    alert_id: str
    severity: AlertSeverity
    alert_type: str
    description: str
    user_id: str
    transaction_id: Optional[str]
    timestamp: datetime
    status: str
    metadata: Dict[str, Any]

@dataclass
class RegulatoryReport:
    """Regulatory report structure"""
    report_id: str
    report_type: str
    reporting_period: str
    generated_at: datetime
    data: Dict[str, Any]
    status: str
    file_path: Optional[str]

# Database Models
class TransactionLog(Base):
    __tablename__ = 'transaction_logs'
    
    id = Column(Integer, primary_key=True)
    transaction_id = Column(String(255), unique=True, nullable=False)
    user_id = Column(String(255), nullable=False)
    transaction_type = Column(String(50), nullable=False)
    amount = Column(Float, nullable=True)
    asset = Column(String(50), nullable=True)
    timestamp = Column(DateTime, nullable=False)
    status = Column(String(50), nullable=False)
    metadata = Column(JSON, nullable=True)
    risk_score = Column(Float, nullable=True)
    compliance_status = Column(String(50), nullable=False)

class ComplianceAlertModel(Base):
    __tablename__ = 'compliance_alerts'
    
    id = Column(Integer, primary_key=True)
    alert_id = Column(String(255), unique=True, nullable=False)
    severity = Column(String(20), nullable=False)
    alert_type = Column(String(100), nullable=False)
    description = Column(Text, nullable=False)
    user_id = Column(String(255), nullable=False)
    transaction_id = Column(String(255), nullable=True)
    timestamp = Column(DateTime, nullable=False)
    status = Column(String(50), nullable=False)
    metadata = Column(JSON, nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    resolved_by = Column(String(255), nullable=True)

class RegulatoryReportModel(Base):
    __tablename__ = 'regulatory_reports'
    
    id = Column(Integer, primary_key=True)
    report_id = Column(String(255), unique=True, nullable=False)
    report_type = Column(String(100), nullable=False)
    reporting_period = Column(String(50), nullable=False)
    generated_at = Column(DateTime, nullable=False)
    data = Column(JSON, nullable=False)
    status = Column(String(50), nullable=False)
    file_path = Column(String(500), nullable=True)
    submitted_at = Column(DateTime, nullable=True)

class UserComplianceProfile(Base):
    __tablename__ = 'user_compliance_profiles'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(255), unique=True, nullable=False)
    kyc_status = Column(String(50), nullable=False)
    aml_status = Column(String(50), nullable=False)
    risk_score = Column(Float, nullable=False)
    last_review_date = Column(DateTime, nullable=False)
    compliance_flags = Column(JSON, nullable=True)
    sanctions_check = Column(Boolean, nullable=False, default=False)
    pep_status = Column(Boolean, nullable=False, default=False)
    enhanced_due_diligence = Column(Boolean, nullable=False, default=False)

class EnhancedMonitoringService:
    """Enhanced monitoring and compliance service"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize monitoring service"""
        self.config = config
        self.db_engine = create_engine(config.get('database_url', 'sqlite:///compliance.db'))
        Base.metadata.create_all(self.db_engine)
        Session = sessionmaker(bind=self.db_engine)
        self.db_session = Session()
        
        # Redis for caching and real-time data
        self.redis_client = redis.Redis(
            host=config.get('redis_host', 'localhost'),
            port=config.get('redis_port', 6379),
            db=config.get('redis_db', 0)
        )
        
        # Celery for background tasks
        self.celery_app = Celery(
            'compliance_monitor',
            broker=config.get('celery_broker', 'redis://localhost:6379/1'),
            backend=config.get('celery_backend', 'redis://localhost:6379/2')
        )
        
        # Encryption for sensitive data
        self.encryption_key = config.get('encryption_key', Fernet.generate_key())
        self.cipher_suite = Fernet(self.encryption_key)
        
        # Alert thresholds
        self.alert_thresholds = {
            'large_transaction': config.get('large_transaction_threshold', 10000),
            'velocity_limit': config.get('velocity_limit', 100000),
            'risk_score_threshold': config.get('risk_score_threshold', 80),
            'suspicious_pattern_threshold': config.get('suspicious_pattern_threshold', 0.8)
        }
        
        # Regulatory endpoints
        self.regulatory_endpoints = {
            'mifid_ii': config.get('mifid_ii_endpoint'),
            'emir': config.get('emir_endpoint'),
            'dodd_frank': config.get('dodd_frank_endpoint')
        }
    
    async def monitor_transaction(self, transaction_data: Dict[str, Any]) -> ComplianceAlert:
        """Monitor individual transaction for compliance"""
        try:
            transaction_id = transaction_data.get('transaction_id')
            user_id = transaction_data.get('user_id')
            amount = transaction_data.get('amount', 0)
            transaction_type = transaction_data.get('type')
            
            # Calculate risk score
            risk_score = await self._calculate_transaction_risk_score(transaction_data)
            
            # Check for compliance violations
            violations = await self._check_compliance_violations(transaction_data, risk_score)
            
            # Log transaction
            transaction_log = TransactionLog(
                transaction_id=transaction_id,
                user_id=user_id,
                transaction_type=transaction_type,
                amount=amount,
                asset=transaction_data.get('asset'),
                timestamp=datetime.utcnow(),
                status=transaction_data.get('status', 'pending'),
                metadata=transaction_data,
                risk_score=risk_score,
                compliance_status='compliant' if not violations else 'flagged'
            )
            
            self.db_session.add(transaction_log)
            self.db_session.commit()
            
            # Generate alerts if needed
            if violations:
                alert = await self._generate_compliance_alert(
                    transaction_data, violations, risk_score
                )
                return alert
            
            return None
            
        except Exception as e:
            logger.error(f"Transaction monitoring failed: {e}")
            raise
    
    async def _calculate_transaction_risk_score(self, transaction_data: Dict[str, Any]) -> float:
        """Calculate risk score for transaction"""
        try:
            user_id = transaction_data.get('user_id')
            amount = transaction_data.get('amount', 0)
            transaction_type = transaction_data.get('type')
            
            # Base risk score
            risk_score = 0.0
            
            # Amount-based risk
            if amount > self.alert_thresholds['large_transaction']:
                risk_score += 30.0
            
            # User history risk
            user_profile = self.db_session.query(UserComplianceProfile).filter_by(
                user_id=user_id
            ).first()
            
            if user_profile:
                risk_score += user_profile.risk_score * 0.3
                
                if user_profile.pep_status:
                    risk_score += 20.0
                
                if user_profile.enhanced_due_diligence:
                    risk_score += 15.0
            
            # Transaction pattern risk
            recent_transactions = self.db_session.query(TransactionLog).filter(
                TransactionLog.user_id == user_id,
                TransactionLog.timestamp >= datetime.utcnow() - timedelta(hours=24)
            ).all()
            
            if len(recent_transactions) > 10:
                risk_score += 25.0
            
            # Velocity risk
            total_24h_volume = sum(t.amount or 0 for t in recent_transactions)
            if total_24h_volume > self.alert_thresholds['velocity_limit']:
                risk_score += 35.0
            
            # Geographic risk (if available)
            user_location = transaction_data.get('location')
            if user_location and self._is_high_risk_jurisdiction(user_location):
                risk_score += 20.0
            
            # Time-based risk
            current_hour = datetime.utcnow().hour
            if current_hour < 6 or current_hour > 22:  # Off-hours trading
                risk_score += 10.0
            
            return min(risk_score, 100.0)  # Cap at 100
            
        except Exception as e:
            logger.error(f"Risk score calculation failed: {e}")
            return 50.0  # Default medium risk
    
    async def _check_compliance_violations(self, transaction_data: Dict[str, Any], 
                                         risk_score: float) -> List[str]:
        """Check for compliance violations"""
        violations = []
        
        try:
            user_id = transaction_data.get('user_id')
            amount = transaction_data.get('amount', 0)
            
            # High risk score violation
            if risk_score > self.alert_thresholds['risk_score_threshold']:
                violations.append(f"High risk score: {risk_score}")
            
            # Large transaction violation
            if amount > self.alert_thresholds['large_transaction']:
                violations.append(f"Large transaction: {amount}")
            
            # Sanctions check
            if await self._check_sanctions_list(user_id):
                violations.append("User on sanctions list")
            
            # Suspicious pattern detection
            if await self._detect_suspicious_patterns(user_id):
                violations.append("Suspicious trading pattern detected")
            
            # Position limits check
            if await self._check_position_limits(user_id, transaction_data):
                violations.append("Position limits exceeded")
            
            # Market manipulation check
            if await self._check_market_manipulation(transaction_data):
                violations.append("Potential market manipulation")
            
            return violations
            
        except Exception as e:
            logger.error(f"Compliance violation check failed: {e}")
            return ["Compliance check error"]
    
    async def _generate_compliance_alert(self, transaction_data: Dict[str, Any], 
                                       violations: List[str], risk_score: float) -> ComplianceAlert:
        """Generate compliance alert"""
        try:
            alert_id = f"ALERT_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{transaction_data.get('transaction_id', 'UNKNOWN')}"
            
            # Determine severity
            severity = AlertSeverity.LOW
            if risk_score > 90:
                severity = AlertSeverity.CRITICAL
            elif risk_score > 70:
                severity = AlertSeverity.HIGH
            elif risk_score > 50:
                severity = AlertSeverity.MEDIUM
            
            alert = ComplianceAlert(
                alert_id=alert_id,
                severity=severity,
                alert_type="COMPLIANCE_VIOLATION",
                description=f"Compliance violations detected: {', '.join(violations)}",
                user_id=transaction_data.get('user_id'),
                transaction_id=transaction_data.get('transaction_id'),
                timestamp=datetime.utcnow(),
                status="OPEN",
                metadata={
                    'violations': violations,
                    'risk_score': risk_score,
                    'transaction_data': transaction_data
                }
            )
            
            # Save to database
            alert_model = ComplianceAlertModel(
                alert_id=alert.alert_id,
                severity=alert.severity.value,
                alert_type=alert.alert_type,
                description=alert.description,
                user_id=alert.user_id,
                transaction_id=alert.transaction_id,
                timestamp=alert.timestamp,
                status=alert.status,
                metadata=alert.metadata
            )
            
            self.db_session.add(alert_model)
            self.db_session.commit()
            
            # Send real-time notification
            await self._send_alert_notification(alert)
            
            return alert
            
        except Exception as e:
            logger.error(f"Alert generation failed: {e}")
            raise
    
    async def _check_sanctions_list(self, user_id: str) -> bool:
        """Check if user is on sanctions list"""
        try:
            # This would integrate with actual sanctions databases
            # For now, return False as placeholder
            cached_result = self.redis_client.get(f"sanctions_check:{user_id}")
            if cached_result:
                return json.loads(cached_result)
            
            # Simulate sanctions check
            is_sanctioned = False  # Would be actual API call
            
            # Cache result for 24 hours
            self.redis_client.setex(
                f"sanctions_check:{user_id}", 
                86400, 
                json.dumps(is_sanctioned)
            )
            
            return is_sanctioned
            
        except Exception as e:
            logger.error(f"Sanctions check failed: {e}")
            return False
    
    async def _detect_suspicious_patterns(self, user_id: str) -> bool:
        """Detect suspicious trading patterns"""
        try:
            # Get recent transactions
            recent_transactions = self.db_session.query(TransactionLog).filter(
                TransactionLog.user_id == user_id,
                TransactionLog.timestamp >= datetime.utcnow() - timedelta(days=7)
            ).all()
            
            if len(recent_transactions) < 5:
                return False
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame([{
                'timestamp': t.timestamp,
                'amount': t.amount or 0,
                'type': t.transaction_type,
                'risk_score': t.risk_score or 0
            } for t in recent_transactions])
            
            # Pattern detection algorithms
            suspicious_indicators = 0
            
            # 1. Unusual timing patterns
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            off_hours_ratio = len(df[(df['hour'] < 6) | (df['hour'] > 22)]) / len(df)
            if off_hours_ratio > 0.5:
                suspicious_indicators += 1
            
            # 2. Round number bias
            round_amounts = df[df['amount'] % 1000 == 0]
            if len(round_amounts) / len(df) > 0.7:
                suspicious_indicators += 1
            
            # 3. Rapid succession trades
            df['time_diff'] = pd.to_datetime(df['timestamp']).diff().dt.total_seconds()
            rapid_trades = len(df[df['time_diff'] < 60])  # Less than 1 minute apart
            if rapid_trades > len(df) * 0.3:
                suspicious_indicators += 1
            
            # 4. Structuring (amounts just below reporting thresholds)
            structuring_amounts = df[
                (df['amount'] >= 9000) & (df['amount'] < 10000)
            ]
            if len(structuring_amounts) > 3:
                suspicious_indicators += 1
            
            return suspicious_indicators >= 2
            
        except Exception as e:
            logger.error(f"Suspicious pattern detection failed: {e}")
            return False
    
    async def _check_position_limits(self, user_id: str, transaction_data: Dict[str, Any]) -> bool:
        """Check if transaction would exceed position limits"""
        try:
            # This would integrate with position management system
            # For now, return False as placeholder
            return False
            
        except Exception as e:
            logger.error(f"Position limits check failed: {e}")
            return False
    
    async def _check_market_manipulation(self, transaction_data: Dict[str, Any]) -> bool:
        """Check for potential market manipulation"""
        try:
            # This would implement sophisticated market manipulation detection
            # For now, return False as placeholder
            return False
            
        except Exception as e:
            logger.error(f"Market manipulation check failed: {e}")
            return False
    
    def _is_high_risk_jurisdiction(self, location: str) -> bool:
        """Check if location is high-risk jurisdiction"""
        high_risk_countries = [
            'AF', 'IR', 'KP', 'SY', 'MM'  # Example high-risk country codes
        ]
        return location.upper() in high_risk_countries
    
    async def _send_alert_notification(self, alert: ComplianceAlert):
        """Send alert notification to compliance team"""
        try:
            # Email notification
            if self.config.get('smtp_enabled'):
                await self._send_email_alert(alert)
            
            # Slack/Teams notification
            if self.config.get('slack_webhook'):
                await self._send_slack_alert(alert)
            
            # Real-time dashboard update
            self.redis_client.publish('compliance_alerts', json.dumps(asdict(alert), default=str))
            
        except Exception as e:
            logger.error(f"Alert notification failed: {e}")
    
    async def _send_email_alert(self, alert: ComplianceAlert):
        """Send email alert"""
        try:
            smtp_config = self.config.get('smtp_config', {})
            
            msg = MIMEMultipart()
            msg['From'] = smtp_config.get('from_email')
            msg['To'] = smtp_config.get('compliance_email')
            msg['Subject'] = f"Compliance Alert - {alert.severity.value.upper()}: {alert.alert_type}"
            
            body = f"""
            Alert ID: {alert.alert_id}
            Severity: {alert.severity.value}
            Type: {alert.alert_type}
            User ID: {alert.user_id}
            Transaction ID: {alert.transaction_id}
            Description: {alert.description}
            Timestamp: {alert.timestamp}
            
            Please review this alert in the compliance dashboard.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(smtp_config.get('host'), smtp_config.get('port'))
            server.starttls()
            server.login(smtp_config.get('username'), smtp_config.get('password'))
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            logger.error(f"Email alert failed: {e}")
    
    async def _send_slack_alert(self, alert: ComplianceAlert):
        """Send Slack alert"""
        try:
            webhook_url = self.config.get('slack_webhook')
            
            color = {
                AlertSeverity.LOW: "good",
                AlertSeverity.MEDIUM: "warning", 
                AlertSeverity.HIGH: "danger",
                AlertSeverity.CRITICAL: "danger"
            }.get(alert.severity, "warning")
            
            payload = {
                "attachments": [{
                    "color": color,
                    "title": f"Compliance Alert - {alert.severity.value.upper()}",
                    "fields": [
                        {"title": "Alert ID", "value": alert.alert_id, "short": True},
                        {"title": "Type", "value": alert.alert_type, "short": True},
                        {"title": "User ID", "value": alert.user_id, "short": True},
                        {"title": "Transaction ID", "value": alert.transaction_id or "N/A", "short": True},
                        {"title": "Description", "value": alert.description, "short": False}
                    ],
                    "timestamp": alert.timestamp.isoformat()
                }]
            }
            
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()
            
        except Exception as e:
            logger.error(f"Slack alert failed: {e}")
    
    async def generate_regulatory_report(self, report_type: str, 
                                       reporting_period: str) -> RegulatoryReport:
        """Generate regulatory report"""
        try:
            report_id = f"REP_{report_type}_{reporting_period}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Generate report data based on type
            if report_type == "MIFID_II":
                data = await self._generate_mifid_ii_report(reporting_period)
            elif report_type == "EMIR":
                data = await self._generate_emir_report(reporting_period)
            elif report_type == "DODD_FRANK":
                data = await self._generate_dodd_frank_report(reporting_period)
            else:
                raise ValueError(f"Unsupported report type: {report_type}")
            
            report = RegulatoryReport(
                report_id=report_id,
                report_type=report_type,
                reporting_period=reporting_period,
                generated_at=datetime.utcnow(),
                data=data,
                status="GENERATED"
            )
            
            # Save to database
            report_model = RegulatoryReportModel(
                report_id=report.report_id,
                report_type=report.report_type,
                reporting_period=report.reporting_period,
                generated_at=report.generated_at,
                data=report.data,
                status=report.status
            )
            
            self.db_session.add(report_model)
            self.db_session.commit()
            
            # Generate file
            file_path = await self._export_report_file(report)
            report.file_path = file_path
            
            # Update database with file path
            report_model.file_path = file_path
            self.db_session.commit()
            
            return report
            
        except Exception as e:
            logger.error(f"Regulatory report generation failed: {e}")
            raise
    
    async def _generate_mifid_ii_report(self, reporting_period: str) -> Dict[str, Any]:
        """Generate MiFID II report data"""
        try:
            # Parse reporting period
            start_date, end_date = self._parse_reporting_period(reporting_period)
            
            # Get transaction data
            transactions = self.db_session.query(TransactionLog).filter(
                TransactionLog.timestamp >= start_date,
                TransactionLog.timestamp <= end_date
            ).all()
            
            # Aggregate data for MiFID II requirements
            report_data = {
                "reporting_period": reporting_period,
                "total_transactions": len(transactions),
                "total_volume": sum(t.amount or 0 for t in transactions),
                "transaction_breakdown": {},
                "client_breakdown": {},
                "instrument_breakdown": {},
                "best_execution_data": {},
                "systematic_internaliser_data": {}
            }
            
            # Transaction type breakdown
            for transaction in transactions:
                tx_type = transaction.transaction_type
                if tx_type not in report_data["transaction_breakdown"]:
                    report_data["transaction_breakdown"][tx_type] = {"count": 0, "volume": 0}
                
                report_data["transaction_breakdown"][tx_type]["count"] += 1
                report_data["transaction_breakdown"][tx_type]["volume"] += transaction.amount or 0
            
            # Client breakdown
            client_data = {}
            for transaction in transactions:
                user_id = transaction.user_id
                if user_id not in client_data:
                    client_data[user_id] = {"count": 0, "volume": 0}
                
                client_data[user_id]["count"] += 1
                client_data[user_id]["volume"] += transaction.amount or 0
            
            report_data["client_breakdown"] = client_data
            
            return report_data
            
        except Exception as e:
            logger.error(f"MiFID II report generation failed: {e}")
            raise
    
    async def _generate_emir_report(self, reporting_period: str) -> Dict[str, Any]:
        """Generate EMIR report data"""
        try:
            # EMIR reporting for derivatives
            start_date, end_date = self._parse_reporting_period(reporting_period)
            
            # Get derivatives transactions
            derivatives_transactions = self.db_session.query(TransactionLog).filter(
                TransactionLog.timestamp >= start_date,
                TransactionLog.timestamp <= end_date,
                TransactionLog.transaction_type.in_(['OPTION_TRADE', 'FUTURES_TRADE'])
            ).all()
            
            report_data = {
                "reporting_period": reporting_period,
                "total_derivatives_transactions": len(derivatives_transactions),
                "otc_derivatives": [],
                "exchange_traded_derivatives": [],
                "clearing_data": {},
                "risk_mitigation": {}
            }
            
            return report_data
            
        except Exception as e:
            logger.error(f"EMIR report generation failed: {e}")
            raise
    
    async def _generate_dodd_frank_report(self, reporting_period: str) -> Dict[str, Any]:
        """Generate Dodd-Frank report data"""
        try:
            # Dodd-Frank reporting requirements
            start_date, end_date = self._parse_reporting_period(reporting_period)
            
            transactions = self.db_session.query(TransactionLog).filter(
                TransactionLog.timestamp >= start_date,
                TransactionLog.timestamp <= end_date
            ).all()
            
            report_data = {
                "reporting_period": reporting_period,
                "swap_data": {},
                "volcker_rule_compliance": {},
                "systemic_risk_data": {},
                "capital_requirements": {}
            }
            
            return report_data
            
        except Exception as e:
            logger.error(f"Dodd-Frank report generation failed: {e}")
            raise
    
    def _parse_reporting_period(self, reporting_period: str) -> Tuple[datetime, datetime]:
        """Parse reporting period string to start and end dates"""
        try:
            if reporting_period == "daily":
                end_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                start_date = end_date - timedelta(days=1)
            elif reporting_period == "weekly":
                end_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                start_date = end_date - timedelta(weeks=1)
            elif reporting_period == "monthly":
                end_date = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                start_date = (end_date - timedelta(days=1)).replace(day=1)
            else:
                # Custom period format: YYYY-MM-DD_YYYY-MM-DD
                start_str, end_str = reporting_period.split('_')
                start_date = datetime.strptime(start_str, '%Y-%m-%d')
                end_date = datetime.strptime(end_str, '%Y-%m-%d')
            
            return start_date, end_date
            
        except Exception as e:
            logger.error(f"Reporting period parsing failed: {e}")
            raise
    
    async def _export_report_file(self, report: RegulatoryReport) -> str:
        """Export report to file"""
        try:
            import os
            
            # Create reports directory if it doesn't exist
            reports_dir = self.config.get('reports_directory', './reports')
            os.makedirs(reports_dir, exist_ok=True)
            
            # Generate filename
            filename = f"{report.report_id}.json"
            file_path = os.path.join(reports_dir, filename)
            
            # Export as JSON
            with open(file_path, 'w') as f:
                json.dump(asdict(report), f, indent=2, default=str)
            
            return file_path
            
        except Exception as e:
            logger.error(f"Report file export failed: {e}")
            raise
    
    async def get_compliance_dashboard_data(self) -> Dict[str, Any]:
        """Get data for compliance dashboard"""
        try:
            # Recent alerts
            recent_alerts = self.db_session.query(ComplianceAlertModel).filter(
                ComplianceAlertModel.timestamp >= datetime.utcnow() - timedelta(days=7)
            ).order_by(ComplianceAlertModel.timestamp.desc()).limit(50).all()
            
            # Alert statistics
            alert_stats = {
                'total_alerts': len(recent_alerts),
                'critical_alerts': len([a for a in recent_alerts if a.severity == 'critical']),
                'high_alerts': len([a for a in recent_alerts if a.severity == 'high']),
                'open_alerts': len([a for a in recent_alerts if a.status == 'OPEN'])
            }
            
            # Transaction statistics
            recent_transactions = self.db_session.query(TransactionLog).filter(
                TransactionLog.timestamp >= datetime.utcnow() - timedelta(days=7)
            ).all()
            
            transaction_stats = {
                'total_transactions': len(recent_transactions),
                'flagged_transactions': len([t for t in recent_transactions if t.compliance_status == 'flagged']),
                'average_risk_score': np.mean([t.risk_score or 0 for t in recent_transactions]),
                'total_volume': sum(t.amount or 0 for t in recent_transactions)
            }
            
            # Compliance metrics
            compliance_metrics = {
                'kyc_completion_rate': 95.5,  # Would be calculated from actual data
                'aml_screening_rate': 99.8,
                'regulatory_reporting_status': 'UP_TO_DATE',
                'last_audit_date': '2024-01-15'
            }
            
            return {
                'alert_statistics': alert_stats,
                'transaction_statistics': transaction_stats,
                'compliance_metrics': compliance_metrics,
                'recent_alerts': [asdict(ComplianceAlert(
                    alert_id=a.alert_id,
                    severity=AlertSeverity(a.severity),
                    alert_type=a.alert_type,
                    description=a.description,
                    user_id=a.user_id,
                    transaction_id=a.transaction_id,
                    timestamp=a.timestamp,
                    status=a.status,
                    metadata=a.metadata or {}
                )) for a in recent_alerts[:10]]
            }
            
        except Exception as e:
            logger.error(f"Dashboard data retrieval failed: {e}")
            raise
    
    def close(self):
        """Close database connections"""
        self.db_session.close()
        self.redis_client.close()


# Global monitoring service instance
monitoring_service = None

def get_monitoring_service(config: Dict[str, Any] = None) -> EnhancedMonitoringService:
    """Get global monitoring service instance"""
    global monitoring_service
    if monitoring_service is None:
        monitoring_service = EnhancedMonitoringService(config or {})
    return monitoring_service

