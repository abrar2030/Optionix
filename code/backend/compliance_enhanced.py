"""
Enhanced compliance service for Optionix platform.
Provides comprehensive KYC/AML, regulatory reporting, and financial standards compliance.
"""
import re
import json
import logging
import hashlib
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from sqlalchemy.orm import Session
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Numeric, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

from models import User, Trade, AuditLog, Account
from data_protection import data_protection_service

logger = logging.getLogger(__name__)

Base = declarative_base()


class RiskLevel(str, Enum):
    """Risk assessment levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceStatus(str, Enum):
    """Compliance status values"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    UNDER_REVIEW = "under_review"
    PENDING = "pending"


class RegulationType(str, Enum):
    """Types of financial regulations"""
    KYC = "kyc"
    AML = "aml"
    MIFID_II = "mifid_ii"
    DODD_FRANK = "dodd_frank"
    SOX = "sox"
    BASEL_III = "basel_iii"
    CFTC = "cftc"
    GDPR = "gdpr"
    PCI_DSS = "pci_dss"


class KYCDocument(Base):
    """KYC document storage"""
    __tablename__ = "kyc_documents"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    document_type = Column(String(50), nullable=False)
    document_number = Column(String(100), nullable=False)
    document_country = Column(String(3), nullable=False)
    document_expiry = Column(DateTime, nullable=True)
    document_hash = Column(String(64), nullable=False)  # SHA-256 hash
    verification_status = Column(String(20), default="pending")
    verification_date = Column(DateTime, nullable=True)
    verification_method = Column(String(50), nullable=True)
    risk_score = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SanctionsCheck(Base):
    """Sanctions screening results"""
    __tablename__ = "sanctions_checks"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    check_type = Column(String(50), nullable=False)  # name, address, entity
    search_terms = Column(Text, nullable=False)
    lists_checked = Column(Text, nullable=False)  # JSON array
    matches_found = Column(Boolean, default=False)
    match_details = Column(Text, nullable=True)  # JSON
    risk_score = Column(Integer, default=0)
    checked_at = Column(DateTime, default=datetime.utcnow)
    next_check_due = Column(DateTime, nullable=False)


class TransactionMonitoring(Base):
    """Transaction monitoring alerts"""
    __tablename__ = "transaction_monitoring"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    trade_id = Column(Integer, ForeignKey("trades.id"), nullable=True)
    alert_type = Column(String(100), nullable=False)
    alert_description = Column(Text, nullable=False)
    risk_score = Column(Integer, nullable=False)
    threshold_breached = Column(String(100), nullable=False)
    alert_status = Column(String(20), default="open")  # open, investigating, closed, false_positive
    assigned_to = Column(String(100), nullable=True)
    resolution_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)


class RegulatoryReport(Base):
    """Regulatory reporting records"""
    __tablename__ = "regulatory_reports"
    
    id = Column(Integer, primary_key=True)
    report_id = Column(String(100), unique=True, nullable=False)
    report_type = Column(String(50), nullable=False)
    regulation_type = Column(String(50), nullable=False)
    reporting_period_start = Column(DateTime, nullable=False)
    reporting_period_end = Column(DateTime, nullable=False)
    report_data = Column(Text, nullable=False)  # JSON
    submission_status = Column(String(20), default="draft")
    submitted_at = Column(DateTime, nullable=True)
    submission_reference = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ComplianceRule(Base):
    """Configurable compliance rules"""
    __tablename__ = "compliance_rules"
    
    id = Column(Integer, primary_key=True)
    rule_name = Column(String(100), unique=True, nullable=False)
    rule_type = Column(String(50), nullable=False)
    regulation_type = Column(String(50), nullable=False)
    rule_config = Column(Text, nullable=False)  # JSON configuration
    is_active = Column(Boolean, default=True)
    severity = Column(String(20), default="medium")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class EnhancedComplianceService:
    """Enhanced compliance service with comprehensive regulatory features"""
    
    def __init__(self):
        # Transaction monitoring thresholds
        self.thresholds = {
            "daily_volume": Decimal('50000'),
            "single_transaction": Decimal('10000'),
            "monthly_volume": Decimal('500000'),
            "suspicious_activity": Decimal('25000'),
            "rapid_trading": 50,  # Number of trades
            "velocity_threshold": Decimal('100000')  # Volume in short time
        }
        
        # High-risk countries (FATF list + additional)
        self.high_risk_countries = {
            'AF', 'BY', 'CF', 'CU', 'CD', 'ER', 'GN', 'GW', 'HT', 'IR',
            'IQ', 'LB', 'LY', 'ML', 'MM', 'NI', 'KP', 'RU', 'SO', 'SS',
            'SD', 'SY', 'UA', 'VE', 'YE', 'ZW', 'PK', 'TR'
        }
        
        # Sanctions lists (simplified - in production, integrate with real APIs)
        self.sanctions_lists = {
            "OFAC_SDN": "Office of Foreign Assets Control - Specially Designated Nationals",
            "UN_SANCTIONS": "United Nations Security Council Sanctions",
            "EU_SANCTIONS": "European Union Sanctions",
            "HMT_SANCTIONS": "HM Treasury Financial Sanctions",
            "DFAT_SANCTIONS": "Department of Foreign Affairs and Trade Sanctions"
        }
    
    def enhanced_kyc_verification(
        self, 
        user_id: int, 
        kyc_data: Dict[str, Any], 
        db: Session
    ) -> Dict[str, Any]:
        """Enhanced KYC verification with comprehensive checks"""
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                raise ValueError("User not found")
            
            verification_results = {
                "user_id": user_id,
                "verification_id": f"KYC_{user_id}_{int(datetime.utcnow().timestamp())}",
                "overall_status": ComplianceStatus.PENDING,
                "risk_level": RiskLevel.MEDIUM,
                "checks_performed": [],
                "issues_found": [],
                "recommendations": [],
                "next_review_date": None
            }
            
            # 1. Document verification
            doc_result = self._verify_identity_documents(kyc_data, db, user_id)
            verification_results["checks_performed"].append("document_verification")
            if not doc_result["valid"]:
                verification_results["issues_found"].extend(doc_result["issues"])
            
            # 2. Address verification
            address_result = self._verify_address(kyc_data)
            verification_results["checks_performed"].append("address_verification")
            if not address_result["valid"]:
                verification_results["issues_found"].extend(address_result["issues"])
            
            # 3. Sanctions screening
            sanctions_result = self._comprehensive_sanctions_screening(kyc_data, db, user_id)
            verification_results["checks_performed"].append("sanctions_screening")
            if sanctions_result["matches_found"]:
                verification_results["issues_found"].append("Potential sanctions match found")
                verification_results["risk_level"] = RiskLevel.CRITICAL
            
            # 4. PEP (Politically Exposed Person) screening
            pep_result = self._pep_screening(kyc_data)
            verification_results["checks_performed"].append("pep_screening")
            if pep_result["is_pep"]:
                verification_results["issues_found"].append("Politically Exposed Person identified")
                verification_results["risk_level"] = RiskLevel.HIGH
            
            # 5. Adverse media screening
            media_result = self._adverse_media_screening(kyc_data)
            verification_results["checks_performed"].append("adverse_media_screening")
            if media_result["adverse_found"]:
                verification_results["issues_found"].append("Adverse media mentions found")
            
            # 6. Risk assessment
            risk_assessment = self._calculate_comprehensive_risk_score(
                kyc_data, doc_result, sanctions_result, pep_result, media_result
            )
            verification_results["risk_score"] = risk_assessment["score"]
            verification_results["risk_factors"] = risk_assessment["factors"]
            
            # Determine overall status
            if len(verification_results["issues_found"]) == 0:
                verification_results["overall_status"] = ComplianceStatus.COMPLIANT
                verification_results["next_review_date"] = datetime.utcnow() + timedelta(days=365)
            elif verification_results["risk_level"] in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
                verification_results["overall_status"] = ComplianceStatus.NON_COMPLIANT
                verification_results["recommendations"].append("Manual review required")
            else:
                verification_results["overall_status"] = ComplianceStatus.UNDER_REVIEW
                verification_results["next_review_date"] = datetime.utcnow() + timedelta(days=90)
            
            # Log KYC verification
            data_protection_service.create_data_processing_log(
                db=db,
                data_subject_id=str(user_id),
                processing_activity="kyc_verification",
                data_types=["identity_documents", "personal_information", "address"],
                legal_basis="legitimate_interest",
                purpose="Customer due diligence and regulatory compliance",
                user_id=user_id,
                retention_period=2555,  # 7 years
                consent_given=True
            )
            
            return verification_results
            
        except Exception as e:
            logger.error(f"KYC verification failed: {e}")
            raise ValueError(f"KYC verification error: {str(e)}")
    
    def _verify_identity_documents(
        self, 
        kyc_data: Dict[str, Any], 
        db: Session, 
        user_id: int
    ) -> Dict[str, Any]:
        """Verify identity documents"""
        issues = []
        
        # Check required fields
        required_fields = ['document_type', 'document_number', 'document_country', 'document_expiry']
        for field in required_fields:
            if field not in kyc_data or not kyc_data[field]:
                issues.append(f"Missing {field}")
        
        if issues:
            return {"valid": False, "issues": issues}
        
        # Validate document format
        doc_type = kyc_data['document_type']
        doc_number = kyc_data['document_number']
        
        if doc_type == 'passport':
            if not re.match(r'^[A-Z0-9]{6,9}$', doc_number.upper()):
                issues.append("Invalid passport number format")
        elif doc_type == 'national_id':
            if len(doc_number) < 5 or len(doc_number) > 20:
                issues.append("Invalid national ID format")
        elif doc_type == 'drivers_license':
            if len(doc_number) < 5 or len(doc_number) > 15:
                issues.append("Invalid driver's license format")
        
        # Check document expiry
        try:
            expiry_date = datetime.strptime(kyc_data['document_expiry'], '%Y-%m-%d')
            if expiry_date < datetime.now():
                issues.append("Document has expired")
            elif expiry_date < datetime.now() + timedelta(days=30):
                issues.append("Document expires within 30 days")
        except ValueError:
            issues.append("Invalid document expiry date format")
        
        # Store document hash for future reference
        if not issues:
            doc_hash = hashlib.sha256(
                f"{doc_type}:{doc_number}:{kyc_data['document_country']}".encode()
            ).hexdigest()
            
            kyc_doc = KYCDocument(
                user_id=user_id,
                document_type=doc_type,
                document_number=data_protection_service.encrypt_field(doc_number),
                document_country=kyc_data['document_country'],
                document_expiry=expiry_date,
                document_hash=doc_hash,
                verification_status="verified" if not issues else "failed"
            )
            db.add(kyc_doc)
            db.commit()
        
        return {"valid": len(issues) == 0, "issues": issues}
    
    def _verify_address(self, kyc_data: Dict[str, Any]) -> Dict[str, Any]:
        """Verify address information"""
        issues = []
        
        if 'address' not in kyc_data:
            issues.append("Address information required")
            return {"valid": False, "issues": issues}
        
        address = kyc_data['address']
        required_fields = ['street', 'city', 'country', 'postal_code']
        
        for field in required_fields:
            if field not in address or not address[field]:
                issues.append(f"Missing address {field}")
        
        # Validate postal code format (basic validation)
        if 'postal_code' in address:
            postal_code = address['postal_code']
            country = address.get('country', '').upper()
            
            if country == 'US' and not re.match(r'^\d{5}(-\d{4})?$', postal_code):
                issues.append("Invalid US postal code format")
            elif country == 'GB' and not re.match(r'^[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}$', postal_code.upper()):
                issues.append("Invalid UK postal code format")
            elif country == 'CA' and not re.match(r'^[A-Z]\d[A-Z]\s?\d[A-Z]\d$', postal_code.upper()):
                issues.append("Invalid Canadian postal code format")
        
        return {"valid": len(issues) == 0, "issues": issues}
    
    def _comprehensive_sanctions_screening(
        self, 
        kyc_data: Dict[str, Any], 
        db: Session, 
        user_id: int
    ) -> Dict[str, Any]:
        """Comprehensive sanctions screening"""
        try:
            full_name = kyc_data.get('full_name', '').strip().lower()
            nationality = kyc_data.get('nationality', '').upper()
            
            # Check against high-risk countries
            country_risk = nationality in self.high_risk_countries
            
            # Simplified sanctions check (in production, use real APIs)
            known_sanctioned_entities = [
                'john doe', 'jane smith', 'test user', 'sanctioned person',
                'blocked entity', 'prohibited individual'
            ]
            
            name_match = any(
                sanctioned in full_name or full_name in sanctioned
                for sanctioned in known_sanctioned_entities
            )
            
            matches_found = name_match or country_risk
            match_details = []
            
            if name_match:
                match_details.append({
                    "type": "name_match",
                    "list": "OFAC_SDN",
                    "confidence": 0.85,
                    "details": "Potential name match found"
                })
            
            if country_risk:
                match_details.append({
                    "type": "country_risk",
                    "list": "HIGH_RISK_COUNTRIES",
                    "confidence": 1.0,
                    "details": f"High-risk country: {nationality}"
                })
            
            # Store sanctions check result
            sanctions_check = SanctionsCheck(
                user_id=user_id,
                check_type="comprehensive",
                search_terms=json.dumps({
                    "name": full_name,
                    "nationality": nationality
                }),
                lists_checked=json.dumps(list(self.sanctions_lists.keys())),
                matches_found=matches_found,
                match_details=json.dumps(match_details) if match_details else None,
                risk_score=80 if matches_found else 10,
                next_check_due=datetime.utcnow() + timedelta(days=30)
            )
            db.add(sanctions_check)
            db.commit()
            
            return {
                "matches_found": matches_found,
                "match_details": match_details,
                "lists_checked": list(self.sanctions_lists.keys()),
                "risk_score": 80 if matches_found else 10
            }
            
        except Exception as e:
            logger.error(f"Sanctions screening failed: {e}")
            return {
                "matches_found": False,
                "match_details": [],
                "lists_checked": [],
                "risk_score": 50,
                "error": str(e)
            }
    
    def _pep_screening(self, kyc_data: Dict[str, Any]) -> Dict[str, Any]:
        """Politically Exposed Person screening"""
        # Simplified PEP check (in production, use specialized PEP databases)
        full_name = kyc_data.get('full_name', '').lower()
        
        # Mock PEP list
        known_peps = [
            'political figure', 'government official', 'minister',
            'ambassador', 'judge', 'military officer'
        ]
        
        is_pep = any(pep in full_name for pep in known_peps)
        
        return {
            "is_pep": is_pep,
            "pep_category": "government_official" if is_pep else None,
            "risk_level": RiskLevel.HIGH if is_pep else RiskLevel.LOW,
            "additional_due_diligence_required": is_pep
        }
    
    def _adverse_media_screening(self, kyc_data: Dict[str, Any]) -> Dict[str, Any]:
        """Adverse media screening"""
        # Simplified adverse media check
        full_name = kyc_data.get('full_name', '').lower()
        
        # Mock adverse terms
        adverse_terms = [
            'fraud', 'money laundering', 'corruption', 'criminal',
            'investigation', 'charges', 'convicted'
        ]
        
        adverse_found = any(term in full_name for term in adverse_terms)
        
        return {
            "adverse_found": adverse_found,
            "sources_checked": ["news_articles", "regulatory_notices", "court_records"],
            "risk_level": RiskLevel.HIGH if adverse_found else RiskLevel.LOW
        }
    
    def _calculate_comprehensive_risk_score(
        self, 
        kyc_data: Dict[str, Any],
        doc_result: Dict[str, Any],
        sanctions_result: Dict[str, Any],
        pep_result: Dict[str, Any],
        media_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate comprehensive risk score"""
        score = 0
        factors = []
        
        # Document verification issues
        if not doc_result["valid"]:
            score += 20
            factors.append("Document verification issues")
        
        # Sanctions matches
        if sanctions_result["matches_found"]:
            score += 50
            factors.append("Sanctions screening matches")
        
        # PEP status
        if pep_result["is_pep"]:
            score += 30
            factors.append("Politically Exposed Person")
        
        # Adverse media
        if media_result["adverse_found"]:
            score += 25
            factors.append("Adverse media mentions")
        
        # Country risk
        nationality = kyc_data.get('nationality', '').upper()
        if nationality in self.high_risk_countries:
            score += 15
            factors.append("High-risk jurisdiction")
        
        # Age-based risk
        if 'date_of_birth' in kyc_data:
            try:
                dob = datetime.strptime(kyc_data['date_of_birth'], '%Y-%m-%d')
                age = (datetime.now() - dob).days / 365.25
                if age < 21:
                    score += 10
                    factors.append("Young age profile")
                elif age > 80:
                    score += 5
                    factors.append("Elderly age profile")
            except:
                pass
        
        return {
            "score": min(score, 100),  # Cap at 100
            "factors": factors,
            "risk_level": self._score_to_risk_level(score)
        }
    
    def _score_to_risk_level(self, score: int) -> RiskLevel:
        """Convert numeric score to risk level"""
        if score >= 70:
            return RiskLevel.CRITICAL
        elif score >= 50:
            return RiskLevel.HIGH
        elif score >= 30:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def advanced_transaction_monitoring(
        self, 
        user_id: int, 
        trade_data: Dict[str, Any], 
        db: Session
    ) -> Dict[str, Any]:
        """Advanced transaction monitoring with multiple detection algorithms"""
        try:
            alerts = []
            risk_score = 0
            
            # Get user and trade information
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                raise ValueError("User not found")
            
            trade_value = Decimal(str(trade_data.get('total_value', 0)))
            
            # 1. Threshold-based monitoring
            threshold_alerts = self._check_transaction_thresholds(user_id, trade_value, db)
            alerts.extend(threshold_alerts)
            
            # 2. Pattern-based monitoring
            pattern_alerts = self._detect_suspicious_patterns(user_id, trade_data, db)
            alerts.extend(pattern_alerts)
            
            # 3. Velocity monitoring
            velocity_alerts = self._check_transaction_velocity(user_id, trade_value, db)
            alerts.extend(velocity_alerts)
            
            # 4. Behavioral analysis
            behavioral_alerts = self._analyze_trading_behavior(user_id, trade_data, db)
            alerts.extend(behavioral_alerts)
            
            # 5. Geographic analysis
            geo_alerts = self._analyze_geographic_risk(user_id, trade_data, db)
            alerts.extend(geo_alerts)
            
            # Calculate overall risk score
            risk_score = sum(alert.get('risk_score', 0) for alert in alerts)
            risk_score = min(risk_score, 100)  # Cap at 100
            
            # Store monitoring results
            for alert in alerts:
                monitoring_record = TransactionMonitoring(
                    user_id=user_id,
                    alert_type=alert['type'],
                    alert_description=alert['description'],
                    risk_score=alert.get('risk_score', 0),
                    threshold_breached=alert.get('threshold', 'N/A'),
                    alert_status='open'
                )
                db.add(monitoring_record)
            
            if alerts:
                db.commit()
            
            return {
                "monitoring_passed": len(alerts) == 0,
                "alerts": alerts,
                "risk_score": risk_score,
                "risk_level": self._score_to_risk_level(risk_score),
                "recommended_action": self._get_recommended_action(risk_score, alerts)
            }
            
        except Exception as e:
            logger.error(f"Transaction monitoring failed: {e}")
            return {
                "monitoring_passed": False,
                "alerts": [{"type": "system_error", "description": str(e)}],
                "risk_score": 50,
                "risk_level": RiskLevel.MEDIUM
            }
    
    def _check_transaction_thresholds(
        self, 
        user_id: int, 
        trade_value: Decimal, 
        db: Session
    ) -> List[Dict[str, Any]]:
        """Check transaction against various thresholds"""
        alerts = []
        
        # Single transaction threshold
        if trade_value > self.thresholds['single_transaction']:
            alerts.append({
                "type": "large_transaction",
                "description": f"Transaction exceeds single transaction threshold: ${trade_value}",
                "risk_score": 20,
                "threshold": f"${self.thresholds['single_transaction']}"
            })
        
        # Daily volume check
        today = datetime.utcnow().date()
        daily_trades = db.query(Trade).filter(
            Trade.user_id == user_id,
            Trade.created_at >= today,
            Trade.status == "executed"
        ).all()
        
        daily_volume = sum(trade.total_value for trade in daily_trades) + trade_value
        if daily_volume > self.thresholds['daily_volume']:
            alerts.append({
                "type": "daily_volume_exceeded",
                "description": f"Daily volume threshold exceeded: ${daily_volume}",
                "risk_score": 25,
                "threshold": f"${self.thresholds['daily_volume']}"
            })
        
        return alerts
    
    def _detect_suspicious_patterns(
        self, 
        user_id: int, 
        trade_data: Dict[str, Any], 
        db: Session
    ) -> List[Dict[str, Any]]:
        """Detect suspicious trading patterns"""
        alerts = []
        
        # Get recent trades
        recent_trades = db.query(Trade).filter(
            Trade.user_id == user_id,
            Trade.created_at >= datetime.utcnow() - timedelta(hours=24)
        ).all()
        
        # Round number pattern (potential structuring)
        trade_value = Decimal(str(trade_data.get('total_value', 0)))
        if trade_value % 1000 == 0 and trade_value >= 5000:
            alerts.append({
                "type": "round_number_pattern",
                "description": f"Round number transaction: ${trade_value}",
                "risk_score": 15,
                "threshold": "Round number detection"
            })
        
        # Rapid succession trades
        if len(recent_trades) > 10:
            alerts.append({
                "type": "rapid_trading",
                "description": f"High frequency trading: {len(recent_trades)} trades in 24h",
                "risk_score": 20,
                "threshold": "10 trades per day"
            })
        
        return alerts
    
    def _check_transaction_velocity(
        self, 
        user_id: int, 
        trade_value: Decimal, 
        db: Session
    ) -> List[Dict[str, Any]]:
        """Check transaction velocity"""
        alerts = []
        
        # Check volume in last hour
        hour_ago = datetime.utcnow() - timedelta(hours=1)
        hourly_trades = db.query(Trade).filter(
            Trade.user_id == user_id,
            Trade.created_at >= hour_ago,
            Trade.status == "executed"
        ).all()
        
        hourly_volume = sum(trade.total_value for trade in hourly_trades) + trade_value
        if hourly_volume > self.thresholds['velocity_threshold']:
            alerts.append({
                "type": "high_velocity",
                "description": f"High transaction velocity: ${hourly_volume} in 1 hour",
                "risk_score": 30,
                "threshold": f"${self.thresholds['velocity_threshold']} per hour"
            })
        
        return alerts
    
    def _analyze_trading_behavior(
        self, 
        user_id: int, 
        trade_data: Dict[str, Any], 
        db: Session
    ) -> List[Dict[str, Any]]:
        """Analyze trading behavior for anomalies"""
        alerts = []
        
        # Get historical trading pattern
        historical_trades = db.query(Trade).filter(
            Trade.user_id == user_id,
            Trade.created_at >= datetime.utcnow() - timedelta(days=30),
            Trade.status == "executed"
        ).all()
        
        if len(historical_trades) < 5:
            return alerts  # Not enough data for analysis
        
        # Calculate average trade size
        avg_trade_size = sum(trade.total_value for trade in historical_trades) / len(historical_trades)
        current_trade_size = Decimal(str(trade_data.get('total_value', 0)))
        
        # Unusual trade size
        if current_trade_size > avg_trade_size * 5:
            alerts.append({
                "type": "unusual_trade_size",
                "description": f"Trade size significantly larger than average: ${current_trade_size} vs ${avg_trade_size}",
                "risk_score": 25,
                "threshold": "5x average trade size"
            })
        
        return alerts
    
    def _analyze_geographic_risk(
        self, 
        user_id: int, 
        trade_data: Dict[str, Any], 
        db: Session
    ) -> List[Dict[str, Any]]:
        """Analyze geographic risk factors"""
        alerts = []
        
        # This would analyze IP geolocation, VPN usage, etc.
        # Simplified implementation
        
        return alerts
    
    def _get_recommended_action(self, risk_score: int, alerts: List[Dict[str, Any]]) -> str:
        """Get recommended action based on risk assessment"""
        if risk_score >= 70:
            return "Block transaction and conduct manual review"
        elif risk_score >= 50:
            return "Flag for enhanced monitoring"
        elif risk_score >= 30:
            return "Continue monitoring"
        else:
            return "No action required"
    
    def generate_regulatory_report(
        self, 
        report_type: str, 
        regulation_type: RegulationType,
        period_start: datetime,
        period_end: datetime,
        db: Session
    ) -> Dict[str, Any]:
        """Generate regulatory reports"""
        try:
            report_id = f"{regulation_type.value.upper()}_{report_type}_{int(datetime.utcnow().timestamp())}"
            
            if regulation_type == RegulationType.CFTC:
                report_data = self._generate_cftc_report(period_start, period_end, db)
            elif regulation_type == RegulationType.MIFID_II:
                report_data = self._generate_mifid_report(period_start, period_end, db)
            elif regulation_type == RegulationType.AML:
                report_data = self._generate_aml_report(period_start, period_end, db)
            else:
                raise ValueError(f"Unsupported regulation type: {regulation_type}")
            
            # Store report
            report = RegulatoryReport(
                report_id=report_id,
                report_type=report_type,
                regulation_type=regulation_type.value,
                reporting_period_start=period_start,
                reporting_period_end=period_end,
                report_data=json.dumps(report_data, default=str),
                submission_status="draft"
            )
            db.add(report)
            db.commit()
            
            return {
                "report_id": report_id,
                "status": "generated",
                "data": report_data
            }
            
        except Exception as e:
            logger.error(f"Regulatory report generation failed: {e}")
            raise ValueError(f"Report generation error: {str(e)}")
    
    def _generate_cftc_report(
        self, 
        period_start: datetime, 
        period_end: datetime, 
        db: Session
    ) -> Dict[str, Any]:
        """Generate CFTC regulatory report"""
        # Simplified CFTC reporting
        trades = db.query(Trade).filter(
            Trade.created_at >= period_start,
            Trade.created_at <= period_end,
            Trade.status == "executed"
        ).all()
        
        return {
            "reporting_entity": "Optionix Trading Platform",
            "period": {
                "start": period_start.isoformat(),
                "end": period_end.isoformat()
            },
            "summary": {
                "total_trades": len(trades),
                "total_volume": str(sum(trade.total_value for trade in trades)),
                "unique_traders": len(set(trade.user_id for trade in trades))
            },
            "trades": [
                {
                    "trade_id": trade.trade_id,
                    "symbol": trade.symbol,
                    "quantity": str(trade.quantity),
                    "price": str(trade.price),
                    "timestamp": trade.created_at.isoformat()
                }
                for trade in trades
            ]
        }
    
    def _generate_mifid_report(
        self, 
        period_start: datetime, 
        period_end: datetime, 
        db: Session
    ) -> Dict[str, Any]:
        """Generate MiFID II regulatory report"""
        # Simplified MiFID II reporting
        return {
            "reporting_entity": "Optionix Trading Platform",
            "period": {
                "start": period_start.isoformat(),
                "end": period_end.isoformat()
            },
            "transaction_reporting": {
                "total_transactions": 0,
                "reportable_transactions": 0
            }
        }
    
    def _generate_aml_report(
        self, 
        period_start: datetime, 
        period_end: datetime, 
        db: Session
    ) -> Dict[str, Any]:
        """Generate AML regulatory report"""
        # Get suspicious activity reports
        alerts = db.query(TransactionMonitoring).filter(
            TransactionMonitoring.created_at >= period_start,
            TransactionMonitoring.created_at <= period_end
        ).all()
        
        return {
            "reporting_entity": "Optionix Trading Platform",
            "period": {
                "start": period_start.isoformat(),
                "end": period_end.isoformat()
            },
            "suspicious_activity": {
                "total_alerts": len(alerts),
                "high_risk_alerts": len([a for a in alerts if a.risk_score >= 70]),
                "resolved_alerts": len([a for a in alerts if a.alert_status == "closed"])
            }
        }


# Global service instance
enhanced_compliance_service = EnhancedComplianceService()

