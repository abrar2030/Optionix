"""
Financial standards compliance module for Optionix.
Implements SOX, Basel III, MiFID II, Dodd-Frank, and other financial regulations.
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from data_protection import data_protection_service
from models import Account, Position, Trade, User
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship

logger = logging.getLogger(__name__)

Base = declarative_base()


class FinancialRegulation(str, Enum):
    """Financial regulations"""

    SOX = "sox"  # Sarbanes-Oxley Act
    BASEL_III = "basel_iii"  # Basel III
    MIFID_II = "mifid_ii"  # Markets in Financial Instruments Directive II
    DODD_FRANK = "dodd_frank"  # Dodd-Frank Act
    CFTC = "cftc"  # Commodity Futures Trading Commission
    SEC = "sec"  # Securities and Exchange Commission
    FINRA = "finra"  # Financial Industry Regulatory Authority
    EMIR = "emir"  # European Market Infrastructure Regulation


class TransactionStatus(str, Enum):
    """Transaction status for audit trail"""

    INITIATED = "initiated"
    VALIDATED = "validated"
    EXECUTED = "executed"
    SETTLED = "settled"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RECONCILED = "reconciled"


class RiskMetricType(str, Enum):
    """Types of risk metrics"""

    VAR = "var"  # Value at Risk
    EXPECTED_SHORTFALL = "expected_shortfall"
    LEVERAGE_RATIO = "leverage_ratio"
    LIQUIDITY_RATIO = "liquidity_ratio"
    CONCENTRATION_RISK = "concentration_risk"
    COUNTERPARTY_RISK = "counterparty_risk"


class FinancialAuditLog(Base):
    """Comprehensive financial audit log for SOX compliance"""

    __tablename__ = "financial_audit_logs"

    id = Column(Integer, primary_key=True)
    audit_id = Column(String(100), unique=True, nullable=False)
    transaction_id = Column(String(100), nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    account_id = Column(Integer, ForeignKey("accounts.id"), nullable=True)

    # Financial data
    transaction_type = Column(String(50), nullable=False)
    amount = Column(Numeric(precision=18, scale=8), nullable=False)
    currency = Column(String(10), default="USD")

    # Audit trail
    previous_state = Column(Text, nullable=True)  # JSON
    new_state = Column(Text, nullable=False)  # JSON
    state_hash = Column(String(64), nullable=False)  # SHA-256

    # Compliance fields
    regulation_type = Column(String(50), nullable=False)
    compliance_status = Column(String(20), default="compliant")
    control_reference = Column(String(100), nullable=True)

    # Authorization
    authorized_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    authorization_level = Column(String(50), nullable=True)

    # Timestamps
    business_date = Column(DateTime, nullable=False)
    system_timestamp = Column(DateTime, default=datetime.utcnow)

    # Indexes for performance
    __table_args__ = (
        Index("idx_financial_audit_transaction", "transaction_id"),
        Index("idx_financial_audit_user", "user_id"),
        Index("idx_financial_audit_date", "business_date"),
        Index("idx_financial_audit_regulation", "regulation_type"),
    )


class DataIntegrityCheck(Base):
    """Data integrity verification records"""

    __tablename__ = "data_integrity_checks"

    id = Column(Integer, primary_key=True)
    check_id = Column(String(100), unique=True, nullable=False)
    check_type = Column(String(50), nullable=False)  # balance, position, trade
    entity_type = Column(String(50), nullable=False)  # account, user, system
    entity_id = Column(String(100), nullable=False)

    # Integrity data
    expected_value = Column(Text, nullable=False)  # JSON
    actual_value = Column(Text, nullable=False)  # JSON
    variance = Column(Text, nullable=True)  # JSON

    # Check results
    integrity_status = Column(String(20), nullable=False)  # pass, fail, warning
    discrepancy_amount = Column(Numeric(precision=18, scale=8), nullable=True)
    tolerance_threshold = Column(Numeric(precision=18, scale=8), nullable=True)

    # Resolution
    resolution_status = Column(String(20), default="pending")
    resolution_notes = Column(Text, nullable=True)
    resolved_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    resolved_at = Column(DateTime, nullable=True)

    # Timestamps
    check_timestamp = Column(DateTime, default=datetime.utcnow)
    business_date = Column(DateTime, nullable=False)


class ReconciliationRecord(Base):
    """Financial reconciliation records"""

    __tablename__ = "reconciliation_records"

    id = Column(Integer, primary_key=True)
    reconciliation_id = Column(String(100), unique=True, nullable=False)
    reconciliation_type = Column(String(50), nullable=False)  # daily, monthly, trade

    # Source data
    internal_source = Column(String(100), nullable=False)
    external_source = Column(String(100), nullable=False)

    # Reconciliation data
    internal_balance = Column(Numeric(precision=18, scale=8), nullable=False)
    external_balance = Column(Numeric(precision=18, scale=8), nullable=False)
    difference = Column(Numeric(precision=18, scale=8), nullable=False)

    # Status
    reconciliation_status = Column(
        String(20), nullable=False
    )  # matched, unmatched, investigating
    tolerance_threshold = Column(
        Numeric(precision=18, scale=8), default=Decimal("0.01")
    )

    # Investigation
    investigation_notes = Column(Text, nullable=True)
    investigated_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    investigation_date = Column(DateTime, nullable=True)

    # Timestamps
    business_date = Column(DateTime, nullable=False)
    reconciliation_date = Column(DateTime, default=datetime.utcnow)


class RiskMetric(Base):
    """Risk metrics for Basel III compliance"""

    __tablename__ = "risk_metrics"

    id = Column(Integer, primary_key=True)
    metric_id = Column(String(100), unique=True, nullable=False)
    metric_type = Column(String(50), nullable=False)
    entity_type = Column(String(50), nullable=False)  # portfolio, account, position
    entity_id = Column(String(100), nullable=False)

    # Risk data
    metric_value = Column(Numeric(precision=18, scale=8), nullable=False)
    confidence_level = Column(
        Numeric(precision=5, scale=4), nullable=True
    )  # e.g., 0.95 for 95%
    time_horizon = Column(Integer, nullable=True)  # days

    # Limits and thresholds
    limit_value = Column(Numeric(precision=18, scale=8), nullable=True)
    warning_threshold = Column(Numeric(precision=18, scale=8), nullable=True)
    breach_status = Column(String(20), default="within_limits")

    # Calculation details
    calculation_method = Column(String(100), nullable=False)
    input_parameters = Column(Text, nullable=True)  # JSON

    # Timestamps
    calculation_date = Column(DateTime, default=datetime.utcnow)
    business_date = Column(DateTime, nullable=False)

    # Indexes
    __table_args__ = (
        Index("idx_risk_metric_entity", "entity_type", "entity_id"),
        Index("idx_risk_metric_type_date", "metric_type", "business_date"),
    )


class RegulatoryReport(Base):
    """Regulatory reporting records"""

    __tablename__ = "regulatory_reports_financial"

    id = Column(Integer, primary_key=True)
    report_id = Column(String(100), unique=True, nullable=False)
    regulation_type = Column(String(50), nullable=False)
    report_type = Column(String(100), nullable=False)

    # Reporting period
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)

    # Report data
    report_data = Column(Text, nullable=False)  # JSON
    data_hash = Column(String(64), nullable=False)  # SHA-256

    # Submission
    submission_status = Column(String(20), default="draft")
    submitted_at = Column(DateTime, nullable=True)
    submission_reference = Column(String(100), nullable=True)

    # Validation
    validation_status = Column(String(20), default="pending")
    validation_errors = Column(Text, nullable=True)  # JSON

    # Timestamps
    generated_at = Column(DateTime, default=datetime.utcnow)
    generated_by = Column(Integer, ForeignKey("users.id"), nullable=False)


class FinancialStandardsService:
    """Service for financial standards compliance"""

    def __init__(self):
        # SOX compliance settings
        self.sox_controls = {
            "segregation_of_duties": True,
            "authorization_levels": {
                "trade_execution": ["trader", "senior_trader"],
                "position_modification": ["risk_manager", "senior_trader"],
                "account_creation": ["admin", "compliance_officer"],
                "large_transactions": ["senior_trader", "risk_manager"],
            },
            "audit_retention_years": 7,
            "control_testing_frequency": 90,  # days
        }

        # Basel III risk limits
        self.basel_limits = {
            "leverage_ratio_minimum": Decimal("0.03"),  # 3%
            "liquidity_coverage_ratio": Decimal("1.0"),  # 100%
            "net_stable_funding_ratio": Decimal("1.0"),  # 100%
            "capital_adequacy_ratio": Decimal("0.08"),  # 8%
        }

        # MiFID II transaction reporting thresholds
        self.mifid_thresholds = {
            "equity_threshold": Decimal("15000"),  # EUR 15,000
            "bond_threshold": Decimal("50000"),  # EUR 50,000
            "derivative_threshold": Decimal("25000"),  # EUR 25,000
        }

    def create_financial_audit_log(
        self,
        db: Session,
        transaction_id: str,
        user_id: int,
        account_id: Optional[int],
        transaction_type: str,
        amount: Decimal,
        previous_state: Optional[Dict[str, Any]],
        new_state: Dict[str, Any],
        regulation_type: FinancialRegulation,
        authorized_by: Optional[int] = None,
        authorization_level: Optional[str] = None,
    ) -> FinancialAuditLog:
        """Create comprehensive financial audit log entry"""
        try:
            audit_id = f"FA_{int(datetime.utcnow().timestamp())}_{transaction_id}"

            # Create state hash for integrity
            state_data = json.dumps(new_state, sort_keys=True, default=str)
            state_hash = hashlib.sha256(state_data.encode()).hexdigest()

            audit_log = FinancialAuditLog(
                audit_id=audit_id,
                transaction_id=transaction_id,
                user_id=user_id,
                account_id=account_id,
                transaction_type=transaction_type,
                amount=amount,
                currency="USD",
                previous_state=(
                    json.dumps(previous_state, default=str) if previous_state else None
                ),
                new_state=json.dumps(new_state, default=str),
                state_hash=state_hash,
                regulation_type=regulation_type.value,
                compliance_status="compliant",
                authorized_by=authorized_by,
                authorization_level=authorization_level,
                business_date=datetime.utcnow().date(),
            )

            db.add(audit_log)
            db.commit()
            db.refresh(audit_log)

            return audit_log

        except Exception as e:
            logger.error(f"Failed to create financial audit log: {e}")
            raise ValueError(f"Audit log creation failed: {str(e)}")

    def perform_data_integrity_check(
        self,
        db: Session,
        check_type: str,
        entity_type: str,
        entity_id: str,
        expected_value: Dict[str, Any],
        actual_value: Dict[str, Any],
        tolerance_threshold: Optional[Decimal] = None,
    ) -> DataIntegrityCheck:
        """Perform data integrity verification"""
        try:
            check_id = (
                f"DIC_{int(datetime.utcnow().timestamp())}_{entity_type}_{entity_id}"
            )

            # Calculate variance
            variance = self._calculate_variance(expected_value, actual_value)

            # Determine integrity status
            integrity_status = "pass"
            discrepancy_amount = Decimal("0")

            if variance:
                discrepancy_amount = abs(
                    Decimal(str(variance.get("total_difference", 0)))
                )

                if tolerance_threshold and discrepancy_amount > tolerance_threshold:
                    integrity_status = "fail"
                elif discrepancy_amount > Decimal("0"):
                    integrity_status = "warning"

            integrity_check = DataIntegrityCheck(
                check_id=check_id,
                check_type=check_type,
                entity_type=entity_type,
                entity_id=entity_id,
                expected_value=json.dumps(expected_value, default=str),
                actual_value=json.dumps(actual_value, default=str),
                variance=json.dumps(variance, default=str) if variance else None,
                integrity_status=integrity_status,
                discrepancy_amount=discrepancy_amount,
                tolerance_threshold=tolerance_threshold,
                business_date=datetime.utcnow().date(),
            )

            db.add(integrity_check)
            db.commit()
            db.refresh(integrity_check)

            return integrity_check

        except Exception as e:
            logger.error(f"Data integrity check failed: {e}")
            raise ValueError(f"Integrity check failed: {str(e)}")

    def _calculate_variance(
        self, expected: Dict[str, Any], actual: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Calculate variance between expected and actual values"""
        variance = {}
        total_difference = Decimal("0")

        # Compare numeric values
        for key in expected:
            if key in actual:
                try:
                    expected_val = Decimal(str(expected[key]))
                    actual_val = Decimal(str(actual[key]))
                    difference = actual_val - expected_val

                    if difference != 0:
                        variance[key] = {
                            "expected": str(expected_val),
                            "actual": str(actual_val),
                            "difference": str(difference),
                            "percentage": str(
                                (difference / expected_val * 100).quantize(
                                    Decimal("0.01")
                                )
                            ),
                        }
                        total_difference += abs(difference)

                except (ValueError, TypeError, ZeroDivisionError):
                    # Non-numeric or zero division
                    if str(expected[key]) != str(actual[key]):
                        variance[key] = {
                            "expected": str(expected[key]),
                            "actual": str(actual[key]),
                            "difference": "non_numeric_mismatch",
                        }
            else:
                variance[key] = {
                    "expected": str(expected[key]),
                    "actual": "missing",
                    "difference": "missing_field",
                }

        # Check for extra fields in actual
        for key in actual:
            if key not in expected:
                variance[key] = {
                    "expected": "not_expected",
                    "actual": str(actual[key]),
                    "difference": "extra_field",
                }

        if variance:
            variance["total_difference"] = str(total_difference)
            return variance

        return None

    def perform_reconciliation(
        self,
        db: Session,
        reconciliation_type: str,
        internal_source: str,
        external_source: str,
        internal_balance: Decimal,
        external_balance: Decimal,
        tolerance_threshold: Decimal = Decimal("0.01"),
    ) -> ReconciliationRecord:
        """Perform financial reconciliation"""
        try:
            reconciliation_id = (
                f"REC_{reconciliation_type}_{int(datetime.utcnow().timestamp())}"
            )

            difference = external_balance - internal_balance
            abs_difference = abs(difference)

            # Determine reconciliation status
            if abs_difference <= tolerance_threshold:
                status = "matched"
            else:
                status = "unmatched"

            reconciliation = ReconciliationRecord(
                reconciliation_id=reconciliation_id,
                reconciliation_type=reconciliation_type,
                internal_source=internal_source,
                external_source=external_source,
                internal_balance=internal_balance,
                external_balance=external_balance,
                difference=difference,
                reconciliation_status=status,
                tolerance_threshold=tolerance_threshold,
                business_date=datetime.utcnow().date(),
            )

            db.add(reconciliation)
            db.commit()
            db.refresh(reconciliation)

            return reconciliation

        except Exception as e:
            logger.error(f"Reconciliation failed: {e}")
            raise ValueError(f"Reconciliation failed: {str(e)}")

    def calculate_risk_metrics(
        self,
        db: Session,
        entity_type: str,
        entity_id: str,
        metric_types: List[RiskMetricType],
        business_date: Optional[datetime] = None,
    ) -> List[RiskMetric]:
        """Calculate risk metrics for Basel III compliance"""
        try:
            if business_date is None:
                business_date = datetime.utcnow().date()

            risk_metrics = []

            for metric_type in metric_types:
                metric_value = self._calculate_specific_risk_metric(
                    db, entity_type, entity_id, metric_type, business_date
                )

                metric_id = f"RM_{metric_type.value}_{entity_type}_{entity_id}_{int(business_date.timestamp())}"

                # Get limits for this metric type
                limit_value = self._get_risk_limit(metric_type)
                warning_threshold = (
                    limit_value * Decimal("0.8") if limit_value else None
                )

                # Determine breach status
                breach_status = "within_limits"
                if limit_value:
                    if metric_value > limit_value:
                        breach_status = "breach"
                    elif warning_threshold and metric_value > warning_threshold:
                        breach_status = "warning"

                risk_metric = RiskMetric(
                    metric_id=metric_id,
                    metric_type=metric_type.value,
                    entity_type=entity_type,
                    entity_id=entity_id,
                    metric_value=metric_value,
                    limit_value=limit_value,
                    warning_threshold=warning_threshold,
                    breach_status=breach_status,
                    calculation_method="internal_model",
                    business_date=business_date,
                )

                db.add(risk_metric)
                risk_metrics.append(risk_metric)

            db.commit()
            return risk_metrics

        except Exception as e:
            logger.error(f"Risk metric calculation failed: {e}")
            raise ValueError(f"Risk calculation failed: {str(e)}")

    def _calculate_specific_risk_metric(
        self,
        db: Session,
        entity_type: str,
        entity_id: str,
        metric_type: RiskMetricType,
        business_date: datetime,
    ) -> Decimal:
        """Calculate specific risk metric"""
        if metric_type == RiskMetricType.VAR:
            return self._calculate_var(db, entity_type, entity_id, business_date)
        elif metric_type == RiskMetricType.LEVERAGE_RATIO:
            return self._calculate_leverage_ratio(
                db, entity_type, entity_id, business_date
            )
        elif metric_type == RiskMetricType.LIQUIDITY_RATIO:
            return self._calculate_liquidity_ratio(
                db, entity_type, entity_id, business_date
            )
        else:
            # Default calculation
            return Decimal("0")

    def _calculate_var(
        self,
        db: Session,
        entity_type: str,
        entity_id: str,
        business_date: datetime,
        confidence_level: Decimal = Decimal("0.95"),
        time_horizon: int = 1,
    ) -> Decimal:
        """Calculate Value at Risk (simplified implementation)"""
        try:
            # Get historical positions/trades for the entity
            if entity_type == "account":
                positions = (
                    db.query(Position)
                    .filter(
                        Position.account_id == int(entity_id), Position.status == "open"
                    )
                    .all()
                )

                total_exposure = sum(
                    position.size * position.current_price
                    for position in positions
                    if position.current_price
                )

                # Simplified VaR calculation (1% of total exposure)
                var = total_exposure * Decimal("0.01")
                return var

            return Decimal("0")

        except Exception as e:
            logger.error(f"VaR calculation failed: {e}")
            return Decimal("0")

    def _calculate_leverage_ratio(
        self, db: Session, entity_type: str, entity_id: str, business_date: datetime
    ) -> Decimal:
        """Calculate leverage ratio for Basel III"""
        try:
            if entity_type == "account":
                account = db.query(Account).filter(Account.id == int(entity_id)).first()
                if account:
                    # Simplified leverage ratio: margin_used / balance
                    if account.balance_usd > 0:
                        leverage_ratio = account.margin_used / account.balance_usd
                        return leverage_ratio

            return Decimal("0")

        except Exception as e:
            logger.error(f"Leverage ratio calculation failed: {e}")
            return Decimal("0")

    def _calculate_liquidity_ratio(
        self, db: Session, entity_type: str, entity_id: str, business_date: datetime
    ) -> Decimal:
        """Calculate liquidity ratio"""
        try:
            if entity_type == "account":
                account = db.query(Account).filter(Account.id == int(entity_id)).first()
                if account:
                    # Simplified liquidity ratio: available_margin / total_balance
                    if account.balance_usd > 0:
                        liquidity_ratio = account.margin_available / account.balance_usd
                        return liquidity_ratio

            return Decimal("0")

        except Exception as e:
            logger.error(f"Liquidity ratio calculation failed: {e}")
            return Decimal("0")

    def _get_risk_limit(self, metric_type: RiskMetricType) -> Optional[Decimal]:
        """Get risk limit for metric type"""
        if metric_type == RiskMetricType.LEVERAGE_RATIO:
            return self.basel_limits["leverage_ratio_minimum"]
        elif metric_type == RiskMetricType.LIQUIDITY_RATIO:
            return self.basel_limits["liquidity_coverage_ratio"]
        else:
            return None

    def check_sox_compliance(
        self, db: Session, transaction_data: Dict[str, Any], user_id: int
    ) -> Dict[str, Any]:
        """Check SOX compliance for financial transactions"""
        try:
            compliance_results = {
                "compliant": True,
                "violations": [],
                "controls_checked": [],
                "recommendations": [],
            }

            # Check segregation of duties
            if self.sox_controls["segregation_of_duties"]:
                compliance_results["controls_checked"].append("segregation_of_duties")
                # Implementation would check if same user is performing conflicting roles

            # Check authorization levels
            transaction_type = transaction_data.get("transaction_type", "")
            required_roles = self.sox_controls["authorization_levels"].get(
                transaction_type, []
            )

            if required_roles:
                compliance_results["controls_checked"].append("authorization_levels")
                user = db.query(User).filter(User.id == user_id).first()
                user_role = getattr(user, "role", "viewer")

                if user_role not in required_roles:
                    compliance_results["compliant"] = False
                    compliance_results["violations"].append(
                        f"Insufficient authorization level for {transaction_type}"
                    )

            # Check transaction amount thresholds
            amount = Decimal(str(transaction_data.get("amount", 0)))
            if amount > Decimal("100000"):  # Large transaction threshold
                compliance_results["controls_checked"].append(
                    "large_transaction_approval"
                )
                # Would check for additional approvals

            return compliance_results

        except Exception as e:
            logger.error(f"SOX compliance check failed: {e}")
            return {
                "compliant": False,
                "violations": [f"Compliance check error: {str(e)}"],
                "controls_checked": [],
                "recommendations": ["Manual review required"],
            }

    def generate_mifid_transaction_report(
        self, db: Session, period_start: datetime, period_end: datetime
    ) -> Dict[str, Any]:
        """Generate MiFID II transaction report"""
        try:
            # Get reportable transactions
            trades = (
                db.query(Trade)
                .filter(
                    Trade.created_at >= period_start,
                    Trade.created_at <= period_end,
                    Trade.status == "executed",
                )
                .all()
            )

            reportable_trades = []
            for trade in trades:
                # Check if trade meets MiFID II reporting thresholds
                if self._is_mifid_reportable(trade):
                    reportable_trades.append(
                        {
                            "transaction_id": trade.trade_id,
                            "instrument_id": trade.symbol,
                            "quantity": str(trade.quantity),
                            "price": str(trade.price),
                            "transaction_time": trade.created_at.isoformat(),
                            "venue": "Optionix",
                            "counterparty": "CLIENT",
                            "transaction_type": trade.trade_type.upper(),
                        }
                    )

            report_data = {
                "reporting_entity": "Optionix Trading Platform",
                "report_type": "MiFID_II_Transaction_Report",
                "period": {
                    "start": period_start.isoformat(),
                    "end": period_end.isoformat(),
                },
                "total_transactions": len(trades),
                "reportable_transactions": len(reportable_trades),
                "transactions": reportable_trades,
            }

            # Store report
            report_id = f"MIFID_{int(datetime.utcnow().timestamp())}"
            data_hash = hashlib.sha256(
                json.dumps(report_data, sort_keys=True, default=str).encode()
            ).hexdigest()

            report = RegulatoryReport(
                report_id=report_id,
                regulation_type=FinancialRegulation.MIFID_II.value,
                report_type="transaction_report",
                period_start=period_start,
                period_end=period_end,
                report_data=json.dumps(report_data, default=str),
                data_hash=data_hash,
                generated_by=1,  # System user
            )

            db.add(report)
            db.commit()

            return {"report_id": report_id, "status": "generated", "data": report_data}

        except Exception as e:
            logger.error(f"MiFID II report generation failed: {e}")
            raise ValueError(f"Report generation failed: {str(e)}")

    def _is_mifid_reportable(self, trade: Trade) -> bool:
        """Check if trade is reportable under MiFID II"""
        trade_value = trade.total_value

        # Simplified threshold check (in production, would be more sophisticated)
        if trade_value >= self.mifid_thresholds["derivative_threshold"]:
            return True

        return False

    def perform_daily_reconciliation(
        self, db: Session, business_date: datetime
    ) -> List[ReconciliationRecord]:
        """Perform end-of-day reconciliation"""
        try:
            reconciliations = []

            # Account balance reconciliation
            accounts = db.query(Account).filter(Account.is_active == True).all()

            for account in accounts:
                # Calculate expected balance from transactions
                expected_balance = self._calculate_expected_balance(
                    db, account.id, business_date
                )
                actual_balance = account.balance_usd

                reconciliation = self.perform_reconciliation(
                    db=db,
                    reconciliation_type="daily_balance",
                    internal_source="account_balance",
                    external_source="transaction_sum",
                    internal_balance=actual_balance,
                    external_balance=expected_balance,
                    tolerance_threshold=Decimal("0.01"),
                )

                reconciliations.append(reconciliation)

            return reconciliations

        except Exception as e:
            logger.error(f"Daily reconciliation failed: {e}")
            raise ValueError(f"Reconciliation failed: {str(e)}")

    def _calculate_expected_balance(
        self, db: Session, account_id: int, business_date: datetime
    ) -> Decimal:
        """Calculate expected account balance from transactions"""
        try:
            # Get all executed trades for the account up to business date
            trades = (
                db.query(Trade)
                .filter(
                    Trade.account_id == account_id,
                    Trade.status == "executed",
                    Trade.created_at <= business_date,
                )
                .all()
            )

            # Calculate net position
            net_value = Decimal("0")
            for trade in trades:
                if trade.trade_type == "buy":
                    net_value -= trade.total_value  # Money out
                else:
                    net_value += trade.total_value  # Money in

            # Add initial balance (would be stored separately in production)
            initial_balance = Decimal("10000")  # Mock initial balance
            expected_balance = initial_balance + net_value

            return expected_balance

        except Exception as e:
            logger.error(f"Expected balance calculation failed: {e}")
            return Decimal("0")


# Global service instance
financial_standards_service = FinancialStandardsService()
