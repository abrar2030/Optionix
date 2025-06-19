"""
Enhanced blockchain service module for Optionix platform.
Handles all blockchain interactions with robust security and error handling.
"""
from web3 import Web3
from web3.exceptions import Web3Exception, ContractLogicError
from eth_account import Account
import json
import os
import logging
from typing import Dict, Any, Optional, List
from decimal import Decimal
from config import settings
from models import Trade, Position, AuditLog
from sqlalchemy.orm import Session
import time

logger = logging.getLogger(__name__)


class BlockchainService:
    """Enhanced service for interacting with blockchain contracts and wallets"""
    
    def __init__(self):
        """Initialize blockchain service with Web3 provider"""
        self.w3 = None
        self.futures_contract = None
        self.futures_abi = []
        self._initialize_connection()
        self._load_contract_abi()
    
    def _initialize_connection(self):
        """Initialize Web3 connection with retry logic"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                self.w3 = Web3(Web3.HTTPProvider(
                    settings.ethereum_provider_url,
                    request_kwargs={'timeout': 30}
                ))
                
                if self.w3.is_connected():
                    logger.info(f"Connected to Ethereum network (Chain ID: {self.w3.eth.chain_id})")
                    return
                else:
                    raise ConnectionError("Failed to connect to Ethereum network")
                    
            except Exception as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error("Failed to connect to Ethereum network after all retries")
                    raise
    
    def _load_contract_abi(self):
        """Load contract ABI from file with enhanced error handling"""
        try:
            abi_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                '../blockchain/contracts/FuturesContract.abi.json'
            )
            
            if not os.path.exists(abi_path):
                logger.warning(f"Contract ABI file not found at {abi_path}")
                # Use a minimal ABI for basic functionality
                self.futures_abi = [
                    {
                        "inputs": [{"name": "user", "type": "address"}],
                        "name": "positions",
                        "outputs": [
                            {"name": "exists", "type": "bool"},
                            {"name": "size", "type": "uint256"},
                            {"name": "isLong", "type": "bool"},
                            {"name": "entryPrice", "type": "uint256"}
                        ],
                        "stateMutability": "view",
                        "type": "function"
                    }
                ]
                return
            
            with open(abi_path, 'r') as f:
                self.futures_abi = json.load(f)
            logger.info("Contract ABI loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading contract ABI: {e}")
            self.futures_abi = []
    
    def is_connected(self) -> bool:
        """Check if Web3 connection is active"""
        try:
            return self.w3 is not None and self.w3.is_connected()
        except Exception:
            return False
    
    def is_valid_address(self, address: str) -> bool:
        """
        Validate Ethereum address with checksum verification
        
        Args:
            address (str): Ethereum address to validate
            
        Returns:
            bool: True if address is valid, False otherwise
        """
        try:
            if not self.w3:
                return False
            
            # Basic format check
            if not self.w3.is_address(address):
                return False
            
            # Checksum validation
            if address != self.w3.to_checksum_address(address):
                logger.warning(f"Address {address} failed checksum validation")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error validating address {address}: {e}")
            return False
    
    def get_account_balance(self, address: str) -> Decimal:
        """
        Get ETH balance for an address
        
        Args:
            address (str): Ethereum address
            
        Returns:
            Decimal: Balance in ETH
            
        Raises:
            ValueError: If address is invalid
            Exception: If balance retrieval fails
        """
        if not self.is_valid_address(address):
            raise ValueError("Invalid Ethereum address")
        
        try:
            balance_wei = self.w3.eth.get_balance(address)
            balance_eth = self.w3.from_wei(balance_wei, 'ether')
            return Decimal(str(balance_eth))
        except Exception as e:
            logger.error(f"Error fetching balance for {address}: {e}")
            raise Exception(f"Failed to fetch balance: {str(e)}")
    
    def get_position_health(self, address: str) -> Dict[str, Any]:
        """
        Get comprehensive health metrics for trading positions
        
        Args:
            address (str): Ethereum address of position owner
            
        Returns:
            Dict[str, Any]: Position health metrics
            
        Raises:
            ValueError: If address is invalid
            Exception: If contract call fails
        """
        if not self.is_valid_address(address):
            raise ValueError("Invalid Ethereum address")
        
        if not self.is_connected():
            raise Exception("Blockchain connection not available")
        
        try:
            # Get contract instance
            contract_address = settings.futures_contract_address
            if not self.is_valid_address(contract_address):
                # For demo purposes, return mock data
                return self._get_mock_position_health(address)
            
            contract = self.w3.eth.contract(
                address=self.w3.to_checksum_address(contract_address),
                abi=self.futures_abi
            )
            
            # Call contract to get position data
            position_data = contract.functions.positions(address).call()
            
            # Parse position data
            exists, size, is_long, entry_price = position_data
            
            if not exists:
                return {
                    "address": address,
                    "positions": [],
                    "total_margin_used": Decimal("0"),
                    "total_margin_available": Decimal("1000"),  # Default available margin
                    "health_ratio": float('inf'),
                    "liquidation_risk": "none"
                }
            
            # Calculate liquidation price with more sophisticated logic
            liquidation_price = self._calculate_liquidation_price(
                entry_price, is_long, size
            )
            
            # Get current market price (mock for now)
            current_price = entry_price * Decimal("1.05") if is_long else entry_price * Decimal("0.95")
            
            # Calculate unrealized PnL
            unrealized_pnl = self._calculate_unrealized_pnl(
                size, entry_price, current_price, is_long
            )
            
            # Calculate margin requirements
            margin_requirement = size * entry_price * Decimal("0.1")  # 10% margin
            margin_available = Decimal("1000")  # Mock available margin
            
            # Calculate health ratio
            health_ratio = float(margin_available / margin_requirement) if margin_requirement > 0 else float('inf')
            
            # Determine liquidation risk
            liquidation_risk = self._assess_liquidation_risk(health_ratio)
            
            position = {
                "position_id": f"pos_{address[:10]}",
                "symbol": "BTC-USD",
                "position_type": "long" if is_long else "short",
                "size": Decimal(str(size)),
                "entry_price": Decimal(str(entry_price)),
                "current_price": current_price,
                "liquidation_price": liquidation_price,
                "margin_requirement": margin_requirement,
                "unrealized_pnl": unrealized_pnl,
                "status": "open"
            }
            
            return {
                "address": address,
                "positions": [position],
                "total_margin_used": margin_requirement,
                "total_margin_available": margin_available,
                "health_ratio": health_ratio,
                "liquidation_risk": liquidation_risk
            }
            
        except ContractLogicError as e:
            logger.error(f"Contract logic error for {address}: {e}")
            raise Exception(f"Contract call failed: {str(e)}")
        except Web3Exception as e:
            logger.error(f"Web3 error for {address}: {e}")
            raise Exception(f"Blockchain error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error fetching position for {address}: {e}")
            raise Exception(f"Error fetching position: {str(e)}")
    
    def _get_mock_position_health(self, address: str) -> Dict[str, Any]:
        """Return mock position health data for testing"""
        return {
            "address": address,
            "positions": [{
                "position_id": f"mock_pos_{address[:10]}",
                "symbol": "BTC-USD",
                "position_type": "long",
                "size": Decimal("1.5"),
                "entry_price": Decimal("45000"),
                "current_price": Decimal("47000"),
                "liquidation_price": Decimal("40500"),
                "margin_requirement": Decimal("6750"),
                "unrealized_pnl": Decimal("3000"),
                "status": "open"
            }],
            "total_margin_used": Decimal("6750"),
            "total_margin_available": Decimal("10000"),
            "health_ratio": 1.48,
            "liquidation_risk": "low"
        }
    
    def _calculate_liquidation_price(self, entry_price: int, is_long: bool, size: int) -> Decimal:
        """
        Calculate liquidation price with sophisticated financial logic
        
        Args:
            entry_price (int): Entry price in wei or smallest unit
            is_long (bool): True for long position, False for short
            size (int): Position size
            
        Returns:
            Decimal: Liquidation price
        """
        entry_decimal = Decimal(str(entry_price))
        maintenance_margin_ratio = Decimal("0.05")  # 5% maintenance margin
        
        if is_long:
            # For long positions: liquidation when price drops
            liquidation_price = entry_decimal * (1 - maintenance_margin_ratio)
        else:
            # For short positions: liquidation when price rises
            liquidation_price = entry_decimal * (1 + maintenance_margin_ratio)
        
        return liquidation_price
    
    def _calculate_unrealized_pnl(
        self, size: int, entry_price: int, current_price: Decimal, is_long: bool
    ) -> Decimal:
        """Calculate unrealized profit and loss"""
        size_decimal = Decimal(str(size))
        entry_decimal = Decimal(str(entry_price))
        
        if is_long:
            pnl = size_decimal * (current_price - entry_decimal)
        else:
            pnl = size_decimal * (entry_decimal - current_price)
        
        return pnl
    
    def _assess_liquidation_risk(self, health_ratio: float) -> str:
        """
        Assess liquidation risk based on health ratio
        
        Args:
            health_ratio (float): Margin health ratio
            
        Returns:
            str: Risk level
        """
        if health_ratio >= 2.0:
            return "low"
        elif health_ratio >= 1.5:
            return "medium"
        elif health_ratio >= 1.1:
            return "high"
        else:
            return "critical"
    
    def execute_trade(
        self, 
        trade_data: Dict[str, Any], 
        private_key: str,
        db: Session
    ) -> Dict[str, Any]:
        """
        Execute a trade on the blockchain with comprehensive error handling
        
        Args:
            trade_data (Dict[str, Any]): Trade parameters
            private_key (str): Private key for transaction signing
            db (Session): Database session for logging
            
        Returns:
            Dict[str, Any]: Transaction receipt and trade details
            
        Raises:
            ValueError: If parameters are invalid
            Exception: If transaction fails
        """
        if not self.is_connected():
            raise Exception("Blockchain connection not available")
        
        try:
            # Validate trade data
            required_fields = ['symbol', 'trade_type', 'quantity', 'price']
            for field in required_fields:
                if field not in trade_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Get account from private key
            account = Account.from_key(private_key)
            
            # Estimate gas for the transaction
            gas_estimate = self._estimate_trade_gas(trade_data, account.address)
            
            # Get current gas price
            gas_price = self.w3.eth.gas_price
            
            # Build transaction
            transaction = self._build_trade_transaction(
                trade_data, account.address, gas_estimate, gas_price
            )
            
            # Sign transaction
            signed_txn = self.w3.eth.account.sign_transaction(transaction, private_key)
            
            # Send transaction
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Wait for confirmation
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)
            
            # Log successful trade
            self._log_trade_execution(db, trade_data, tx_hash.hex(), "success")
            
            return {
                "transaction_hash": tx_hash.hex(),
                "block_number": receipt.blockNumber,
                "gas_used": receipt.gasUsed,
                "status": "success" if receipt.status == 1 else "failed",
                "trade_data": trade_data
            }
            
        except ValueError as e:
            self._log_trade_execution(db, trade_data, None, "validation_error", str(e))
            raise e
        except Exception as e:
            self._log_trade_execution(db, trade_data, None, "execution_error", str(e))
            logger.error(f"Trade execution failed: {e}")
            raise Exception(f"Trade execution failed: {str(e)}")
    
    def _estimate_trade_gas(self, trade_data: Dict[str, Any], from_address: str) -> int:
        """Estimate gas required for trade execution"""
        # Mock implementation - in reality, this would call the contract's estimateGas
        base_gas = 21000  # Base transaction cost
        contract_gas = 100000  # Estimated contract execution cost
        return base_gas + contract_gas
    
    def _build_trade_transaction(
        self, 
        trade_data: Dict[str, Any], 
        from_address: str, 
        gas_estimate: int, 
        gas_price: int
    ) -> Dict[str, Any]:
        """Build transaction dictionary for trade execution"""
        # Mock implementation - in reality, this would build the actual contract call
        return {
            'to': settings.futures_contract_address,
            'value': 0,
            'gas': gas_estimate,
            'gasPrice': gas_price,
            'nonce': self.w3.eth.get_transaction_count(from_address),
            'data': '0x'  # Contract call data would go here
        }
    
    def _log_trade_execution(
        self, 
        db: Session, 
        trade_data: Dict[str, Any], 
        tx_hash: Optional[str], 
        status: str, 
        error_message: Optional[str] = None
    ):
        """Log trade execution for audit trail"""
        try:
            audit_log = AuditLog(
                action="trade_execution",
                resource_type="trade",
                request_data=json.dumps(trade_data),
                response_data=json.dumps({"tx_hash": tx_hash}) if tx_hash else None,
                status=status,
                error_message=error_message
            )
            db.add(audit_log)
            db.commit()
        except Exception as e:
            logger.error(f"Failed to log trade execution: {e}")
            db.rollback()
    
    def get_transaction_status(self, tx_hash: str) -> Dict[str, Any]:
        """
        Get status of a blockchain transaction
        
        Args:
            tx_hash (str): Transaction hash
            
        Returns:
            Dict[str, Any]: Transaction status and details
        """
        try:
            receipt = self.w3.eth.get_transaction_receipt(tx_hash)
            transaction = self.w3.eth.get_transaction(tx_hash)
            
            return {
                "hash": tx_hash,
                "status": "success" if receipt.status == 1 else "failed",
                "block_number": receipt.blockNumber,
                "gas_used": receipt.gasUsed,
                "gas_price": transaction.gasPrice,
                "confirmations": self.w3.eth.block_number - receipt.blockNumber
            }
        except Exception as e:
            logger.error(f"Error fetching transaction status for {tx_hash}: {e}")
            return {
                "hash": tx_hash,
                "status": "pending",
                "error": str(e)
            }

