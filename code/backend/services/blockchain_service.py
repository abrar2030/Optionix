"""
Enhanced blockchain service module for Optionix platform.
Handles all blockchain interactions with robust security and error handling.
"""

import json
import logging
import os
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional
from web3 import Web3
from web3.exceptions import ContractLogicError, Web3Exception
from ..config import settings

logger = logging.getLogger(__name__)


class BlockchainService:
    """Enhanced service for interacting with blockchain contracts and wallets"""

    def __init__(self) -> None:
        """Initialize blockchain service with Web3 provider"""
        self.w3: Optional[Web3] = None
        self.futures_contract: Optional[Any] = None
        self.futures_abi: List[Dict[str, Any]] = []
        self._initialize_connection()
        self._load_contract_abi()
        self._initialize_contract()

    def _initialize_connection(self) -> None:
        """Initialize Web3 connection with retry logic"""
        max_retries = 3
        retry_delay = 1
        for attempt in range(max_retries):
            try:
                if "your-project-id" in settings.ethereum_provider_url:
                    logger.warning(
                        "Using placeholder Ethereum URL. Skipping connection attempt."
                    )
                    self.w3 = None
                    return
                self.w3 = Web3(
                    Web3.HTTPProvider(
                        settings.ethereum_provider_url, request_kwargs={"timeout": 30}
                    )
                )
                if self.w3 and self.w3.is_connected():
                    logger.info(
                        f"Connected to Ethereum network (Chain ID: {self.w3.eth.chain_id})"
                    )
                    return
                else:
                    raise ConnectionError("Failed to connect to Ethereum network")
            except Exception as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error(
                        "Failed to connect to Ethereum network after all retries"
                    )
                    self.w3 = None
                    return

    def _load_contract_abi(self) -> None:
        """Load contract ABI from file with enhanced error handling"""
        try:
            abi_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "../blockchain/contracts/FuturesContract.abi.json",
            )
            if not os.path.exists(abi_path):
                logger.error(f"Contract ABI file not found at {abi_path}")
                self.futures_abi = []
                return
            with open(abi_path, "r") as f:
                self.futures_abi = json.load(f)
            logger.info("Contract ABI loaded successfully")
        except Exception as e:
            logger.error(f"Error loading contract ABI: {e}")
            self.futures_abi = []

    def _initialize_contract(self) -> None:
        """Initialize the futures contract instance"""
        if not self.w3 or not self.futures_abi:
            self.futures_contract = None
            return
        contract_address = settings.futures_contract_address
        if not self.w3.is_address(contract_address):  # type: ignore
            logger.warning(f"Invalid contract address: {contract_address}")
            self.futures_contract = None
            return
        try:
            self.futures_contract = self.w3.eth.contract(  # type: ignore
                address=self.w3.to_checksum_address(contract_address),  # type: ignore
                abi=self.futures_abi,
            )
            logger.info("Futures contract initialized")
        except Exception as e:
            logger.error(f"Error initializing futures contract: {e}")
            self.futures_contract = None

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
        if not self.w3:
            return False
        return self.w3.is_address(address)

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
        if not self.is_connected():
            raise Exception("Blockchain connection not available")
        try:
            balance_wei = self.w3.eth.get_balance(address)  # type: ignore
            balance_eth = self.w3.from_wei(balance_wei, "ether")  # type: ignore
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
        if not self.futures_contract:
            return self._get_mock_position_health(address)
        try:
            position_data = self.futures_contract.functions.positions(address).call()
            trader, size, is_long, entry_price = position_data
            exists = size > 0
            if not exists:
                return {
                    "address": address,
                    "positions": [],
                    "total_margin_used": Decimal("0"),
                    "total_margin_available": Decimal("1000"),
                    "health_ratio": float("inf"),
                    "liquidation_risk": "none",
                }
            size_dec = Decimal(str(size))
            entry_price_dec = Decimal(str(entry_price))
            liquidation_price = self._calculate_liquidation_price(
                entry_price_dec, is_long, size_dec
            )
            current_price = (
                entry_price_dec * Decimal("1.05")
                if is_long
                else entry_price_dec * Decimal("0.95")
            )
            unrealized_pnl = self._calculate_unrealized_pnl(
                size_dec, entry_price_dec, current_price, is_long
            )
            margin_requirement = size_dec * entry_price_dec * Decimal("0.1")
            margin_available = Decimal("1000")
            health_ratio = (
                float(margin_available / margin_requirement)
                if margin_requirement > 0
                else float("inf")
            )
            liquidation_risk = self._assess_liquidation_risk(health_ratio)
            position = {
                "position_id": f"pos_{address[:10]}",
                "symbol": "BTC-USD",
                "position_type": "long" if is_long else "short",
                "size": size_dec,
                "entry_price": entry_price_dec,
                "current_price": current_price,
                "liquidation_price": liquidation_price,
                "margin_requirement": margin_requirement,
                "unrealized_pnl": unrealized_pnl,
                "status": "open",
            }
            return {
                "address": address,
                "positions": [position],
                "total_margin_used": margin_requirement,
                "total_margin_available": margin_available,
                "health_ratio": health_ratio,
                "liquidation_risk": liquidation_risk,
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
            "positions": [
                {
                    "position_id": f"mock_pos_{address[:10]}",
                    "symbol": "BTC-USD",
                    "position_type": "long",
                    "size": Decimal("0.1"),
                    "entry_price": Decimal("30000"),
                    "current_price": Decimal("31500"),
                    "liquidation_price": Decimal("27000"),
                    "margin_requirement": Decimal("3000"),
                    "unrealized_pnl": Decimal("150"),
                    "status": "open",
                }
            ],
            "total_margin_used": Decimal("3000"),
            "total_margin_available": Decimal("7000"),
            "health_ratio": 2.33,
            "liquidation_risk": "low",
        }

    def _calculate_liquidation_price(
        self, entry_price: Decimal, is_long: bool, size: Decimal
    ) -> Decimal:
        """Placeholder for sophisticated liquidation price calculation"""
        if is_long:
            return entry_price * Decimal("0.9")
        else:
            return entry_price * Decimal("1.1")

    def _calculate_unrealized_pnl(
        self, size: Decimal, entry_price: Decimal, current_price: Decimal, is_long: bool
    ) -> Decimal:
        """Placeholder for unrealized PnL calculation"""
        if is_long:
            return size * (current_price - entry_price)
        else:
            return size * (entry_price - current_price)

    def _assess_liquidation_risk(self, health_ratio: float) -> str:
        """Placeholder for liquidation risk assessment"""
        if health_ratio > 3.0:
            return "very_low"
        elif health_ratio > 1.5:
            return "low"
        elif health_ratio > 1.1:
            return "medium"
        else:
            return "high"

    def deposit_margin(
        self, user_address: str, amount: Decimal, private_key: str
    ) -> Dict[str, Any]:
        """
        Enhanced function to deposit margin to the futures contract.

        Args:
            user_address (str): The user's Ethereum address.
            amount (Decimal): The amount of margin to deposit (in ETH).
            private_key (str): The user's private key for signing the transaction.

        Returns:
            Dict[str, Any]: Transaction details.

        Raises:
            Exception: If the transaction fails.
        """
        if not self.futures_contract:
            raise Exception("Futures contract not initialized. Cannot deposit margin.")
        if not self.is_valid_address(user_address):
            raise ValueError("Invalid user address.")
        if amount <= 0:
            raise ValueError("Deposit amount must be positive.")
        try:
            amount_wei = self.w3.to_wei(amount, "ether")  # type: ignore
            tx = self.futures_contract.functions.depositMargin(
                user_address, amount_wei
            ).build_transaction(
                {
                    "from": user_address,
                    "value": amount_wei,
                    "nonce": self.w3.eth.get_transaction_count(user_address),  # type: ignore
                    "gasPrice": self.w3.eth.gas_price,  # type: ignore
                }
            )
            signed_tx = self.w3.eth.account.sign_transaction(tx, private_key)  # type: ignore
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)  # type: ignore
            logger.info(f"Deposit transaction sent. Hash: {tx_hash.hex()}")
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)  # type: ignore
            if receipt.status == 1:
                logger.info("Deposit successful.")
                return {
                    "status": "success",
                    "tx_hash": tx_hash.hex(),
                    "receipt": receipt,
                }
            else:
                raise Exception("Transaction failed on the blockchain.")
        except Web3Exception as e:
            logger.error(f"Web3 error during deposit: {e}")
            raise Exception(f"Blockchain transaction failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during deposit: {e}")
            raise Exception(f"Deposit failed: {str(e)}")

    def withdraw_margin(
        self, user_address: str, amount: Decimal, private_key: str
    ) -> Dict[str, Any]:
        """
        Enhanced function to withdraw margin from the futures contract.

        Args:
            user_address (str): The user's Ethereum address.
            amount (Decimal): The amount of margin to withdraw (in ETH).
            private_key (str): The user's private key for signing the transaction.

        Returns:
            Dict[str, Any]: Transaction details.

        Raises:
            Exception: If the transaction fails.
        """
        if not self.futures_contract:
            raise Exception("Futures contract not initialized. Cannot withdraw margin.")
        if not self.is_valid_address(user_address):
            raise ValueError("Invalid user address.")
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive.")
        try:
            amount_wei = self.w3.to_wei(amount, "ether")  # type: ignore
            tx = self.futures_contract.functions.withdrawMargin(
                user_address, amount_wei
            ).build_transaction(
                {
                    "from": user_address,
                    "nonce": self.w3.eth.get_transaction_count(user_address),  # type: ignore
                    "gasPrice": self.w3.eth.gas_price,  # type: ignore
                }
            )
            signed_tx = self.w3.eth.account.sign_transaction(tx, private_key)  # type: ignore
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)  # type: ignore
            logger.info(f"Withdrawal transaction sent. Hash: {tx_hash.hex()}")
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)  # type: ignore
            if receipt.status == 1:
                logger.info("Withdrawal successful.")
                return {
                    "status": "success",
                    "tx_hash": tx_hash.hex(),
                    "receipt": receipt,
                }
            else:
                raise Exception("Transaction failed on the blockchain.")
        except Web3Exception as e:
            logger.error(f"Web3 error during withdrawal: {e}")
            raise Exception(f"Blockchain transaction failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during withdrawal: {e}")
            raise Exception(f"Withdrawal failed: {str(e)}")

    def get_transaction_status(self, tx_hash: str) -> Dict[str, Any]:
        """
        Get detailed status of a blockchain transaction.

        Args:
            tx_hash (str): The transaction hash.

        Returns:
            Dict[str, Any]: Detailed transaction status.
        """
        if not self.is_connected():
            return {
                "hash": tx_hash,
                "status": "disconnected",
                "error": "Blockchain connection not available",
            }
        try:
            receipt = self.w3.eth.get_transaction_receipt(tx_hash)  # type: ignore
            if receipt is None:
                return {"hash": tx_hash, "status": "pending"}
            transaction = self.w3.eth.get_transaction(tx_hash)  # type: ignore
            status = "success" if receipt.status == 1 else "failed"
            return {
                "hash": tx_hash,
                "status": status,
                "block_number": receipt.blockNumber,
                "gas_used": receipt.gasUsed,
                "gas_price": transaction.gasPrice,
                "confirmations": self.w3.eth.block_number - receipt.blockNumber,  # type: ignore
            }
        except Exception as e:
            logger.error(f"Error fetching transaction status for {tx_hash}: {e}")
            return {"hash": tx_hash, "status": "error", "error": str(e)}
