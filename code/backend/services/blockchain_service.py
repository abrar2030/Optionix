"""
Blockchain service module for Optionix platform.
Handles all blockchain interactions and abstracts Web3 functionality.
"""
from web3 import Web3
import json
import os
import logging

logger = logging.getLogger(__name__)

class BlockchainService:
    """Service for interacting with blockchain contracts and wallets"""
    
    def __init__(self, provider_url='http://localhost:8545'):
        """
        Initialize blockchain service with Web3 provider
        
        Args:
            provider_url (str): URL for Web3 HTTP provider
        """
        self.w3 = Web3(Web3.HTTPProvider(provider_url))
        self.futures_contract = None
        self.futures_abi = []
        self._load_contract_abi()
    
    def _load_contract_abi(self):
        """Load contract ABI from file"""
        try:
            abi_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                '../blockchain/contracts/FuturesContract.abi.json'
            )
            with open(abi_path) as f:
                self.futures_abi = json.load(f)
            logger.info("Contract ABI loaded successfully")
        except Exception as e:
            logger.error(f"Error loading contract ABI: {e}")
    
    def is_valid_address(self, address):
        """
        Check if an Ethereum address is valid
        
        Args:
            address (str): Ethereum address to validate
            
        Returns:
            bool: True if address is valid, False otherwise
        """
        return self.w3.is_address(address)
    
    def get_position_health(self, address):
        """
        Get health metrics for a trading position
        
        Args:
            address (str): Ethereum address of position owner
            
        Returns:
            dict: Position health metrics
            
        Raises:
            ValueError: If address is invalid
            Exception: If contract call fails
        """
        if not self.is_valid_address(address):
            raise ValueError("Invalid Ethereum address")
        
        try:
            # In production, this would be a deployed contract address
            contract_address = '0x0000000000000000000000000000000000000000'
            contract = self.w3.eth.contract(address=contract_address, abi=self.futures_abi)
            position = contract.functions.positions(address).call()
            
            # Calculate liquidation price based on position
            entry_price = position[3]
            is_long = position[2]
            liquidation_price = entry_price * 0.9 if is_long else entry_price * 1.1
            
            return {
                "address": address,
                "size": position[1],
                "is_long": is_long,
                "entry_price": entry_price,
                "liquidation_price": liquidation_price
            }
        except Exception as e:
            logger.error(f"Error fetching position for {address}: {e}")
            raise Exception(f"Error fetching position: {str(e)}")
    
    def execute_trade(self, trade_data):
        """
        Execute a trade on the blockchain
        
        Args:
            trade_data (dict): Trade parameters
            
        Returns:
            dict: Transaction receipt
            
        Raises:
            ValueError: If parameters are invalid
            Exception: If transaction fails
        """
        # Implementation would depend on specific contract methods
        # This is a placeholder for the actual implementation
        raise NotImplementedError("Trade execution not implemented yet")
