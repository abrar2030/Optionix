"""
Execution Engine for Optionix platform.

This module implements the core trade execution engine with order management,
smart routing, and execution algorithms.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import uuid
import logging
import threading
import time
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    """Enum for order status values"""
    CREATED = "CREATED"
    VALIDATED = "VALIDATED"
    REJECTED = "REJECTED"
    PENDING = "PENDING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"

class OrderType(Enum):
    """Enum for order types"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

class OrderSide(Enum):
    """Enum for order sides"""
    BUY = "BUY"
    SELL = "SELL"

class TimeInForce(Enum):
    """Enum for time in force options"""
    GTC = "GOOD_TILL_CANCEL"
    IOC = "IMMEDIATE_OR_CANCEL"
    FOK = "FILL_OR_KILL"
    GTD = "GOOD_TILL_DATE"

class ExecutionAlgorithm(Enum):
    """Enum for execution algorithms"""
    MARKET = "MARKET"
    TWAP = "TWAP"
    VWAP = "VWAP"
    IMPLEMENTATION_SHORTFALL = "IMPLEMENTATION_SHORTFALL"
    PERCENTAGE_OF_VOLUME = "PERCENTAGE_OF_VOLUME"

class Order:
    """
    Class representing an order in the system.
    """
    
    def __init__(self, instrument, quantity, side, order_type=OrderType.MARKET, 
                 price=None, stop_price=None, time_in_force=TimeInForce.GTC,
                 expiry_time=None, account_id=None, algorithm=ExecutionAlgorithm.MARKET,
                 algorithm_params=None):
        """
        Initialize a new order.
        
        Args:
            instrument (str): Instrument identifier
            quantity (float): Order quantity
            side (OrderSide): Order side (BUY or SELL)
            order_type (OrderType, optional): Order type
            price (float, optional): Limit price for limit orders
            stop_price (float, optional): Stop price for stop orders
            time_in_force (TimeInForce, optional): Time in force
            expiry_time (datetime, optional): Expiry time for GTD orders
            account_id (str, optional): Account identifier
            algorithm (ExecutionAlgorithm, optional): Execution algorithm
            algorithm_params (dict, optional): Parameters for execution algorithm
        """
        self.order_id = str(uuid.uuid4())
        self.instrument = instrument
        self.quantity = quantity
        self.side = side if isinstance(side, OrderSide) else OrderSide(side)
        self.order_type = order_type if isinstance(order_type, OrderType) else OrderType(order_type)
        self.price = price
        self.stop_price = stop_price
        self.time_in_force = time_in_force if isinstance(time_in_force, TimeInForce) else TimeInForce(time_in_force)
        self.expiry_time = expiry_time
        self.account_id = account_id or "default"
        self.algorithm = algorithm if isinstance(algorithm, ExecutionAlgorithm) else ExecutionAlgorithm(algorithm)
        self.algorithm_params = algorithm_params or {}
        
        # Order state
        self.status = OrderStatus.CREATED
        self.creation_time = datetime.now()
        self.last_updated = self.creation_time
        self.executed_quantity = 0
        self.average_price = 0
        self.fills = []
        self.rejection_reason = None
        self.parent_order_id = None
        self.child_order_ids = []
    
    def validate(self):
        """
        Validate the order.
        
        Returns:
            bool: True if valid, False otherwise
            str: Rejection reason if invalid
        """
        # Check required fields
        if not self.instrument:
            return False, "Missing instrument"
        
        if not self.quantity or self.quantity <= 0:
            return False, "Invalid quantity"
        
        # Check price for limit orders
        if self.order_type == OrderType.LIMIT and (self.price is None or self.price <= 0):
            return False, "Missing or invalid price for limit order"
        
        # Check stop price for stop orders
        if self.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and (self.stop_price is None or self.stop_price <= 0):
            return False, "Missing or invalid stop price for stop order"
        
        # Check expiry time for GTD orders
        if self.time_in_force == TimeInForce.GTD and (self.expiry_time is None or self.expiry_time <= datetime.now()):
            return False, "Missing or invalid expiry time for GTD order"
        
        return True, None
    
    def update_status(self, status, reason=None):
        """
        Update order status.
        
        Args:
            status (OrderStatus): New status
            reason (str, optional): Reason for status change
        """
        self.status = status if isinstance(status, OrderStatus) else OrderStatus(status)
        self.last_updated = datetime.now()
        
        if status == OrderStatus.REJECTED:
            self.rejection_reason = reason
        
        logger.info(f"Order {self.order_id} status updated to {status.value}" + 
                   (f" - Reason: {reason}" if reason else ""))
    
    def add_fill(self, quantity, price, timestamp=None):
        """
        Add a fill to the order.
        
        Args:
            quantity (float): Fill quantity
            price (float): Fill price
            timestamp (datetime, optional): Fill timestamp
        """
        timestamp = timestamp or datetime.now()
        
        # Create fill record
        fill = {
            "fill_id": str(uuid.uuid4()),
            "order_id": self.order_id,
            "quantity": quantity,
            "price": price,
            "timestamp": timestamp
        }
        
        # Add to fills list
        self.fills.append(fill)
        
        # Update executed quantity and average price
        self.executed_quantity += quantity
        
        # Calculate new average price
        self.average_price = sum(f["quantity"] * f["price"] for f in self.fills) / self.executed_quantity
        
        # Update status
        if self.executed_quantity >= self.quantity:
            self.update_status(OrderStatus.FILLED)
        elif self.executed_quantity > 0:
            self.update_status(OrderStatus.PARTIALLY_FILLED)
        
        logger.info(f"Order {self.order_id} filled: {quantity} @ {price}")
    
    def cancel(self, reason=None):
        """
        Cancel the order.
        
        Args:
            reason (str, optional): Cancellation reason
        
        Returns:
            bool: True if cancelled, False otherwise
        """
        if self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
            return False
        
        self.update_status(OrderStatus.CANCELLED, reason)
        return True
    
    def to_dict(self):
        """
        Convert order to dictionary.
        
        Returns:
            dict: Order as dictionary
        """
        return {
            "order_id": self.order_id,
            "instrument": self.instrument,
            "quantity": self.quantity,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "price": self.price,
            "stop_price": self.stop_price,
            "time_in_force": self.time_in_force.value,
            "expiry_time": self.expiry_time.isoformat() if self.expiry_time else None,
            "account_id": self.account_id,
            "algorithm": self.algorithm.value,
            "algorithm_params": self.algorithm_params,
            "status": self.status.value,
            "creation_time": self.creation_time.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "executed_quantity": self.executed_quantity,
            "average_price": self.average_price,
            "fills": self.fills,
            "rejection_reason": self.rejection_reason,
            "parent_order_id": self.parent_order_id,
            "child_order_ids": self.child_order_ids
        }


class OrderManager:
    """
    System for managing order lifecycle.
    """
    
    def __init__(self):
        """Initialize order manager."""
        self.orders = {}
        self.order_history = {}
        self._lock = threading.RLock()
    
    def create_order(self, order_params):
        """
        Create a new order.
        
        Args:
            order_params (dict or Order): Order parameters or Order object
            
        Returns:
            Order: Created order
        """
        with self._lock:
            # Create order object if parameters provided
            if isinstance(order_params, dict):
                order = Order(**order_params)
            else:
                order = order_params
            
            # Validate order
            is_valid, reason = order.validate()
            
            if is_valid:
                order.update_status(OrderStatus.VALIDATED)
            else:
                order.update_status(OrderStatus.REJECTED, reason)
                logger.warning(f"Order validation failed: {reason}")
            
            # Store order
            self.orders[order.order_id] = order
            self.order_history[order.order_id] = [order.to_dict()]
            
            return order
    
    def get_order(self, order_id):
        """
        Get order by ID.
        
        Args:
            order_id (str): Order ID
            
        Returns:
            Order: Order object or None if not found
        """
        return self.orders.get(order_id)
    
    def update_order(self, order_id, updates):
        """
        Update an existing order.
        
        Args:
            order_id (str): Order ID
            updates (dict): Updates to apply
            
        Returns:
            Order: Updated order or None if not found
        """
        with self._lock:
            order = self.get_order(order_id)
            
            if not order:
                logger.warning(f"Order {order_id} not found for update")
                return None
            
            # Check if order can be updated
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
                logger.warning(f"Cannot update order {order_id} with status {order.status.value}")
                return order
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(order, key):
                    setattr(order, key, value)
            
            # Revalidate order
            is_valid, reason = order.validate()
            
            if is_valid:
                order.update_status(OrderStatus.VALIDATED)
            else:
                order.update_status(OrderStatus.REJECTED, reason)
                logger.warning(f"Order update validation failed: {reason}")
            
            # Update history
            self.order_history[order_id].append(order.to_dict())
            
            return order
    
    def cancel_order(self, order_id, reason=None):
        """
        Cancel an order.
        
        Args:
            order_id (str): Order ID
            reason (str, optional): Cancellation reason
            
        Returns:
            bool: True if cancelled, False otherwise
        """
        with self._lock:
            order = self.get_order(order_id)
            
            if not order:
                logger.warning(f"Order {order_id} not found for cancellation")
                return False
            
            result = order.cancel(reason)
            
            if result:
                # Update history
                self.order_history[order_id].append(order.to_dict())
            
            return result
    
    def add_fill(self, order_id, quantity, price, timestamp=None):
        """
        Add a fill to an order.
        
        Args:
            order_id (str): Order ID
            quantity (float): Fill quantity
            price (float): Fill price
            timestamp (datetime, optional): Fill timestamp
            
        Returns:
            bool: True if fill added, False otherwise
        """
        with self._lock:
            order = self.get_order(order_id)
            
            if not order:
                logger.warning(f"Order {order_id} not found for fill")
                return False
            
            # Check if order can be filled
            if order.status in [OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
                logger.warning(f"Cannot fill order {order_id} with status {order.status.value}")
                return False
            
            # Add fill
            order.add_fill(quantity, price, timestamp)
            
            # Update history
            self.order_history[order_id].append(order.to_dict())
            
            return True
    
    def get_order_history(self, order_id):
        """
        Get order history.
        
        Args:
            order_id (str): Order ID
            
        Returns:
            list: Order history or empty list if not found
        """
        return self.order_history.get(order_id, [])
    
    def get_orders_by_status(self, status):
        """
        Get orders by status.
        
        Args:
            status (OrderStatus): Order status
            
        Returns:
            list: Orders with the specified status
        """
        status = status if isinstance(status, OrderStatus) else OrderStatus(status)
        return [order for order in self.orders.values() if order.status == status]
    
    def get_orders_by_account(self, account_id):
        """
        Get orders by account.
        
        Args:
            account_id (str): Account ID
            
        Returns:
            list: Orders for the specified account
        """
        return [order for order in self.orders.values() if order.account_id == account_id]
    
    def get_orders_by_instrument(self, instrument):
        """
        Get orders by instrument.
        
        Args:
            instrument (str): Instrument identifier
            
        Returns:
            list: Orders for the specified instrument
        """
        return [order for order in self.orders.values() if order.instrument == instrument]
    
    def create_child_orders(self, parent_order_id, child_orders):
        """
        Create child orders for a parent order.
        
        Args:
            parent_order_id (str): Parent order ID
            child_orders (list): List of child order parameters
            
        Returns:
            list: Created child orders
        """
        with self._lock:
            parent_order = self.get_order(parent_order_id)
            
            if not parent_order:
                logger.warning(f"Parent order {parent_order_id} not found")
                return []
            
            created_orders = []
            
            for params in child_orders:
                # Create child order
                child_order = self.create_order(params)
                
                # Link to parent
                child_order.parent_order_id = parent_order_id
                parent_order.child_order_ids.append(child_order.order_id)
                
                created_orders.append(child_order)
            
            # Update parent order history
            self.order_history[parent_order_id].append(parent_order.to_dict())
            
            return created_orders


class ExecutionEngine:
    """
    Core engine for processing and executing trades.
    """
    
    def __init__(self, config=None):
        """
        Initialize execution engine.
        
        Args:
            config (dict, optional): Configuration parameters
        """
        self.config = config or {}
        self.order_manager = OrderManager()
        self.market_data = {}
        self.execution_threads = {}
        self._lock = threading.RLock()
    
    def submit_order(self, order_params):
        """
        Submit an order for execution.
        
        Args:
            order_params (dict or Order): Order parameters or Order object
            
        Returns:
            dict: Order submission result
        """
        # Create order
        order = self.order_manager.create_order(order_params)
        
        # Check if order was rejected during validation
        if order.status == OrderStatus.REJECTED:
            return {
                "status": "rejected",
                "order_id": order.order_id,
                "reason": order.rejection_reason
            }
        
        # Update status to pending
        order.update_status(OrderStatus.PENDING)
        
        # Execute order based on algorithm
        if order.algorithm == ExecutionAlgorithm.MARKET:
            # Execute immediately
            self._execute_market_order(order)
        else:
            # Start execution thread for algorithmic execution
            self._start_execution_thread(order)
        
        return {
            "status": "accepted",
            "order_id": order.order_id
        }
    
    def _execute_market_order(self, order):
        """
        Execute a market order immediately.
        
        Args:
            order (Order): Order to execute
        """
        # Simulate market execution
        # In a real system, this would interact with exchange/broker APIs
        
        # Get current market price
        market_price = self._get_market_price(order.instrument)
        
        if market_price is None:
            order.update_status(OrderStatus.REJECTED, "Unable to determine market price")
            return
        
        # Add fill
        self.order_manager.add_fill(order.order_id, order.quantity, market_price)
    
    def _start_execution_thread(self, order):
        """
        Start execution thread for algorithmic execution.
        
        Args:
            order (Order): Order to execute
        """
        with self._lock:
            # Create execution thread
            thread = threading.Thread(
                target=self._run_execution_algorithm,
                args=(order,),
                daemon=True
            )
            
            # Store thread
            self.execution_threads[order.order_id] = {
                "thread": thread,
                "stop_flag": threading.Event()
            }
            
            # Start thread
            thread.start()
    
    def _run_execution_algorithm(self, order):
        """
        Run execution algorithm for an order.
        
        Args:
            order (Order): Order to execute
        """
        try:
            # Get stop flag
            stop_flag = self.execution_threads[order.order_id]["stop_flag"]
            
            # Select algorithm
            if order.algorithm == ExecutionAlgorithm.TWAP:
                self._execute_twap(order, stop_flag)
            elif order.algorithm == ExecutionAlgorithm.VWAP:
                self._execute_vwap(order, stop_flag)
            elif order.algorithm == ExecutionAlgorithm.IMPLEMENTATION_SHORTFALL:
                self._execute_implementation_shortfall(order, stop_flag)
            elif order.algorithm == ExecutionAlgorithm.PERCENTAGE_OF_VOLUME:
                self._execute_percentage_of_volume(order, stop_flag)
            else:
                # Fallback to market execution
                self._execute_market_order(order)
        
        except Exception as e:
            logger.error(f"Error executing order {order.order_id}: {str(e)}")
            order.update_status(OrderStatus.REJECTED, f"Execution error: {str(e)}")
        
        finally:
            # Clean up thread reference
            with self._lock:
                if order.order_id in self.execution_threads:
                    del self.execution_threads[order.order_id]
    
    def _execute_twap(self, order, stop_flag):
        """
        Execute order using TWAP (Time-Weighted Average Price) algorithm.
        
        Args:
            order (Order): Order to execute
            stop_flag (threading.Event): Stop flag for early termination
        """
        # Get algorithm parameters
        duration_minutes = order.algorithm_params.get("duration_minutes", 60)
        num_slices = order.algorithm_params.get("num_slices", 10)
        
        # Calculate slice size and interval
        slice_size = order.quantity / num_slices
        interval_seconds = (duration_minutes * 60) / num_slices
        
        # Execute slices
        remaining_quantity = order.quantity
        
        for i in range(num_slices):
            # Check if execution should stop
            if stop_flag.is_set() or order.status in [OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
                break
            
            # Calculate slice quantity (last slice may be different due to rounding)
            if i == num_slices - 1:
                slice_quantity = remaining_quantity
            else:
                slice_quantity = slice_size
            
            # Get current market price
            market_price = self._get_market_price(order.instrument)
            
            if market_price is None:
                logger.warning(f"Unable to determine market price for {order.instrument}")
                continue
            
            # Add fill
            self.order_manager.add_fill(order.order_id, slice_quantity, market_price)
            
            # Update remaining quantity
            remaining_quantity -= slice_quantity
            
            # Wait for next slice (unless this is the last slice)
            if i < num_slices - 1 and not stop_flag.is_set():
                time.sleep(interval_seconds)
    
    def _execute_vwap(self, order, stop_flag):
        """
        Execute order using VWAP (Volume-Weighted Average Price) algorithm.
        
        Args:
            order (Order): Order to execute
            stop_flag (threading.Event): Stop flag for early termination
        """
        # Get algorithm parameters
        duration_minutes = order.algorithm_params.get("duration_minutes", 60)
        num_slices = order.algorithm_params.get("num_slices", 10)
        
        # Get volume profile (in a real system, this would be based on historical data)
        volume_profile = self._get_volume_profile(order.instrument, num_slices)
        
        # Calculate slice sizes based on volume profile
        total_profile = sum(volume_profile)
        slice_sizes = [order.quantity * (v / total_profile) for v in volume_profile]
        
        # Calculate interval
        interval_seconds = (duration_minutes * 60) / num_slices
        
        # Execute slices
        remaining_quantity = order.quantity
        
        for i in range(num_slices):
            # Check if execution should stop
            if stop_flag.is_set() or order.status in [OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
                break
            
            # Calculate slice quantity (last slice may be different due to rounding)
            if i == num_slices - 1:
                slice_quantity = remaining_quantity
            else:
                slice_quantity = slice_sizes[i]
            
            # Get current market price
            market_price = self._get_market_price(order.instrument)
            
            if market_price is None:
                logger.warning(f"Unable to determine market price for {order.instrument}")
                continue
            
            # Add fill
            self.order_manager.add_fill(order.order_id, slice_quantity, market_price)
            
            # Update remaining quantity
            remaining_quantity -= slice_quantity
            
            # Wait for next slice (unless this is the last slice)
            if i < num_slices - 1 and not stop_flag.is_set():
                time.sleep(interval_seconds)
    
    def _execute_implementation_shortfall(self, order, stop_flag):
        """
        Execute order using Implementation Shortfall algorithm.
        
        Args:
            order (Order): Order to execute
            stop_flag (threading.Event): Stop flag for early termination
        """
        # Get algorithm parameters
        urgency = order.algorithm_params.get("urgency", "medium")
        max_participation_rate = order.algorithm_params.get("max_participation_rate", 0.3)
        
        # Determine execution schedule based on urgency
        if urgency == "high":
            # Execute quickly
            market_impact_threshold = 0.05  # 5%
            initial_size = 0.5  # 50% of order
        elif urgency == "low":
            # Execute slowly
            market_impact_threshold = 0.02  # 2%
            initial_size = 0.2  # 20% of order
        else:  # medium
            # Balanced approach
            market_impact_threshold = 0.03  # 3%
            initial_size = 0.3  # 30% of order
        
        # Get initial market price
        initial_price = self._get_market_price(order.instrument)
        
        if initial_price is None:
            order.update_status(OrderStatus.REJECTED, "Unable to determine market price")
            return
        
        # Execute initial block
        initial_quantity = order.quantity * initial_size
        self.order_manager.add_fill(order.order_id, initial_quantity, initial_price)
        
        # Execute remaining quantity using adaptive strategy
        remaining_quantity = order.quantity - initial_quantity
        
        while remaining_quantity > 0 and not stop_flag.is_set():
            # Check if order status has changed
            if order.status in [OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
                break
            
            # Get current market price and volume
            current_price = self._get_market_price(order.instrument)
            current_volume = self._get_market_volume(order.instrument)
            
            if current_price is None or current_volume is None:
                time.sleep(5)  # Wait and retry
                continue
            
            # Calculate price impact
            price_impact = abs(current_price - initial_price) / initial_price
            
            # Determine execution size based on market conditions
            if price_impact > market_impact_threshold:
                # Market is moving against us, slow down
                execution_size = min(remaining_quantity, current_volume * max_participation_rate * 0.5)
            else:
                # Normal execution
                execution_size = min(remaining_quantity, current_volume * max_participation_rate)
            
            # Ensure minimum execution size
            execution_size = max(execution_size, 1)
            
            # Execute slice
            self.order_manager.add_fill(order.order_id, execution_size, current_price)
            
            # Update remaining quantity
            remaining_quantity -= execution_size
            
            # Wait before next execution
            time.sleep(10)
    
    def _execute_percentage_of_volume(self, order, stop_flag):
        """
        Execute order using Percentage of Volume algorithm.
        
        Args:
            order (Order): Order to execute
            stop_flag (threading.Event): Stop flag for early termination
        """
        # Get algorithm parameters
        target_percentage = order.algorithm_params.get("target_percentage", 0.1)  # 10%
        min_execution_size = order.algorithm_params.get("min_execution_size", 1)
        
        # Execute order in slices
        remaining_quantity = order.quantity
        
        while remaining_quantity > 0 and not stop_flag.is_set():
            # Check if order status has changed
            if order.status in [OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
                break
            
            # Get current market volume
            market_volume = self._get_market_volume(order.instrument)
            
            if market_volume is None or market_volume == 0:
                time.sleep(5)  # Wait and retry
                continue
            
            # Calculate execution size
            execution_size = min(remaining_quantity, max(market_volume * target_percentage, min_execution_size))
            
            # Get current market price
            market_price = self._get_market_price(order.instrument)
            
            if market_price is None:
                time.sleep(5)  # Wait and retry
                continue
            
            # Execute slice
            self.order_manager.add_fill(order.order_id, execution_size, market_price)
            
            # Update remaining quantity
            remaining_quantity -= execution_size
            
            # Wait before next execution
            time.sleep(10)
    
    def cancel_order(self, order_id, reason=None):
        """
        Cancel an order.
        
        Args:
            order_id (str): Order ID
            reason (str, optional): Cancellation reason
            
        Returns:
            dict: Cancellation result
        """
        # Cancel order
        result = self.order_manager.cancel_order(order_id, reason)
        
        # Stop execution thread if running
        with self._lock:
            if order_id in self.execution_threads:
                self.execution_threads[order_id]["stop_flag"].set()
        
        return {
            "status": "success" if result else "failed",
            "order_id": order_id
        }
    
    def get_order_status(self, order_id):
        """
        Get order status.
        
        Args:
            order_id (str): Order ID
            
        Returns:
            dict: Order status
        """
        order = self.order_manager.get_order(order_id)
        
        if not order:
            return {
                "status": "not_found",
                "order_id": order_id
            }
        
        return {
            "status": "success",
            "order_id": order_id,
            "order_status": order.status.value,
            "executed_quantity": order.executed_quantity,
            "average_price": order.average_price,
            "fills": order.fills
        }
    
    def get_execution_metrics(self, order_id=None, account_id=None, start_time=None, end_time=None):
        """
        Get execution performance metrics.
        
        Args:
            order_id (str, optional): Filter by order ID
            account_id (str, optional): Filter by account ID
            start_time (datetime, optional): Start time for filtering
            end_time (datetime, optional): End time for filtering
            
        Returns:
            dict: Execution metrics
        """
        # Get orders based on filters
        if order_id:
            orders = [self.order_manager.get_order(order_id)]
            orders = [o for o in orders if o is not None]
        elif account_id:
            orders = self.order_manager.get_orders_by_account(account_id)
        else:
            orders = list(self.order_manager.orders.values())
        
        # Apply time filters
        if start_time:
            orders = [o for o in orders if o.creation_time >= start_time]
        
        if end_time:
            orders = [o for o in orders if o.creation_time <= end_time]
        
        # Calculate metrics
        total_orders = len(orders)
        filled_orders = len([o for o in orders if o.status == OrderStatus.FILLED])
        partially_filled_orders = len([o for o in orders if o.status == OrderStatus.PARTIALLY_FILLED])
        cancelled_orders = len([o for o in orders if o.status == OrderStatus.CANCELLED])
        rejected_orders = len([o for o in orders if o.status == OrderStatus.REJECTED])
        
        total_executed_quantity = sum(o.executed_quantity for o in orders)
        total_value = sum(o.executed_quantity * o.average_price for o in orders if o.average_price)
        
        # Calculate fill rate
        fill_rate = filled_orders / total_orders if total_orders > 0 else 0
        
        # Calculate average execution time for filled orders
        execution_times = []
        
        for order in orders:
            if order.status == OrderStatus.FILLED and order.fills:
                start_time = order.creation_time
                end_time = max(fill["timestamp"] for fill in order.fills)
                execution_time = (end_time - start_time).total_seconds()
                execution_times.append(execution_time)
        
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        return {
            "total_orders": total_orders,
            "filled_orders": filled_orders,
            "partially_filled_orders": partially_filled_orders,
            "cancelled_orders": cancelled_orders,
            "rejected_orders": rejected_orders,
            "fill_rate": fill_rate,
            "total_executed_quantity": total_executed_quantity,
            "total_value": total_value,
            "average_execution_time": avg_execution_time
        }
    
    def _get_market_price(self, instrument):
        """
        Get current market price for an instrument.
        
        Args:
            instrument (str): Instrument identifier
            
        Returns:
            float: Market price or None if not available
        """
        # In a real system, this would fetch real-time market data
        # For simulation, return a random price around the last known price
        
        if instrument in self.market_data and "last_price" in self.market_data[instrument]:
            last_price = self.market_data[instrument]["last_price"]
            # Add some random noise
            return last_price * (1 + np.random.normal(0, 0.001))
        
        # Default price if no data available
        return 100.0
    
    def _get_market_volume(self, instrument):
        """
        Get current market volume for an instrument.
        
        Args:
            instrument (str): Instrument identifier
            
        Returns:
            float: Market volume or None if not available
        """
        # In a real system, this would fetch real-time market data
        # For simulation, return a random volume
        
        if instrument in self.market_data and "avg_volume" in self.market_data[instrument]:
            avg_volume = self.market_data[instrument]["avg_volume"]
            # Add some random noise
            return max(1, avg_volume * (1 + np.random.normal(0, 0.1)))
        
        # Default volume if no data available
        return 1000.0
    
    def _get_volume_profile(self, instrument, num_slices):
        """
        Get volume profile for an instrument.
        
        Args:
            instrument (str): Instrument identifier
            num_slices (int): Number of slices
            
        Returns:
            list: Volume profile
        """
        # In a real system, this would be based on historical data
        # For simulation, use a typical U-shaped volume profile
        
        # U-shaped profile (higher volume at open and close)
        if num_slices <= 2:
            return [1] * num_slices
        
        profile = []
        for i in range(num_slices):
            # Calculate position in [0, 1] range
            pos = i / (num_slices - 1)
            # U-shape function: higher at edges, lower in middle
            vol = 1 + 0.5 * (1 - 4 * (pos - 0.5)**2)
            profile.append(vol)
        
        return profile
    
    def update_market_data(self, instrument, data):
        """
        Update market data for an instrument.
        
        Args:
            instrument (str): Instrument identifier
            data (dict): Market data
        """
        with self._lock:
            self.market_data[instrument] = data
