#!/usr/bin/env python3
"""
JARVIS 3.0 - Position & Risk Management System (FIXED)
Advanced position management with explicit trading actions and comprehensive risk control.

FIXES APPLIED:
- Added bounds checking for all financial calculations
- Fixed division by zero errors in P&L calculations
- Enhanced risk assessment with proper validation
- Improved stop loss calculation with bounds
- Added market context validation
- Fixed position sizing validation
- Enhanced error handling throughout

Author: JARVIS 3.0 Team
Version: 3.1 (FIXED - Enhanced Risk Management)
"""

import sqlite3
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List, Any
import json
import os
from config import CONFIG

class Position:
    """Represents a trading position with enhanced validation."""
    
    def __init__(self, symbol: str, side: str, size: float, entry_price: float, 
                 timestamp: int, strategy_mode: str = None):
        """
        Initialize a position with validation.
        
        Args:
            symbol: Trading pair symbol
            side: 'LONG' or 'SHORT'
            size: Position size (percentage of portfolio)
            entry_price: Entry price (must be > 0)
            timestamp: Entry timestamp
            strategy_mode: Strategy used for this position
        """
        # Validation
        if entry_price <= 0:
            raise ValueError(f"Entry price must be positive, got {entry_price}")
        if size <= 0:
            raise ValueError(f"Position size must be positive, got {size}")
        if side not in ['LONG', 'SHORT']:
            raise ValueError(f"Side must be 'LONG' or 'SHORT', got {side}")
        
        self.symbol = symbol
        self.side = side
        self.size = size
        self.entry_price = entry_price
        self.timestamp = timestamp
        self.strategy_mode = strategy_mode or "unknown"
        
        # Risk management
        self.stop_loss = None
        self.take_profit = None
        self.trailing_stop = None
        
        # Tracking with validation
        self.current_price = entry_price
        self.unrealized_pnl = 0.0
        self.max_profit = 0.0
        self.max_drawdown = 0.0
        
        # Position ID for tracking
        self.position_id = f"{symbol}_{side}_{timestamp}"
        
    def update_price(self, current_price: float):
        """Update current price and calculate metrics with safety checks."""
        if current_price <= 0:
            raise ValueError(f"Current price must be positive, got {current_price}")
            
        self.current_price = current_price
        
        # Calculate unrealized P&L with safety checks
        if self.entry_price > 0:  # Additional safety check
            if self.side == 'LONG':
                self.unrealized_pnl = ((current_price - self.entry_price) / self.entry_price) * self.size
            else:  # SHORT
                self.unrealized_pnl = ((self.entry_price - current_price) / self.entry_price) * self.size
        else:
            self.unrealized_pnl = 0.0
        
        # Track maximum profit with bounds checking
        if self.unrealized_pnl > self.max_profit:
            self.max_profit = self.unrealized_pnl
        
        # Calculate drawdown safely
        if self.max_profit > 0:
            current_drawdown = self.max_profit - self.unrealized_pnl
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown
    
    def should_close(self) -> Tuple[bool, str]:
        """Check if position should be closed based on risk management rules."""
        if self.stop_loss is None and self.take_profit is None:
            return False, ""
            
        # Stop loss checks with validation
        if self.stop_loss is not None and self.stop_loss > 0:
            if self.side == 'LONG' and self.current_price <= self.stop_loss:
                return True, "STOP_LOSS"
            if self.side == 'SHORT' and self.current_price >= self.stop_loss:
                return True, "STOP_LOSS"
        
        # Take profit checks with validation
        if self.take_profit is not None and self.take_profit > 0:
            if self.side == 'LONG' and self.current_price >= self.take_profit:
                return True, "TAKE_PROFIT"
            if self.side == 'SHORT' and self.current_price <= self.take_profit:
                return True, "TAKE_PROFIT"
        
        return False, ""
    
    def to_dict(self) -> Dict:
        """Convert position to dictionary with validation."""
        return {
            'position_id': self.position_id,
            'symbol': self.symbol,
            'side': self.side,
            'size': max(0, self.size),  # Ensure positive
            'entry_price': max(0, self.entry_price),
            'current_price': max(0, self.current_price),
            'timestamp': self.timestamp,
            'strategy_mode': self.strategy_mode,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'trailing_stop': self.trailing_stop,
            'unrealized_pnl': self.unrealized_pnl,
            'max_profit': self.max_profit,
            'max_drawdown': max(0, self.max_drawdown)
        }


class PositionManager:
    """
    Enhanced position and risk management system for JARVIS 3.0 with comprehensive safety checks.
    """
    
    def __init__(self, db_path: str = None, initial_balance: float = 10000.0):
        """
        Initialize position manager with validation.
        
        Args:
            db_path: Path to database
            initial_balance: Starting portfolio balance (must be > 0)
        """
        if initial_balance <= 0:
            raise ValueError(f"Initial balance must be positive, got {initial_balance}")
            
        self.db_path = db_path or CONFIG.DATABASE_PATH
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        
        # Active positions
        self.active_positions: Dict[str, Position] = {}
        
        # Trading history
        self.trade_history: List[Dict] = []
        
        # Risk metrics with bounds
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.consecutive_losses = 0
        
        # Emergency stops
        self.emergency_stop = False
        self.daily_loss_limit_reached = False
        
        # Market regime tracking
        self.current_market_regime = "NORMAL"  # NORMAL, HIGH_VOLATILITY, LOW_LIQUIDITY
        self.last_health_check = int(time.time())
        
        # Setup logging
        self.logger = logging.getLogger('JARVIS_PositionManager')
        
        # Initialize database
        self._initialize_trading_db()
        
        # Load existing state
        self._load_existing_state()
        
        # Validate system state
        self._validate_system_state()
        
        self.logger.info("JARVIS 3.0 Position Manager initialized with enhanced safety")
    
    def _validate_system_state(self):
        """Validate system state on startup."""
        try:
            # Validate balance
            if self.current_balance < 0:
                self.logger.warning(f"Negative balance detected: {self.current_balance}, resetting to initial")
                self.current_balance = self.initial_balance
            
            # Validate positions
            invalid_positions = []
            for pos_id, position in self.active_positions.items():
                if position.entry_price <= 0 or position.size <= 0:
                    invalid_positions.append(pos_id)
            
            # Remove invalid positions
            for pos_id in invalid_positions:
                self.logger.warning(f"Removing invalid position: {pos_id}")
                del self.active_positions[pos_id]
                self._remove_position_from_db(pos_id)
            
            self.logger.info(f"System validation complete: {len(self.active_positions)} valid positions")
            
        except Exception as e:
            self.logger.error(f"System validation failed: {e}")
    
    def _initialize_trading_db(self):
        """Initialize trading database tables with enhanced structure."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Active positions table with validation constraints
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS active_positions (
                    position_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL CHECK (side IN ('LONG', 'SHORT')),
                    size REAL NOT NULL CHECK (size > 0),
                    entry_price REAL NOT NULL CHECK (entry_price > 0),
                    current_price REAL NOT NULL CHECK (current_price > 0),
                    timestamp INTEGER NOT NULL,
                    strategy_mode TEXT,
                    stop_loss REAL CHECK (stop_loss IS NULL OR stop_loss > 0),
                    take_profit REAL CHECK (take_profit IS NULL OR take_profit > 0),
                    trailing_stop REAL CHECK (trailing_stop IS NULL OR trailing_stop > 0),
                    unrealized_pnl REAL DEFAULT 0.0,
                    max_profit REAL DEFAULT 0.0,
                    max_drawdown REAL DEFAULT 0.0 CHECK (max_drawdown >= 0)
                )
            """)
            
            # Trade history table with enhanced validation
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trade_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    position_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL CHECK (side IN ('LONG', 'SHORT')),
                    size REAL NOT NULL CHECK (size > 0),
                    entry_price REAL NOT NULL CHECK (entry_price > 0),
                    exit_price REAL NOT NULL CHECK (exit_price > 0),
                    entry_timestamp INTEGER NOT NULL,
                    exit_timestamp INTEGER NOT NULL,
                    strategy_mode TEXT,
                    exit_reason TEXT,
                    pnl_percent REAL,
                    pnl_amount REAL,
                    duration_minutes INTEGER CHECK (duration_minutes >= 0)
                )
            """)
            
            # Portfolio state table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    balance REAL NOT NULL CHECK (balance >= 0),
                    total_pnl REAL,
                    daily_pnl REAL,
                    active_positions INTEGER NOT NULL CHECK (active_positions >= 0),
                    max_drawdown REAL DEFAULT 0.0 CHECK (max_drawdown >= 0),
                    consecutive_losses INTEGER DEFAULT 0 CHECK (consecutive_losses >= 0),
                    market_regime TEXT DEFAULT 'NORMAL'
                )
            """)
            
            # Risk monitoring table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS risk_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL CHECK (severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
                    description TEXT NOT NULL,
                    action_taken TEXT,
                    resolved INTEGER DEFAULT 0
                )
            """)
            
            conn.commit()
            conn.close()
            self.logger.info("Enhanced trading database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize trading database: {e}")
            raise
    
    def _load_existing_state(self):
        """Load existing positions and portfolio state with validation."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Load active positions with validation
            positions_df = pd.read_sql_query("SELECT * FROM active_positions", conn)
            loaded_positions = 0
            
            for _, row in positions_df.iterrows():
                try:
                    # Validate data before creating position
                    if row['entry_price'] <= 0 or row['size'] <= 0 or row['current_price'] <= 0:
                        self.logger.warning(f"Skipping invalid position: {row['position_id']}")
                        continue
                    
                    position = Position(
                        symbol=row['symbol'],
                        side=row['side'],
                        size=float(row['size']),
                        entry_price=float(row['entry_price']),
                        timestamp=int(row['timestamp']),
                        strategy_mode=row['strategy_mode']
                    )
                    position.current_price = float(row['current_price'])
                    position.stop_loss = float(row['stop_loss']) if row['stop_loss'] is not None else None
                    position.take_profit = float(row['take_profit']) if row['take_profit'] is not None else None
                    position.trailing_stop = float(row['trailing_stop']) if row['trailing_stop'] is not None else None
                    position.unrealized_pnl = float(row['unrealized_pnl'] or 0.0)
                    position.max_profit = float(row['max_profit'] or 0.0)
                    position.max_drawdown = float(row['max_drawdown'] or 0.0)
                    
                    self.active_positions[position.position_id] = position
                    loaded_positions += 1
                    
                except Exception as e:
                    self.logger.warning(f"Failed to load position {row.get('position_id', 'unknown')}: {e}")
            
            # Load latest portfolio state with validation
            portfolio_df = pd.read_sql_query(
                "SELECT * FROM portfolio_state ORDER BY timestamp DESC LIMIT 1", conn
            )
            if not portfolio_df.empty:
                row = portfolio_df.iloc[0]
                # Validate portfolio data
                balance = float(row['balance'])
                if balance > 0:
                    self.current_balance = balance
                    self.total_pnl = float(row['total_pnl'] or 0.0)
                    self.daily_pnl = float(row['daily_pnl'] or 0.0)
                    self.max_drawdown = max(0.0, float(row['max_drawdown'] or 0.0))
                    self.consecutive_losses = max(0, int(row['consecutive_losses'] or 0))
                    self.current_market_regime = row.get('market_regime', 'NORMAL')
            
            conn.close()
            self.logger.info(f"Loaded {loaded_positions} valid active positions")
            
        except Exception as e:
            self.logger.warning(f"Could not load existing state: {e}")
    
    def validate_market_context(self, market_context: Dict) -> Dict:
        """Validate and sanitize market context data."""
        validated_context = {
            'volatility': 'MEDIUM',
            'fear': 5.0,
            'confidence': 0.5,
            'position_count': 0,
            'ta_summary': [0.0] * 4,
            'historical_performance': {'win_rate': 0.5, 'avg_profit': 0.0, 'sample_size': 0},
            'market_condition_effectiveness': {'win_rate': 0.5, 'avg_profit': 0.0, 'sample_size': 0}
        }
        
        if not isinstance(market_context, dict):
            self.logger.warning("Invalid market context, using defaults")
            return validated_context
        
        # Validate volatility
        volatility = market_context.get('volatility', 'MEDIUM')
        if volatility in ['LOW', 'MEDIUM', 'HIGH']:
            validated_context['volatility'] = volatility
        
        # Validate fear (0-10 scale)
        try:
            fear = float(market_context.get('fear', 5.0))
            validated_context['fear'] = max(0.0, min(10.0, fear))
        except (ValueError, TypeError):
            pass
        
        # Validate confidence (0-1 scale)
        try:
            confidence = float(market_context.get('confidence', 0.5))
            validated_context['confidence'] = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            pass
        
        # Validate position count
        try:
            position_count = int(market_context.get('position_count', 0))
            validated_context['position_count'] = max(0, position_count)
        except (ValueError, TypeError):
            pass
        
        # Validate TA summary
        ta_summary = market_context.get('ta_summary', [])
        if isinstance(ta_summary, (list, tuple)) and len(ta_summary) >= 4:
            try:
                validated_ta = [float(x) for x in ta_summary[:4]]
                # Bound TA values to reasonable range
                validated_context['ta_summary'] = [max(-10.0, min(10.0, x)) for x in validated_ta]
            except (ValueError, TypeError):
                pass
        
        return validated_context
    
    def can_open_position(self, symbol: str, size: float, current_price: float = 0.0, 
                         market_context: Dict = None) -> Tuple[bool, str]:
        """
        Enhanced position validation with market context and volatility considerations.
        
        Args:
            symbol: Trading pair symbol
            size: Requested position size
            current_price: Current market price for volatility assessment
            market_context: Market conditions for dynamic risk assessment
        """
        # Basic validation
        if size <= 0:
            return False, f"Position size must be positive, got {size}"
        
        # Check emergency stop
        if self.emergency_stop:
            return False, "Emergency stop activated"
        
        # Check daily loss limit
        if self.daily_pnl <= -CONFIG.RiskLimits.DAILY_DRAWDOWN_LIMIT:
            return False, f"Daily drawdown limit reached: {self.daily_pnl:.1%}"
        
        # Check maximum positions
        if len(self.active_positions) >= CONFIG.RiskLimits.MAX_CONCURRENT_POSITIONS:
            return False, f"Maximum concurrent positions limit reached ({CONFIG.RiskLimits.MAX_CONCURRENT_POSITIONS})"
        
        # Enhanced position size validation with market context
        max_position_size = CONFIG.RiskLimits.MAX_SYMBOL_EXPOSURE
        
        # Adjust for market volatility if available
        if market_context:
            validated_context = self.validate_market_context(market_context)
            volatility = validated_context['volatility']
            fear = validated_context['fear']
            
            # Reduce max position size in high volatility or high fear
            if volatility == 'HIGH':
                max_position_size *= 0.7  # 30% reduction
            elif volatility == 'LOW':
                max_position_size *= 1.1  # 10% increase (but still capped)
            
            if fear > 7.0:
                max_position_size *= 0.8  # 20% reduction for high fear
            
            # Cap the adjusted size
            max_position_size = min(max_position_size, CONFIG.RiskLimits.MAX_SYMBOL_EXPOSURE)
        
        if size > max_position_size:
            return False, f"Position size {size:.1%} exceeds adjusted maximum {max_position_size:.1%}"
        
        # Check symbol exposure with correlation consideration
        symbol_exposure = sum(pos.size for pos in self.active_positions.values() if pos.symbol == symbol)
        if symbol_exposure + size > CONFIG.RiskLimits.MAX_SYMBOL_EXPOSURE:
            return False, f"Total {symbol} exposure would be {symbol_exposure + size:.1%}, exceeds limit {CONFIG.RiskLimits.MAX_SYMBOL_EXPOSURE:.1%}"
        
        # Check consecutive losses
        if self.consecutive_losses >= CONFIG.RiskLimits.MAX_CONSECUTIVE_LOSSES:
            return False, f"Maximum consecutive losses reached ({self.consecutive_losses})"
        
        # Check available balance
        required_margin = size * self.current_balance
        if required_margin > self.current_balance * 0.9:  # Keep 10% buffer
            return False, f"Insufficient balance: need {required_margin:.2f}, have {self.current_balance:.2f}"
        
        return True, "Position can be opened"
    
    def calculate_stop_loss(self, action: str, current_price: float, market_context: Dict) -> float:
        """
        Calculate appropriate stop loss level with enhanced bounds checking.
        
        Args:
            action: Trading action ('BUY', 'SELL', 'HOLD')
            current_price: Current market price (must be > 0)
            market_context: Market condition data
            
        Returns:
            Stop loss price level (0.0 if HOLD or invalid)
        """
        if action == 'HOLD' or current_price <= 0:
            return 0.0
        
        # Validate market context
        validated_context = self.validate_market_context(market_context)
        
        # Base stop loss distance with bounds
        base_distance = 0.015  # 1.5%
        min_distance = 0.003   # 0.3% minimum
        max_distance = 0.05    # 5.0% maximum
        
        # Adjust for volatility
        volatility = validated_context['volatility']
        if volatility == 'HIGH':
            distance_multiplier = 1.8  # Wider stops for high volatility
        elif volatility == 'LOW':
            distance_multiplier = 0.6  # Tighter stops for low volatility
        else:
            distance_multiplier = 1.0
        
        # Adjust for confidence
        confidence = validated_context['confidence']
        if confidence > 0.8:
            distance_multiplier *= 0.85  # Tighter stops with high confidence
        elif confidence < 0.3:
            distance_multiplier *= 1.3   # Wider stops with low confidence
        
        # Calculate stop distance with bounds
        stop_distance = base_distance * distance_multiplier
        stop_distance = max(min_distance, min(max_distance, stop_distance))
        
        # Calculate stop price
        if action == 'BUY':
            stop_price = current_price * (1 - stop_distance)
        else:  # SELL
            stop_price = current_price * (1 + stop_distance)
        
        # Validate stop price
        if stop_price <= 0:
            self.logger.warning(f"Invalid stop price calculated: {stop_price}, using fallback")
            return current_price * 0.95 if action == 'BUY' else current_price * 1.05
        
        return stop_price
    
    def calculate_take_profit(self, action: str, current_price: float, market_context: Dict) -> float:
        """
        Calculate appropriate take profit level with risk/reward optimization.
        
        Args:
            action: Trading action ('BUY', 'SELL', 'HOLD')
            current_price: Current market price (must be > 0)
            market_context: Market condition data
            
        Returns:
            Take profit price level (0.0 if HOLD or invalid)
        """
        if action == 'HOLD' or current_price <= 0:
            return 0.0
        
        # Calculate stop loss distance for risk/reward ratio
        stop_price = self.calculate_stop_loss(action, current_price, market_context)
        if stop_price <= 0:
            return 0.0
        
        # Calculate stop distance
        if action == 'BUY':
            stop_distance = abs(current_price - stop_price) / current_price
        else:  # SELL
            stop_distance = abs(stop_price - current_price) / current_price
        
        # Target risk/reward ratio (configurable)
        risk_reward_ratio = 2.0  # 2:1 reward to risk
        
        # Adjust ratio based on market conditions
        validated_context = self.validate_market_context(market_context)
        
        # Increase target in trending markets
        ta_summary = validated_context['ta_summary']
        if len(ta_summary) >= 4:
            momentum = abs(ta_summary[3]) if abs(ta_summary[3]) < 10 else 0  # Bound momentum
            if momentum > 0.3:  # Strong momentum
                risk_reward_ratio *= 1.5
        
        # Calculate take profit distance
        tp_distance = stop_distance * risk_reward_ratio
        tp_distance = min(tp_distance, 0.15)  # Cap at 15% maximum
        
        # Calculate take profit price
        if action == 'BUY':
            tp_price = current_price * (1 + tp_distance)
        else:  # SELL
            tp_price = current_price * (1 - tp_distance)
        
        # Validate take profit price
        if tp_price <= 0:
            self.logger.warning(f"Invalid take profit price calculated: {tp_price}")
            return 0.0
        
        return tp_price
    
    def assess_risk(self, action: str, market_context: Dict, current_price: float = 0.0) -> Dict:
        """
        Enhanced comprehensive risk assessment with proper bounds and validation.
        
        Args:
            action: Trading action
            market_context: Market condition data (will be validated)
            current_price: Current market price for additional context
            
        Returns:
            Risk assessment dictionary with validation
        """
        # Validate inputs
        validated_context = self.validate_market_context(market_context)
        
        # Extract validated risk factors
        volatility = validated_context['volatility']
        fear = validated_context['fear']  # 0-10 scale
        confidence = validated_context['confidence']  # 0-1 scale
        position_count = validated_context['position_count']
        
        # Calculate base risk score (1-10 scale) with proper bounds
        risk_score = 5.0  # Neutral baseline
        risk_factors = []
        
        # Volatility impact (bounded)
        if volatility == 'HIGH':
            risk_score += 2.0
            risk_factors.append("High market volatility")
        elif volatility == 'LOW':
            risk_score -= 1.0
            risk_factors.append("Low volatility environment")
        else:
            risk_factors.append("Normal volatility")
        
        # Fear adjustment (0-10 scale to risk adjustment)
        fear_adjustment = (fear - 5.0) * 0.4  # Scale to ±2.0
        risk_score += fear_adjustment
        if fear > 7.0:
            risk_factors.append("Elevated fear levels")
        elif fear < 3.0:
            risk_factors.append("Low fear supports risk-taking")
        
        # Confidence adjustment (0-1 scale to risk adjustment)
        confidence_adjustment = (0.5 - confidence) * 3.0  # Scale to ±1.5
        risk_score += confidence_adjustment
        if confidence < 0.3:
            risk_factors.append("Low confidence in signals")
        elif confidence > 0.8:
            risk_factors.append("High confidence in analysis")
        
        # Position concentration risk
        if position_count > 3:
            concentration_risk = (position_count - 3) * 0.5
            risk_score += min(concentration_risk, 2.0)  # Cap at +2.0
            risk_factors.append(f"High position concentration ({position_count} positions)")
        
        # Action-specific risk
        if action in ['BUY', 'SELL']:
            risk_score += 0.5  # Active trading adds risk
            risk_factors.append("Active trading position")
        else:
            risk_factors.append("Holding position (lower risk)")
        
        # Current market regime impact
        if self.current_market_regime == "HIGH_VOLATILITY":
            risk_score += 1.0
            risk_factors.append("High volatility market regime")
        elif self.current_market_regime == "LOW_LIQUIDITY":
            risk_score += 1.5
            risk_factors.append("Low liquidity conditions")
        
        # Consecutive losses impact
        if self.consecutive_losses > 2:
            loss_penalty = min(self.consecutive_losses * 0.3, 1.5)
            risk_score += loss_penalty
            risk_factors.append(f"Recent consecutive losses ({self.consecutive_losses})")
        
        # Bound final risk score strictly
        risk_score = max(1.0, min(10.0, risk_score))
        
        # Risk categorization with proper thresholds
        if risk_score <= 3.0:
            risk_level = 'LOW'
            risk_description = 'Favorable conditions, manageable risk'
            max_position_multiplier = 1.2
        elif risk_score <= 5.5:
            risk_level = 'MEDIUM'
            risk_description = 'Moderate risk, standard position sizing'
            max_position_multiplier = 1.0
        elif risk_score <= 7.5:
            risk_level = 'HIGH'
            risk_description = 'Elevated risk, reduce position size'
            max_position_multiplier = 0.6
        else:
            risk_level = 'EXTREME'
            risk_description = 'High risk environment, avoid new positions'
            max_position_multiplier = 0.3
        
        # Calculate dynamic limits
        base_position_size = CONFIG.RiskLimits.MAX_SYMBOL_EXPOSURE
        adjusted_max_position = base_position_size * max_position_multiplier
        
        # Maximum acceptable loss based on risk score
        base_max_loss = 0.02  # 2% base
        risk_adjusted_loss = base_max_loss * (11 - risk_score) / 10  # Higher risk = lower max loss
        max_acceptable_loss = max(0.005, min(0.05, risk_adjusted_loss))  # Bound between 0.5% and 5%
        
        return {
            'risk_score': round(risk_score, 2),
            'risk_level': risk_level,
            'risk_description': risk_description,
            'max_position_size': round(adjusted_max_position, 3),
            'max_acceptable_loss': round(max_acceptable_loss, 4),
            'risk_factors': risk_factors,
            'risk_breakdown': {
                'base_score': 5.0,
                'volatility_adjustment': 2.0 if volatility == 'HIGH' else (-1.0 if volatility == 'LOW' else 0.0),
                'fear_adjustment': round(fear_adjustment, 2),
                'confidence_adjustment': round(confidence_adjustment, 2),
                'concentration_risk': min((position_count - 3) * 0.5, 2.0) if position_count > 3 else 0.0,
                'action_risk': 0.5 if action in ['BUY', 'SELL'] else 0.0
            },
            'recommendations': self._generate_risk_recommendations(risk_level, risk_factors)
        }
    
    def _generate_risk_recommendations(self, risk_level: str, risk_factors: List[str]) -> List[str]:
        """Generate specific risk management recommendations."""
        recommendations = []
        
        if risk_level == 'EXTREME':
            recommendations.extend([
                "Consider avoiding new positions",
                "Review existing positions for early exit",
                "Wait for better market conditions"
            ])
        elif risk_level == 'HIGH':
            recommendations.extend([
                "Reduce position sizes significantly",
                "Use tighter stop losses",
                "Consider partial profit taking"
            ])
        elif risk_level == 'MEDIUM':
            recommendations.extend([
                "Use standard risk management",
                "Monitor positions closely",
                "Maintain disciplined approach"
            ])
        else:  # LOW risk
            recommendations.extend([
                "Favorable conditions for trading",
                "Standard or slightly larger positions acceptable",
                "Maintain proper risk controls"
            ])
        
        # Add specific recommendations based on risk factors
        if "High market volatility" in risk_factors:
            recommendations.append("Use wider stop losses to account for volatility")
        if "Low confidence in signals" in risk_factors:
            recommendations.append("Wait for higher conviction setups")
        if "High position concentration" in risk_factors:
            recommendations.append("Consider closing some positions to reduce concentration")
        
        return recommendations
    
    def log_risk_event(self, event_type: str, severity: str, description: str, action_taken: str = None):
        """Log risk events for monitoring and analysis."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO risk_events 
                (timestamp, event_type, severity, description, action_taken)
                VALUES (?, ?, ?, ?, ?)
            """, (
                int(time.time() * 1000),
                event_type,
                severity,
                description,
                action_taken or "None"
            ))
            
            conn.commit()
            conn.close()
            
            if severity in ['HIGH', 'CRITICAL']:
                self.logger.warning(f"Risk Event [{severity}]: {description}")
            
        except Exception as e:
            self.logger.error(f"Failed to log risk event: {e}")
    
    def open_position(self, symbol: str, side: str, size: float, current_price: float,
                     strategy_params: Dict, strategy_mode: str = None, 
                     market_context: Dict = None) -> Optional[str]:
        """
        Open a new trading position with enhanced validation and risk management.
        
        Args:
            symbol: Trading pair symbol
            side: 'LONG' or 'SHORT'
            size: Position size (percentage of portfolio)
            current_price: Current market price
            strategy_params: Strategy parameters including stops
            strategy_mode: Strategy mode used
            market_context: Market conditions for risk assessment
            
        Returns:
            Position ID if successful, None otherwise
        """
        # Enhanced validation
        if current_price <= 0:
            self.logger.error(f"Invalid current price: {current_price}")
            return None
        
        if size <= 0:
            self.logger.error(f"Invalid position size: {size}")
            return None
        
        if side not in ['LONG', 'SHORT']:
            self.logger.error(f"Invalid side: {side}")
            return None
        
        # Check if position can be opened with market context
        can_open, reason = self.can_open_position(symbol, size, current_price, market_context)
        if not can_open:
            self.logger.warning(f"Cannot open position: {reason}")
            self.log_risk_event("POSITION_REJECTED", "MEDIUM", reason)
            return None
        
        # Risk assessment
        if market_context:
            risk_assessment = self.assess_risk(side, market_context, current_price)
            
            # Check if risk is too high
            if risk_assessment['risk_level'] == 'EXTREME':
                self.logger.warning(f"Extreme risk detected, rejecting position: {risk_assessment['risk_description']}")
                self.log_risk_event("HIGH_RISK_REJECTION", "HIGH", 
                                  f"Position rejected due to extreme risk: {risk_assessment['risk_description']}")
                return None
            
            # Adjust position size based on risk
            if risk_assessment['risk_level'] == 'HIGH':
                original_size = size
                size = min(size, risk_assessment['max_position_size'])
                if size != original_size:
                    self.logger.info(f"Position size adjusted for risk: {original_size:.3f} -> {size:.3f}")
        
        try:
            # Create position with validation
            timestamp = int(time.time() * 1000)
            position = Position(symbol, side, size, current_price, timestamp, strategy_mode)
            
            # Set risk management parameters with validation
            if 'stop_loss' in strategy_params and strategy_params['stop_loss'] > 0:
                if side == 'LONG':
                    stop_price = current_price * (1 - strategy_params['stop_loss'])
                else:  # SHORT
                    stop_price = current_price * (1 + strategy_params['stop_loss'])
                
                # Validate stop price
                if stop_price > 0:
                    position.stop_loss = stop_price
            
            if 'profit_target' in strategy_params and strategy_params['profit_target'] > 0:
                if side == 'LONG':
                    tp_price = current_price * (1 + strategy_params['profit_target'])
                else:  # SHORT
                    tp_price = current_price * (1 - strategy_params['profit_target'])
                
                # Validate take profit price
                if tp_price > 0:
                    position.take_profit = tp_price
            
            if 'trailing_stop' in strategy_params and strategy_params['trailing_stop'] > 0:
                position.trailing_stop = strategy_params['trailing_stop']
            
            # Add to active positions
            self.active_positions[position.position_id] = position
            
            # Save to database
            self._save_position(position)
            
            # Log successful position opening
            self.log_risk_event("POSITION_OPENED", "LOW", 
                              f"Opened {side} position for {symbol}: size={size:.3f}, price={current_price:.4f}")
            
            self.logger.info(f"Opened {side} position for {symbol}: size={size:.3f}, price={current_price:.4f}, ID={position.position_id}")
            
            return position.position_id
            
        except Exception as e:
            self.logger.error(f"Failed to open position: {e}")
            self.log_risk_event("POSITION_ERROR", "HIGH", f"Failed to open position: {str(e)}")
            return None
    
    def close_position(self, position_id: str, current_price: float, 
                      exit_reason: str = "MANUAL") -> bool:
        """
        Close an active position with enhanced validation and tracking.
        
        Args:
            position_id: Position identifier
            current_price: Current market price
            exit_reason: Reason for closing position
            
        Returns:
            True if successful, False otherwise
        """
        if position_id not in self.active_positions:
            self.logger.warning(f"Position {position_id} not found")
            return False
        
        if current_price <= 0:
            self.logger.error(f"Invalid exit price: {current_price}")
            return False
        
        try:
            position = self.active_positions[position_id]
            
            # Update final price with validation
            position.update_price(current_price)
            
            # Calculate final P&L with safety checks
            pnl_percent = position.unrealized_pnl
            pnl_amount = pnl_percent * self.current_balance
            
            # Validate P&L calculation
            if abs(pnl_percent) > 1.0:  # More than 100% P&L seems wrong
                self.logger.warning(f"Unusual P&L calculated: {pnl_percent:.1%}, validating...")
                # Recalculate manually
                if position.entry_price > 0:
                    if position.side == 'LONG':
                        manual_pnl = ((current_price - position.entry_price) / position.entry_price) * position.size
                    else:
                        manual_pnl = ((position.entry_price - current_price) / position.entry_price) * position.size
                    
                    if abs(manual_pnl - pnl_percent) > 0.01:  # More than 1% difference
                        self.logger.warning(f"P&L mismatch detected, using manual calculation: {manual_pnl:.1%}")
                        pnl_percent = manual_pnl
                        pnl_amount = pnl_percent * self.current_balance
            
            # Calculate duration safely
            current_time = int(time.time() * 1000)
            duration_minutes = max(0, (current_time - position.timestamp) / (1000 * 60))
            
            # Update portfolio metrics with validation
            self.total_pnl += pnl_percent
            self.daily_pnl += pnl_percent
            self.current_balance = max(0, self.current_balance + pnl_amount)  # Ensure non-negative
            
            # Track consecutive losses
            if pnl_percent < -0.001:  # Consider < -0.1% as loss
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
            
            # Update max drawdown
            if self.initial_balance > 0:
                current_drawdown = max(0, (self.initial_balance - self.current_balance) / self.initial_balance)
                if current_drawdown > self.max_drawdown:
                    self.max_drawdown = current_drawdown
            
            # Create trade record with validation
            trade_record = {
                'position_id': position.position_id,
                'symbol': position.symbol,
                'side': position.side,
                'size': position.size,
                'entry_price': position.entry_price,
                'exit_price': current_price,
                'entry_timestamp': position.timestamp,
                'exit_timestamp': current_time,
                'strategy_mode': position.strategy_mode or 'unknown',
                'exit_reason': exit_reason,
                'pnl_percent': pnl_percent,
                'pnl_amount': pnl_amount,
                'duration_minutes': duration_minutes
            }
            
            # Validate trade record
            for key, value in trade_record.items():
                if key in ['entry_price', 'exit_price', 'size'] and (value is None or value <= 0):
                    self.logger.error(f"Invalid trade record: {key} = {value}")
                    return False
            
            self.trade_history.append(trade_record)
            self._save_trade_history(trade_record)
            
            # Remove from active positions
            del self.active_positions[position_id]
            self._remove_position_from_db(position_id)
            
            # Log trade closure
            result_type = "PROFIT" if pnl_percent > 0 else "LOSS"
            self.log_risk_event("POSITION_CLOSED", "LOW", 
                              f"Closed position {position_id}: {result_type} of {pnl_percent:.2%}")
            
            # Check emergency conditions
            self._check_emergency_conditions()
            
            self.logger.info(f"Closed position {position_id}: P&L={pnl_percent:.3%} ({result_type}), reason={exit_reason}, duration={duration_minutes:.1f}min")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to close position {position_id}: {e}")
            self.log_risk_event("POSITION_CLOSE_ERROR", "HIGH", f"Failed to close position: {str(e)}")
            return False
    
    def update_positions(self, market_data: Dict[str, float]):
        """
        Update all active positions with current market prices and enhanced monitoring.
        
        Args:
            market_data: Dictionary of symbol -> current price
        """
        if not isinstance(market_data, dict):
            self.logger.warning("Invalid market data format")
            return
        
        positions_to_close = []
        
        for position_id, position in self.active_positions.items():
            if position.symbol in market_data:
                current_price = market_data[position.symbol]
                
                # Validate price
                if current_price <= 0:
                    self.logger.warning(f"Invalid price for {position.symbol}: {current_price}")
                    continue
                
                try:
                    # Update position with new price
                    old_pnl = position.unrealized_pnl
                    position.update_price(current_price)
                    
                    # Check for unusual price movements
                    if abs(position.unrealized_pnl - old_pnl) > 0.05:  # More than 5% change
                        self.logger.info(f"Large P&L change for {position_id}: {old_pnl:.2%} -> {position.unrealized_pnl:.2%}")
                    
                    # Check if position should be closed
                    should_close, reason = position.should_close()
                    if should_close:
                        positions_to_close.append((position_id, current_price, reason))
                    
                    # Update trailing stop
                    self._update_trailing_stop(position)
                    
                    # Save updated position
                    self._save_position(position)
                    
                except Exception as e:
                    self.logger.error(f"Error updating position {position_id}: {e}")
        
        # Close positions that hit stops
        for position_id, price, reason in positions_to_close:
            self.close_position(position_id, price, reason)
    
    def _update_trailing_stop(self, position: Position):
        """Update trailing stop for a position with enhanced validation."""
        if position.trailing_stop is None or position.trailing_stop <= 0:
            return
        
        try:
            if position.side == 'LONG' and position.unrealized_pnl > 0:
                # Move stop loss up with the price
                new_stop = position.current_price * (1 - position.trailing_stop)
                if position.stop_loss is None or new_stop > position.stop_loss:
                    old_stop = position.stop_loss
                    position.stop_loss = new_stop
                    self.logger.debug(f"Updated trailing stop for {position.position_id}: {old_stop} -> {new_stop}")
            
            elif position.side == 'SHORT' and position.unrealized_pnl > 0:
                # Move stop loss down with the price
                new_stop = position.current_price * (1 + position.trailing_stop)
                if position.stop_loss is None or new_stop < position.stop_loss:
                    old_stop = position.stop_loss
                    position.stop_loss = new_stop
                    self.logger.debug(f"Updated trailing stop for {position.position_id}: {old_stop} -> {new_stop}")
                    
        except Exception as e:
            self.logger.error(f"Error updating trailing stop for {position.position_id}: {e}")
    
    def _check_emergency_conditions(self):
        """Check for emergency stop conditions with enhanced monitoring."""
        try:
            # Portfolio loss check
            if self.initial_balance > 0:
                portfolio_loss = (self.initial_balance - self.current_balance) / self.initial_balance
                
                if portfolio_loss >= CONFIG.RiskLimits.EMERGENCY_STOP_LOSS:
                    if not self.emergency_stop:
                        self.emergency_stop = True
                        self.log_risk_event("EMERGENCY_STOP", "CRITICAL", 
                                          f"Portfolio loss {portfolio_loss:.1%} exceeds emergency limit")
                        self.logger.critical(f"EMERGENCY STOP ACTIVATED: Portfolio loss {portfolio_loss:.1%}")
            
            # Daily loss limit check
            if self.daily_pnl <= -CONFIG.RiskLimits.DAILY_DRAWDOWN_LIMIT:
                if not self.daily_loss_limit_reached:
                    self.daily_loss_limit_reached = True
                    self.log_risk_event("DAILY_LIMIT", "HIGH", f"Daily loss limit reached: {self.daily_pnl:.1%}")
                    self.logger.warning(f"Daily loss limit reached: {self.daily_pnl:.1%}")
            
            # Check for rapid consecutive losses
            if self.consecutive_losses >= CONFIG.RiskLimits.MAX_CONSECUTIVE_LOSSES:
                self.log_risk_event("CONSECUTIVE_LOSSES", "HIGH", 
                                  f"Maximum consecutive losses reached: {self.consecutive_losses}")
                
        except Exception as e:
            self.logger.error(f"Error checking emergency conditions: {e}")
    
    def perform_health_check(self) -> Dict:
        """Perform comprehensive system health check."""
        health_status = {
            'timestamp': int(time.time()),
            'overall_status': 'HEALTHY',
            'issues': [],
            'warnings': [],
            'metrics': {}
        }
        
        try:
            # Check balance consistency
            if self.current_balance < 0:
                health_status['issues'].append("Negative balance detected")
                health_status['overall_status'] = 'UNHEALTHY'
            
            # Check position validity
            invalid_positions = 0
            total_exposure = 0
            
            for pos_id, position in self.active_positions.items():
                if position.entry_price <= 0 or position.size <= 0:
                    invalid_positions += 1
                total_exposure += position.size
            
            if invalid_positions > 0:
                health_status['issues'].append(f"{invalid_positions} invalid positions detected")
                health_status['overall_status'] = 'UNHEALTHY'
            
            # Check exposure limits
            if total_exposure > 1.0:  # More than 100% exposure
                health_status['warnings'].append(f"High total exposure: {total_exposure:.1%}")
                if health_status['overall_status'] == 'HEALTHY':
                    health_status['overall_status'] = 'WARNING'
            
            # Check emergency conditions
            if self.emergency_stop:
                health_status['issues'].append("Emergency stop is active")
                health_status['overall_status'] = 'EMERGENCY'
            
            if self.daily_loss_limit_reached:
                health_status['warnings'].append("Daily loss limit reached")
                if health_status['overall_status'] == 'HEALTHY':
                    health_status['overall_status'] = 'WARNING'
            
            # Calculate metrics
            health_status['metrics'] = {
                'current_balance': self.current_balance,
                'daily_pnl': self.daily_pnl,
                'total_pnl': self.total_pnl,
                'max_drawdown': self.max_drawdown,
                'active_positions': len(self.active_positions),
                'consecutive_losses': self.consecutive_losses,
                'total_exposure': total_exposure,
                'emergency_stop': self.emergency_stop
            }
            
            self.last_health_check = int(time.time())
            
        except Exception as e:
            health_status['issues'].append(f"Health check error: {str(e)}")
            health_status['overall_status'] = 'ERROR'
        
        return health_status
    
    # Database operations with enhanced error handling
    def _save_position(self, position: Position):
        """Save position to database with validation."""
        try:
            # Validate position data before saving
            pos_dict = position.to_dict()
            for key in ['entry_price', 'current_price', 'size']:
                if pos_dict[key] <= 0:
                    raise ValueError(f"Invalid {key}: {pos_dict[key]}")
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO active_positions 
                (position_id, symbol, side, size, entry_price, current_price, timestamp,
                 strategy_mode, stop_loss, take_profit, trailing_stop, unrealized_pnl,
                 max_profit, max_drawdown)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                position.position_id, position.symbol, position.side, position.size,
                position.entry_price, position.current_price, position.timestamp,
                position.strategy_mode, position.stop_loss, position.take_profit,
                position.trailing_stop, position.unrealized_pnl, position.max_profit,
                position.max_drawdown
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to save position {position.position_id}: {e}")
    
    def _save_trade_history(self, trade_record: Dict):
        """Save completed trade to history with validation."""
        try:
            # Validate trade record
            required_fields = ['position_id', 'symbol', 'side', 'entry_price', 'exit_price']
            for field in required_fields:
                if field not in trade_record or trade_record[field] is None:
                    raise ValueError(f"Missing required field: {field}")
                
                if field in ['entry_price', 'exit_price'] and trade_record[field] <= 0:
                    raise ValueError(f"Invalid {field}: {trade_record[field]}")
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO trade_history 
                (position_id, symbol, side, size, entry_price, exit_price,
                 entry_timestamp, exit_timestamp, strategy_mode, exit_reason,
                 pnl_percent, pnl_amount, duration_minutes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_record['position_id'], trade_record['symbol'], trade_record['side'],
                trade_record['size'], trade_record['entry_price'], trade_record['exit_price'],
                trade_record['entry_timestamp'], trade_record['exit_timestamp'],
                trade_record['strategy_mode'], trade_record['exit_reason'],
                trade_record['pnl_percent'], trade_record['pnl_amount'],
                trade_record['duration_minutes']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to save trade history: {e}")
    
    def _remove_position_from_db(self, position_id: str):
        """Remove position from database with validation."""
        try:
            if not position_id:
                raise ValueError("Position ID cannot be empty")
                
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM active_positions WHERE position_id = ?", (position_id,))
            
            if cursor.rowcount == 0:
                self.logger.warning(f"Position {position_id} not found in database")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to remove position from database: {e}")
    
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio summary with validation."""
        try:
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.active_positions.values())
            
            # Calculate performance metrics safely
            total_trades = len(self.trade_history)
            winning_trades = len([trade for trade in self.trade_history if trade.get('pnl_percent', 0) > 0])
            losing_trades = total_trades - winning_trades
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Calculate average win/loss safely
            winning_pnls = [trade['pnl_percent'] for trade in self.trade_history 
                           if trade.get('pnl_percent', 0) > 0]
            losing_pnls = [trade['pnl_percent'] for trade in self.trade_history 
                          if trade.get('pnl_percent', 0) < 0]
            
            avg_win = np.mean(winning_pnls) if winning_pnls else 0
            avg_loss = np.mean(losing_pnls) if losing_pnls else 0
            
            # Calculate profit factor safely
            total_wins = sum(winning_pnls) if winning_pnls else 0
            total_losses = abs(sum(losing_pnls)) if losing_pnls else 0
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            return {
                'current_balance': round(self.current_balance, 2),
                'initial_balance': round(self.initial_balance, 2),
                'total_pnl': round(self.total_pnl, 4),
                'total_pnl_percent': round(self.total_pnl * 100, 2),
                'daily_pnl': round(self.daily_pnl, 4),
                'daily_pnl_percent': round(self.daily_pnl * 100, 2),
                'unrealized_pnl': round(total_unrealized_pnl, 4),
                'unrealized_pnl_percent': round(total_unrealized_pnl * 100, 2),
                'max_drawdown': round(self.max_drawdown, 4),
                'max_drawdown_percent': round(self.max_drawdown * 100, 2),
                'active_positions': len(self.active_positions),
                'consecutive_losses': self.consecutive_losses,
                'emergency_stop': self.emergency_stop,
                'daily_loss_limit_reached': self.daily_loss_limit_reached,
                'current_market_regime': self.current_market_regime,
                'performance_metrics': {
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'win_rate': round(win_rate, 3),
                    'avg_win': round(avg_win, 4),
                    'avg_loss': round(avg_loss, 4),
                    'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 'N/A'
                },
                'risk_metrics': {
                    'total_exposure': sum(pos.size for pos in self.active_positions.values()),
                    'largest_position': max([pos.size for pos in self.active_positions.values()], default=0),
                    'portfolio_health': self.perform_health_check()['overall_status']
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating portfolio summary: {e}")
            return {
                'error': str(e),
                'current_balance': self.current_balance,
                'emergency_stop': self.emergency_stop
            }
    
    def get_active_positions_summary(self) -> List[Dict]:
        """Get summary of all active positions with validation."""
        try:
            return [pos.to_dict() for pos in self.active_positions.values()]
        except Exception as e:
            self.logger.error(f"Error getting active positions summary: {e}")
            return []
    
    def reset_daily_metrics(self):
        """Reset daily metrics (call at start of new trading day)."""
        try:
            self.daily_pnl = 0.0
            self.daily_loss_limit_reached = False
            self.consecutive_losses = 0  # Reset daily
            
            # Log daily reset
            self.log_risk_event("DAILY_RESET", "LOW", "Daily metrics reset for new trading day")
            self.logger.info("Daily metrics reset for new trading day")
            
        except Exception as e:
            self.logger.error(f"Error resetting daily metrics: {e}")
    
    def save_portfolio_state(self):
        """Save current portfolio state to database with validation."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO portfolio_state 
                (timestamp, balance, total_pnl, daily_pnl, active_positions,
                 max_drawdown, consecutive_losses, market_regime)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                int(time.time() * 1000), 
                max(0, self.current_balance),  # Ensure non-negative
                self.total_pnl, 
                self.daily_pnl, 
                len(self.active_positions), 
                max(0, self.max_drawdown),
                max(0, self.consecutive_losses),
                self.current_market_regime
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to save portfolio state: {e}")


class ExplicitTradingActions:
    """
    Enhanced action system with comprehensive validation and market context awareness.
    """
    
    def __init__(self):
        """Initialize explicit trading actions system with enhanced validation."""
        self.setup_logging()
        
        # Position tracking with validation
        self.position_state = {
            'current_position': 0.0,    # -1.0 to 1.0 (short to long)
            'entry_price': 0.0,
            'unrealized_pnl': 0.0,
            'position_age': 0,          # Minutes since position opened
            'position_size_usd': 0.0,   # Dollar amount in position
            'entry_timestamp': 0,
            'stop_loss_price': 0.0,
            'take_profit_price': 0.0
        }
        
        # Enhanced risk thresholds with validation
        self.risk_thresholds = {
            'max_position_size': 0.25,      # 25% max position size
            'max_drawdown': 0.05,           # 5% max drawdown
            'volatility_high': 0.05,        # 5% volatility threshold
            'fear_high': 7.0,               # High fear level
            'confidence_low': 0.4,          # Low confidence threshold
            'max_price_change': 0.10,       # 10% max single price change
            'min_position_size': 0.001      # 0.1% minimum position size
        }
        
        self.logger.info("Enhanced Explicit Trading Actions System initialized")
    
    def setup_logging(self):
        """Setup logging for actions system."""
        self.logger = logging.getLogger('JARVIS_Actions')
    
    def validate_inputs(self, raw_action: int, current_price: float, market_context: Dict) -> Tuple[bool, str]:
        """Validate all inputs before processing."""
        # Validate action
        if not isinstance(raw_action, int) or raw_action not in [0, 1, 2]:
            return False, f"Invalid action: {raw_action}, must be 0, 1, or 2"
        
        # Validate price
        if not isinstance(current_price, (int, float)) or current_price <= 0:
            return False, f"Invalid current price: {current_price}, must be positive number"
        
        # Validate market context
        if not isinstance(market_context, dict):
            return False, "Market context must be a dictionary"
        
        return True, "Inputs valid"
    
    def sanitize_market_context(self, market_context: Dict) -> Dict:
        """Sanitize and validate market context data."""
        sanitized = {
            'ta_summary': [0.0, 0.0, 0.0, 0.0],
            'volatility': 'MEDIUM',
            'fear': 5.0,
            'confidence': 0.5,
            'position_count': 0,
            'historical_performance': {'win_rate': 0.5, 'avg_profit': 0.0},
            'market_condition_effectiveness': {'win_rate': 0.5}
        }
        
        # Sanitize TA summary
        ta_summary = market_context.get('ta_summary', [])
        if isinstance(ta_summary, (list, tuple)) and len(ta_summary) >= 4:
            try:
                sanitized_ta = []
                for i, val in enumerate(ta_summary[:4]):
                    # Bound TA values to reasonable range
                    clean_val = float(val) if val is not None else 0.0
                    clean_val = max(-10.0, min(10.0, clean_val))
                    sanitized_ta.append(clean_val)
                sanitized['ta_summary'] = sanitized_ta
            except (ValueError, TypeError):
                pass
        
        # Sanitize volatility
        volatility = market_context.get('volatility', 'MEDIUM')
        if volatility in ['LOW', 'MEDIUM', 'HIGH']:
            sanitized['volatility'] = volatility
        
        # Sanitize fear (0-10 scale)
        try:
            fear = float(market_context.get('fear', 5.0))
            sanitized['fear'] = max(0.0, min(10.0, fear))
        except (ValueError, TypeError):
            pass
        
        # Sanitize confidence (0-1 scale)
        try:
            confidence = float(market_context.get('confidence', 0.5))
            sanitized['confidence'] = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            pass
        
        # Sanitize position count
        try:
            position_count = int(market_context.get('position_count', 0))
            sanitized['position_count'] = max(0, position_count)
        except (ValueError, TypeError):
            pass
        
        return sanitized
    
    def update_position_state(self, position_size: float, entry_price: float, 
                             current_price: float, timestamp: int):
        """
        Update current position state with enhanced validation.
        
        Args:
            position_size: Current position size (-1.0 to 1.0)
            entry_price: Entry price of position
            current_price: Current market price
            timestamp: Current timestamp
        """
        try:
            # Validate inputs
            if not isinstance(position_size, (int, float)) or abs(position_size) > 1.0:
                self.logger.warning(f"Invalid position size: {position_size}, bounded to [-1.0, 1.0]")
                position_size = max(-1.0, min(1.0, float(position_size)))
            
            if entry_price <= 0 or current_price <= 0:
                self.logger.warning(f"Invalid prices: entry={entry_price}, current={current_price}")
                return
            
            if not isinstance(timestamp, int) or timestamp <= 0:
                self.logger.warning(f"Invalid timestamp: {timestamp}")
                timestamp = int(time.time() * 1000)
            
            self.position_state['current_position'] = position_size
            self.position_state['entry_price'] = entry_price
            
            # Calculate position age safely
            if self.position_state['entry_timestamp'] > 0:
                self.position_state['position_age'] = max(0, (timestamp - self.position_state['entry_timestamp']) / 60000)
            else:
                self.position_state['entry_timestamp'] = timestamp
                self.position_state['position_age'] = 0
            
            # Calculate unrealized P&L with validation
            if position_size != 0 and entry_price > 0 and current_price > 0:
                if position_size > 0:  # Long position
                    pnl_percent = (current_price - entry_price) / entry_price * 100
                else:  # Short position
                    pnl_percent = (entry_price - current_price) / entry_price * 100
                
                # Bound P&L to reasonable range
                self.position_state['unrealized_pnl'] = max(-100, min(1000, pnl_percent))
            else:
                self.position_state['unrealized_pnl'] = 0.0
                
        except Exception as e:
            self.logger.error(f"Error updating position state: {e}")
    
    def define_explicit_action(self, raw_action: int, current_price: float, 
                              market_context: Dict) -> Dict:
        """
        Convert neural network output to explicit financial action with comprehensive validation.
        
        Args:
            raw_action: Neural network action (0=BUY, 1=SELL, 2=HOLD)
            current_price: Current market price
            market_context: Complete market context information
            
        Returns:
            Complete explicit action dictionary with validation
        """
        try:
            # Validate inputs
            is_valid, error_msg = self.validate_inputs(raw_action, current_price, market_context)
            if not is_valid:
                self.logger.error(f"Input validation failed: {error_msg}")
                return self._get_safe_fallback_action(current_price, error_msg)
            
            # Sanitize market context
            sanitized_context = self.sanitize_market_context(market_context)
            
            # Map neural network output to action
            action_mapping = {0: 'BUY', 1: 'SELL', 2: 'HOLD'}
            base_action = action_mapping.get(raw_action, 'HOLD')
            
            # Create comprehensive explicit action
            explicit_action = {
                'base_action': base_action,
                'raw_action_code': raw_action,
                'current_position': self.position_state['current_position'],
                'current_price': current_price,
                'timestamp': int(time.time()),
                'input_validation': {'status': 'PASSED', 'message': 'All inputs valid'},
                
                # Core action understanding
                'financial_meaning': self.get_financial_meaning(base_action, current_price),
                'expected_outcome': self.predict_outcome(base_action, sanitized_context),
                'risk_assessment': self.assess_risk(base_action, sanitized_context, current_price),
                'position_impact': self.calculate_position_impact(base_action),
                
                # Trading levels with validation
                'stop_loss_level': self.calculate_stop_loss(base_action, current_price, sanitized_context),
                'take_profit_level': self.calculate_take_profit(base_action, current_price, sanitized_context),
                
                # Decision context
                'market_context': sanitized_context,
                'confidence_factors': self.analyze_confidence_factors(sanitized_context),
                'timing_analysis': self.analyze_timing(sanitized_context),
                
                # Action validation
                'action_validity': self.validate_action(base_action, sanitized_context, current_price),
                'alternative_actions': self.suggest_alternatives(base_action, sanitized_context)
            }
            
            # Final validation of action
            validation_result = self._validate_explicit_action(explicit_action)
            explicit_action['final_validation'] = validation_result
            
            if not validation_result['is_valid']:
                self.logger.warning(f"Action validation failed: {validation_result['issues']}")
                # Return safer version if validation fails
                explicit_action['base_action'] = 'HOLD'
                explicit_action['financial_meaning'] = f"HOLD due to validation issues: {validation_result['issues']}"
            
            return explicit_action
            
        except Exception as e:
            self.logger.error(f"Error in define_explicit_action: {e}")
            return self._get_safe_fallback_action(current_price, f"System error: {str(e)}")
    
    def _get_safe_fallback_action(self, current_price: float, error_reason: str) -> Dict:
        """Generate safe fallback action when errors occur."""
        return {
            'base_action': 'HOLD',
            'raw_action_code': 2,
            'current_price': max(0, current_price),
            'timestamp': int(time.time()),
            'input_validation': {'status': 'FAILED', 'message': error_reason},
            'financial_meaning': f"HOLD due to error: {error_reason}",
            'expected_outcome': "Minimal risk, preserve capital during error recovery",
            'risk_assessment': {
                'risk_level': 'LOW',
                'risk_score': 2,
                'risk_description': 'Safe fallback action'
            },
            'position_impact': {
                'impact_type': 'MAINTAIN',
                'position_change': 0.0
            },
            'stop_loss_level': 0.0,
            'take_profit_level': 0.0,
            'action_validity': {'is_valid': True, 'recommendation': 'SAFE_FALLBACK'},
            'error': True,
            'error_reason': error_reason
        }
    
    def _validate_explicit_action(self, action: Dict) -> Dict:
        """Validate the generated explicit action for consistency."""
        validation = {
            'is_valid': True,
            'issues': [],
            'warnings': []
        }
        
        try:
            # Check action consistency
            if action['base_action'] not in ['BUY', 'SELL', 'HOLD']:
                validation['issues'].append(f"Invalid base action: {action['base_action']}")
                validation['is_valid'] = False
            
            # Check price levels
            current_price = action['current_price']
            stop_loss = action.get('stop_loss_level', 0)
            take_profit = action.get('take_profit_level', 0)
            
            if action['base_action'] == 'BUY':
                if stop_loss > 0 and stop_loss >= current_price:
                    validation['issues'].append("Stop loss should be below current price for BUY")
                    validation['is_valid'] = False
                
                if take_profit > 0 and take_profit <= current_price:
                    validation['issues'].append("Take profit should be above current price for BUY")
                    validation['is_valid'] = False
            
            elif action['base_action'] == 'SELL':
                if stop_loss > 0 and stop_loss <= current_price:
                    validation['issues'].append("Stop loss should be above current price for SELL")
                    validation['is_valid'] = False
                
                if take_profit > 0 and take_profit >= current_price:
                    validation['issues'].append("Take profit should be below current price for SELL")
                    validation['is_valid'] = False
            
            # Check risk assessment
            risk_assessment = action.get('risk_assessment', {})
            risk_score = risk_assessment.get('risk_score', 5)
            
            if not isinstance(risk_score, (int, float)) or risk_score < 1 or risk_score > 10:
                validation['warnings'].append(f"Risk score out of bounds: {risk_score}")
            
            # Check position impact
            position_impact = action.get('position_impact', {})
            position_change = position_impact.get('position_change', 0)
            
            if abs(position_change) > 1.0:  # More than 100% position change
                validation['warnings'].append(f"Large position change: {position_change}")
                
        except Exception as e:
            validation['issues'].append(f"Validation error: {str(e)}")
            validation['is_valid'] = False
        
        return validation
    
    def get_financial_meaning(self, action: str, price: float) -> str:
        """
        Provide explicit financial definition with enhanced validation.
        
        Args:
            action: Trading action (BUY/SELL/HOLD)
            price: Current price (validated to be > 0)
            
        Returns:
            Human-readable financial meaning
        """
        try:
            current_pos = self.position_state['current_position']
            position_size_usd = abs(current_pos) * price * 1000  # Assuming $1000 base
            
            if action == 'BUY':
                if current_pos >= 0:
                    if abs(current_pos) < 0.01:  # Essentially no position
                        return f"OPEN LONG position at ${price:.2f} - entering bullish trade expecting price increase"
                    else:
                        return f"INCREASE LONG exposure from {current_pos:.1%} at ${price:.2f} - adding to winning position (value: ${position_size_usd:.0f})"
                else:
                    return f"REDUCE SHORT exposure from {current_pos:.1%} at ${price:.2f} - covering short position to limit losses"
                    
            elif action == 'SELL':
                if current_pos <= 0:
                    if abs(current_pos) < 0.01:  # Essentially no position
                        return f"OPEN SHORT position at ${price:.2f} - entering bearish trade expecting price decrease"
                    else:
                        return f"INCREASE SHORT exposure from {current_pos:.1%} at ${price:.2f} - adding to short position (value: ${position_size_usd:.0f})"
                else:
                    unrealized_pnl = self.position_state['unrealized_pnl']
                    if unrealized_pnl > 0:
                        return f"CLOSE LONG position from {current_pos:.1%} at ${price:.2f} - taking profits (+{unrealized_pnl:.1f}%)"
                    else:
                        return f"CLOSE LONG position from {current_pos:.1%} at ${price:.2f} - cutting losses ({unrealized_pnl:.1f}%)"
                        
            else:  # HOLD
                if abs(current_pos) < 0.01:
                    return f"STAY FLAT - no position, monitoring market at ${price:.2f} for better opportunity"
                else:
                    unrealized_pnl = self.position_state['unrealized_pnl']
                    pnl_sign = "+" if unrealized_pnl >= 0 else ""
                    position_type = "LONG" if current_pos > 0 else "SHORT"
                    return f"MAINTAIN {position_type} {abs(current_pos):.1%} position, current P&L: {pnl_sign}{unrealized_pnl:.2f}%, value: ${position_size_usd:.0f}"
                    
        except Exception as e:
            self.logger.error(f"Error generating financial meaning: {e}")
            return f"HOLD at ${price:.2f} due to calculation error"
    
    def predict_outcome(self, action: str, market_context: Dict) -> str:
        """
        Predict expected outcome with enhanced market analysis.
        
        Args:
            action: Trading action
            market_context: Sanitized market condition data
            
        Returns:
            Expected outcome description with probabilities
        """
        try:
            # Extract validated market indicators
            ta_summary = market_context['ta_summary']
            volatility = market_context['volatility']
            historical_performance = market_context['historical_performance']
            
            breakout_strength = ta_summary[0] if len(ta_summary) > 0 else 0
            momentum = ta_summary[3] if len(ta_summary) > 3 else 0
            
            # Get historical context with validation
            historical_win_rate = historical_performance.get('win_rate', 0.5)
            historical_avg_profit = historical_performance.get('avg_profit', 0)
            
            # Ensure win rate is reasonable
            if not isinstance(historical_win_rate, (int, float)) or historical_win_rate < 0 or historical_win_rate > 1:
                historical_win_rate = 0.5
            
            if action == 'BUY':
                if breakout_strength > 2.0 and momentum > 0.3:
                    expected_profit = min(2.5 + (breakout_strength * 0.5), 8.0)  # Cap at 8%
                    expected_loss = max(1.0, expected_profit * 0.4)  # Risk management
                    confidence = min(85, 60 + (breakout_strength * 5))
                    return f"Strong bullish setup: {confidence:.0f}% chance of +{expected_profit:.1f}% profit, {100-confidence:.0f}% chance of -{expected_loss:.1f}% loss (Historical: {historical_win_rate:.0%} win rate)"
                    
                elif momentum > 0.1:
                    expected_profit = min(1.5 + (momentum * 2), 5.0)
                    expected_loss = max(0.8, expected_profit * 0.5)
                    confidence = min(75, 55 + (momentum * 10))
                    return f"Moderate bullish momentum: {confidence:.0f}% chance of +{expected_profit:.1f}% profit, {100-confidence:.0f}% chance of -{expected_loss:.1f}% loss (Historical: {historical_win_rate:.0%} win rate)"
                else:
                    return f"Low conviction buy: 50% chance of +1.0% profit, 50% chance of -0.8% loss (Historical: {historical_win_rate:.0%} win rate)"
                    
            elif action == 'SELL':
                if breakout_strength < -2.0 and momentum < -0.3:
                    expected_profit = min(2.5 + (abs(breakout_strength) * 0.5), 8.0)
                    expected_loss = max(1.0, expected_profit * 0.4)
                    confidence = min(85, 60 + (abs(breakout_strength) * 5))
                    return f"Strong bearish setup: {confidence:.0f}% chance of +{expected_profit:.1f}% profit, {100-confidence:.0f}% chance of -{expected_loss:.1f}% loss (Historical: {historical_win_rate:.0%} win rate)"
                    
                elif momentum < -0.1:
                    expected_profit = min(1.5 + (abs(momentum) * 2), 5.0)
                    expected_loss = max(0.8, expected_profit * 0.5)
                    confidence = min(75, 55 + (abs(momentum) * 10))
                    return f"Moderate bearish momentum: {confidence:.0f}% chance of +{expected_profit:.1f}% profit, {100-confidence:.0f}% chance of -{expected_loss:.1f}% loss (Historical: {historical_win_rate:.0%} win rate)"
                else:
                    return f"Low conviction sell: 50% chance of +1.0% profit, 50% chance of -0.8% loss (Historical: {historical_win_rate:.0%} win rate)"
            else:  # HOLD
                # Calculate expected sideways movement
                volatility_factor = 1.5 if volatility == 'HIGH' else (0.5 if volatility == 'LOW' else 1.0)
                expected_range = 0.5 * volatility_factor
                return f"Expect sideways movement: ±{expected_range:.1f}% range, preserve capital and wait for clearer signals (Historical: {historical_win_rate:.0%} success rate)"
                
        except Exception as e:
            self.logger.error(f"Error predicting outcome: {e}")
            return "Outcome prediction unavailable due to error, proceed with caution"
    
    def assess_risk(self, action: str, market_context: Dict, current_price: float) -> Dict:
        """
        Enhanced comprehensive risk assessment with detailed analysis.
        
        Args:
            action: Trading action
            market_context: Sanitized market condition data
            current_price: Current market price
            
        Returns:
            Comprehensive risk assessment dictionary
        """
        try:
            # Extract validated risk factors
            volatility = market_context['volatility']
            fear = market_context['fear']  # 0-10 scale
            confidence = market_context['confidence']  # 0-1 scale
            position_count = market_context['position_count']
            
            # Calculate base risk score (1-10 scale) with detailed breakdown
            risk_components = {
                'base_risk': 5.0,  # Neutral baseline
                'volatility_risk': 0.0,
                'fear_risk': 0.0,
                'confidence_risk': 0.0,
                'concentration_risk': 0.0,
                'action_risk': 0.0,
                'market_regime_risk': 0.0
            }
            
            # Volatility risk assessment
            if volatility == 'HIGH':
                risk_components['volatility_risk'] = 2.5
            elif volatility == 'LOW':
                risk_components['volatility_risk'] = -0.5
            
            # Fear-based risk adjustment
            risk_components['fear_risk'] = (fear - 5.0) * 0.3  # Scale fear to risk
            
            # Confidence risk (low confidence = higher risk)
            risk_components['confidence_risk'] = (0.5 - confidence) * 2.0
            
            # Position concentration risk
            if position_count > 3:
                risk_components['concentration_risk'] = min((position_count - 3) * 0.4, 2.0)
            
            # Action-specific risk
            if action in ['BUY', 'SELL']:
                risk_components['action_risk'] = 0.8  # Active trading adds risk
            
            # Market regime risk (if available)
            # This could be enhanced with additional market data
            
            # Calculate total risk score
            total_risk = sum(risk_components.values())
            risk_score = max(1.0, min(10.0, total_risk))
            
            # Risk categorization with enhanced descriptions
            if risk_score <= 3.0:
                risk_level = 'LOW'
                risk_description = 'Favorable market conditions with manageable risk profile'
                action_recommendation = 'Normal position sizing acceptable'
                max_position_multiplier = 1.2
            elif risk_score <= 5.5:
                risk_level = 'MEDIUM'
                risk_description = 'Moderate risk environment requiring standard precautions'
                action_recommendation = 'Use standard risk management protocols'
                max_position_multiplier = 1.0
            elif risk_score <= 7.5:
                risk_level = 'HIGH'
                risk_description = 'Elevated risk conditions requiring reduced exposure'
                action_recommendation = 'Reduce position sizes and use tighter stops'
                max_position_multiplier = 0.6
            else:
                risk_level = 'EXTREME'
                risk_description = 'High-risk environment - consider avoiding new positions'
                action_recommendation = 'Avoid new positions, consider closing existing ones'
                max_position_multiplier = 0.3
            
            # Calculate dynamic risk limits
            base_position_size = self.risk_thresholds['max_position_size']
            adjusted_max_position = base_position_size * max_position_multiplier
            
            # Risk-adjusted stop loss
            base_stop = 0.02  # 2% base stop
            risk_adjusted_stop = base_stop * (1 + (risk_score - 5) * 0.1)  # Adjust based on risk
            max_stop_loss = max(0.005, min(0.08, risk_adjusted_stop))
            
            # Expected risk/reward ratio
            risk_reward_ratio = max(1.5, 3.0 - (risk_score - 5) * 0.2)  # Better ratio in lower risk
            
            return {
                'risk_score': round(risk_score, 2),
                'risk_level': risk_level,
                'risk_description': risk_description,
                'action_recommendation': action_recommendation,
                'max_position_size': round(adjusted_max_position, 3),
                'max_stop_loss': round(max_stop_loss, 4),
                'recommended_risk_reward_ratio': round(risk_reward_ratio, 1),
                'risk_components': {k: round(v, 2) for k, v in risk_components.items()},
                'risk_factors': self._identify_risk_factors(market_context, risk_score),
                'mitigation_strategies': self._generate_mitigation_strategies(risk_level, market_context)
            }
            
        except Exception as e:
            self.logger.error(f"Error in risk assessment: {e}")
            return {
                'risk_score': 8.0,  # Conservative fallback
                'risk_level': 'HIGH',
                'risk_description': f'Risk assessment error: {str(e)}',
                'action_recommendation': 'Use conservative approach due to assessment error',
                'max_position_size': 0.05,  # Very conservative
                'max_stop_loss': 0.015,
                'error': True
            }
    
    def _identify_risk_factors(self, market_context: Dict, risk_score: float) -> List[str]:
        """Identify specific risk factors contributing to the risk score."""
        factors = []
        
        volatility = market_context['volatility']
        fear = market_context['fear']
        confidence = market_context['confidence']
        position_count = market_context['position_count']
        
        if volatility == 'HIGH':
            factors.append("High market volatility increases price uncertainty")
        elif volatility == 'LOW':
            factors.append("Low volatility provides more predictable price action")
        
        if fear > 7.0:
            factors.append("Elevated fear levels suggest market stress")
        elif fear < 3.0:
            factors.append("Low fear levels support risk-taking")
        
        if confidence < 0.3:
            factors.append("Low confidence in trading signals")
        elif confidence > 0.8:
            factors.append("High confidence supports position taking")
        
        if position_count > 3:
            factors.append(f"High position concentration ({position_count} active positions)")
        
        if risk_score > 7.5:
            factors.append("Overall risk environment suggests extreme caution")
        
        return factors if factors else ["No significant risk factors identified"]
    
    def _generate_mitigation_strategies(self, risk_level: str, market_context: Dict) -> List[str]:
        """Generate specific risk mitigation strategies."""
        strategies = []
        
        if risk_level == 'EXTREME':
            strategies.extend([
                "Avoid opening new positions",
                "Consider closing existing positions",
                "Wait for risk conditions to improve",
                "Increase cash reserves"
            ])
        elif risk_level == 'HIGH':
            strategies.extend([
                "Reduce position sizes by 40-50%",
                "Use tighter stop losses",
                "Take partial profits more aggressively",
                "Avoid adding to losing positions"
            ])
        elif risk_level == 'MEDIUM':
            strategies.extend([
                "Use standard position sizing",
                "Maintain disciplined stop losses",
                "Monitor positions more closely",
                "Be prepared to reduce exposure quickly"
            ])
        else:  # LOW
            strategies.extend([
                "Normal position sizing acceptable",
                "Can consider slightly larger positions",
                "Standard stop loss levels appropriate",
                "Good environment for active trading"
            ])
        
        # Add context-specific strategies
        volatility = market_context['volatility']
        if volatility == 'HIGH':
            strategies.append("Use wider stop losses to account for volatility")
        
        fear = market_context['fear']
        if fear > 7.0:
            strategies.append("Wait for fear to subside before increasing exposure")
        
        return strategies
    
    def calculate_position_impact(self, action: str) -> Dict:
        """
        Calculate position impact with enhanced validation and analysis.
        
        Args:
            action: Trading action
            
        Returns:
            Detailed position impact analysis
        """
        try:
            current_pos = self.position_state['current_position']
            
            # Validate current position
            current_pos = max(-1.0, min(1.0, current_pos))  # Bound to valid range
            
            # Standard position sizing with risk adjustment
            base_size = 0.1  # 10% base position size
            max_size = self.risk_thresholds['max_position_size']
            min_size = self.risk_thresholds['min_position_size']
            
            # Calculate new position based on action
            if action == 'BUY':
                if current_pos >= 0:
                    new_position = min(current_pos + base_size, max_size)
                    change = new_position - current_pos
                    impact_type = 'INCREASE_LONG' if current_pos > min_size else 'OPEN_LONG'
                else:
                    # Covering short position
                    new_position = min(current_pos + base_size, 0.0)
                    change = new_position - current_pos
                    impact_type = 'COVER_SHORT'
                    
            elif action == 'SELL':
                if current_pos <= 0:
                    new_position = max(current_pos - base_size, -max_size)
                    change = new_position - current_pos
                    impact_type = 'INCREASE_SHORT' if current_pos < -min_size else 'OPEN_SHORT'
                else:
                    # Reducing long position
                    new_position = max(current_pos - base_size, 0.0)
                    change = new_position - current_pos
                    impact_type = 'REDUCE_LONG'
            else:  # HOLD
                new_position = current_pos
                change = 0.0
                impact_type = 'MAINTAIN'
            
            # Calculate leverage and risk metrics
            leverage_change = abs(new_position) - abs(current_pos)
            risk_exposure_change = abs(change) * 100  # Convert to percentage points
            
            # Calculate estimated dollar impact (assuming $10,000 portfolio)
            estimated_portfolio_value = 10000.0
            dollar_impact = abs(change) * estimated_portfolio_value
            
            return {
                'current_position': round(current_pos, 4),
                'new_position': round(new_position, 4),
                'position_change': round(change, 4),
                'position_change_percent': round(change * 100, 2),
                'impact_type': impact_type,
                'leverage_impact': round(leverage_change, 4),
                'risk_exposure_change': round(risk_exposure_change, 2),
                'estimated_dollar_impact': round(dollar_impact, 2),
                'portfolio_allocation': {
                    'current_allocation': round(abs(current_pos) * 100, 2),
                    'new_allocation': round(abs(new_position) * 100, 2),
                    'allocation_change': round(abs(change) * 100, 2)
                },
                'risk_metrics': {
                    'position_within_limits': abs(new_position) <= max_size,
                    'position_above_minimum': abs(new_position) >= min_size or new_position == 0,
                    'safe_sizing': abs(change) <= base_size * 1.5
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating position impact: {e}")
            return {
                'current_position': 0.0,
                'new_position': 0.0,
                'position_change': 0.0,
                'impact_type': 'ERROR',
                'error': str(e)
            }
    
    def calculate_stop_loss(self, action: str, current_price: float, market_context: Dict) -> float:
        """
        Calculate stop loss with enhanced bounds checking and market adaptation.
        
        Args:
            action: Trading action ('BUY', 'SELL', 'HOLD')
            current_price: Current market price (validated > 0)
            market_context: Sanitized market condition data
            
        Returns:
            Stop loss price level (0.0 if HOLD or invalid)
        """
        try:
            if action == 'HOLD' or current_price <= 0:
                return 0.0
            
            # Base stop loss parameters with strict bounds
            base_distance = 0.015  # 1.5% base
            min_distance = 0.003   # 0.3% absolute minimum
            max_distance = 0.08    # 8.0% absolute maximum
            
            # Extract market factors
            volatility = market_context['volatility']
            confidence = market_context['confidence']
            fear = market_context['fear']
            
            # Volatility adjustment
            volatility_multipliers = {
                'HIGH': 2.0,    # Much wider stops for high volatility
                'MEDIUM': 1.0,  # Standard stops
                'LOW': 0.7      # Tighter stops for low volatility
            }
            distance_multiplier = volatility_multipliers.get(volatility, 1.0)
            
            # Confidence adjustment (higher confidence = tighter stops)
            if confidence > 0.8:
                distance_multiplier *= 0.8  # 20% tighter
            elif confidence < 0.3:
                distance_multiplier *= 1.4  # 40% wider
            
            # Fear adjustment (higher fear = wider stops for safety)
            fear_adjustment = 1.0 + (fear - 5.0) * 0.1  # ±50% based on fear
            distance_multiplier *= max(0.5, min(2.0, fear_adjustment))
            
            # Calculate final stop distance with strict bounds
            stop_distance = base_distance * distance_multiplier
            stop_distance = max(min_distance, min(max_distance, stop_distance))
            
            # Calculate stop price
            if action == 'BUY':
                stop_price = current_price * (1 - stop_distance)
            else:  # SELL
                stop_price = current_price * (1 + stop_distance)
            
            # Final validation
            if stop_price <= 0:
                self.logger.warning(f"Invalid stop price calculated: {stop_price}")
                return 0.0
            
            # Ensure stop is reasonable distance from current price
            actual_distance = abs(stop_price - current_price) / current_price
            if actual_distance < min_distance or actual_distance > max_distance:
                self.logger.warning(f"Stop distance {actual_distance:.1%} outside bounds, recalculating")
                # Recalculate with bound distance
                bounded_distance = max(min_distance, min(max_distance, actual_distance))
                if action == 'BUY':
                    stop_price = current_price * (1 - bounded_distance)
                else:
                    stop_price = current_price * (1 + bounded_distance)
            
            return round(stop_price, 6)  # Round to reasonable precision
            
        except Exception as e:
            self.logger.error(f"Error calculating stop loss: {e}")
            return 0.0
    
    def calculate_take_profit(self, action: str, current_price: float, market_context: Dict) -> float:
        """
        Calculate take profit with risk/reward optimization and market conditions.
        
        Args:
            action: Trading action ('BUY', 'SELL', 'HOLD')
            current_price: Current market price (validated > 0)
            market_context: Sanitized market condition data
            
        Returns:
            Take profit price level (0.0 if HOLD or invalid)
        """
        try:
            if action == 'HOLD' or current_price <= 0:
                return 0.0
            
            # Calculate stop loss distance for risk/reward calculation
            stop_price = self.calculate_stop_loss(action, current_price, market_context)
            if stop_price <= 0:
                return 0.0
            
            # Calculate actual stop distance
            stop_distance = abs(stop_price - current_price) / current_price
            
            # Base risk/reward ratio with market adjustments
            base_risk_reward = 2.0  # 2:1 reward to risk
            min_risk_reward = 1.2   # Minimum acceptable
            max_risk_reward = 4.0   # Maximum target
            
            # Adjust ratio based on market conditions
            ta_summary = market_context['ta_summary']
            volatility = market_context['volatility']
            confidence = market_context['confidence']
            
            # Momentum adjustment
            if len(ta_summary) >= 4:
                momentum = ta_summary[3]
                if abs(momentum) > 0.3:  # Strong momentum
                    base_risk_reward *= 1.5  # Extend targets in strong trends
                elif abs(momentum) < 0.1:  # Weak momentum
                    base_risk_reward *= 0.8  # Reduce targets in choppy conditions
            
            # Volatility adjustment
            if volatility == 'HIGH':
                base_risk_reward *= 1.2  # Slightly higher targets to account for noise
            elif volatility == 'LOW':
                base_risk_reward *= 0.9  # Slightly lower targets in calm conditions
            
            # Confidence adjustment
            if confidence > 0.8:
                base_risk_reward *= 1.3  # Higher targets with high confidence
            elif confidence < 0.3:
                base_risk_reward *= 0.7  # Lower targets with low confidence
            
            # Bound the risk/reward ratio
            risk_reward_ratio = max(min_risk_reward, min(max_risk_reward, base_risk_reward))
            
            # Calculate take profit distance
            tp_distance = stop_distance * risk_reward_ratio
            
            # Apply maximum take profit limits
            max_tp_distance = 0.15  # 15% maximum take profit
            tp_distance = min(tp_distance, max_tp_distance)
            
            # Calculate take profit price
            if action == 'BUY':
                tp_price = current_price * (1 + tp_distance)
            else:  # SELL
                tp_price = current_price * (1 - tp_distance)
            
            # Final validation
            if tp_price <= 0:
                self.logger.warning(f"Invalid take profit price calculated: {tp_price}")
                return 0.0
            
            return round(tp_price, 6)  # Round to reasonable precision
            
        except Exception as e:
            self.logger.error(f"Error calculating take profit: {e}")
            return 0.0
    
    def analyze_confidence_factors(self, market_context: Dict) -> List[str]:
        """
        Analyze factors affecting decision confidence with enhanced detail.
        
        Args:
            market_context: Sanitized market condition data
            
        Returns:
            List of detailed confidence factor descriptions
        """
        try:
            factors = []
            
            # Technical analysis strength
            ta_summary = market_context['ta_summary']
            if len(ta_summary) >= 4:
                breakout_strength = abs(ta_summary[0])
                momentum = abs(ta_summary[3])
                
                if breakout_strength > 2.0:
                    factors.append(f"Strong technical breakout signal (strength: {breakout_strength:.1f})")
                elif breakout_strength > 1.0:
                    factors.append(f"Moderate breakout signal (strength: {breakout_strength:.1f})")
                
                if momentum > 0.5:
                    factors.append(f"Strong momentum confirmation (momentum: {momentum:.1f})")
                elif momentum > 0.2:
                    factors.append(f"Moderate momentum (momentum: {momentum:.1f})")
                elif momentum < 0.1:
                    factors.append("Weak momentum suggests caution")
            
            # Historical performance analysis
            historical = market_context['historical_performance']
            win_rate = historical.get('win_rate', 0.5)
            avg_profit = historical.get('avg_profit', 0)
            
            if win_rate > 0.75:
                factors.append(f"Excellent historical pattern success ({win_rate:.0%} win rate)")
            elif win_rate > 0.65:
                factors.append(f"Good historical pattern performance ({win_rate:.0%} win rate)")
            elif win_rate < 0.35:
                factors.append(f"Poor historical pattern performance ({win_rate:.0%} win rate)")
            
            if avg_profit > 0.02:  # > 2% average profit
                factors.append(f"Strong historical profitability ({avg_profit:.1%} avg profit)")
            elif avg_profit < -0.01:  # < -1% average
                factors.append(f"Historical losses concern ({avg_profit:.1%} avg)")
            
            # Market conditions analysis
            volatility = market_context['volatility']
            fear = market_context['fear']
            confidence = market_context['confidence']
            
            if volatility == 'HIGH':
                factors.append("High volatility increases uncertainty and risk")
            elif volatility == 'LOW':
                factors.append("Low volatility provides stable trading environment")
            
            if fear > 8.0:
                factors.append("Extreme fear levels suggest market stress")
            elif fear > 6.0:
                factors.append("Elevated fear suggests caution")
            elif fear < 3.0:
                factors.append("Low fear supports risk-taking")
            
            if confidence > 0.85:
                factors.append("Very high confidence in analysis")
            elif confidence > 0.65:
                factors.append("Good confidence in trading signals")
            elif confidence < 0.35:
                factors.append("Low confidence suggests waiting for better setup")
            
            # Position context
            position_count = market_context['position_count']
            if position_count > 4:
                factors.append(f"High position concentration ({position_count} positions) adds risk")
            elif position_count == 0:
                factors.append("No existing positions - fresh opportunity")
            
            return factors if factors else ["No significant confidence factors identified"]
            
        except Exception as e:
            self.logger.error(f"Error analyzing confidence factors: {e}")
            return [f"Confidence analysis error: {str(e)}"]
    
    def analyze_timing(self, market_context: Dict) -> Dict:
        """
        Analyze timing factors with enhanced market session awareness.
        
        Args:
            market_context: Sanitized market condition data
            
        Returns:
            Comprehensive timing analysis dictionary
        """
        try:
            current_time = datetime.now()
            hour = current_time.hour
            weekday = current_time.weekday()  # 0=Monday, 6=Sunday
            
            # Market session analysis
            session = self.get_market_session(hour)
            
            # Volume and liquidity assessment
            liquidity_assessment = self.assess_liquidity(hour, weekday)
            
            # Volatility timing
            volatility_timing = self.assess_volatility_timing(hour, market_context['volatility'])
            
            # Overall timing recommendation
            timing_score = self.calculate_timing_score(hour, weekday, session, liquidity_assessment)
            
            return {
                'current_session': session,
                'hour_of_day': hour,
                'day_of_week': current_time.strftime('%A'),
                'liquidity_assessment': liquidity_assessment,
                'volatility_timing': volatility_timing,
                'timing_score': timing_score,
                'timing_recommendation': self.get_timing_recommendation(timing_score),
                'optimal_entry_window': timing_score > 7,
                'risk_factors': self.identify_timing_risks(hour, weekday, session)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing timing: {e}")
            return {
                'error': str(e),
                'timing_recommendation': 'Timing analysis unavailable',
                'optimal_entry_window': False
            }
    
    def get_market_session(self, hour: int) -> str:
        """Determine current market session with crypto considerations."""
        # Crypto markets are 24/7, but still have patterns
        if 0 <= hour < 8:
            return 'ASIAN'
        elif 8 <= hour < 16:
            return 'EUROPEAN'  
        else:
            return 'AMERICAN'
    
    def assess_liquidity(self, hour: int, weekday: int) -> Dict:
        """Assess market liquidity based on time patterns."""
        # Peak crypto trading hours typically align with major forex sessions
        liquidity_score = 5  # Base score 1-10
        
        # Hour-based liquidity
        if 8 <= hour <= 12 or 14 <= hour <= 18:  # European/US overlap
            liquidity_score += 3
        elif 20 <= hour <= 24 or 0 <= hour <= 4:  # Lower activity
            liquidity_score -= 2
        
        # Day-based liquidity (weekends typically lower)
        if weekday >= 5:  # Weekend
            liquidity_score -= 1
        elif weekday == 0:  # Monday
            liquidity_score += 1
        
        liquidity_score = max(1, min(10, liquidity_score))
        
        if liquidity_score >= 8:
            level = 'HIGH'
            description = 'Peak trading hours with high liquidity'
        elif liquidity_score >= 6:
            level = 'MEDIUM'
            description = 'Moderate liquidity conditions'
        else:
            level = 'LOW'
            description = 'Lower liquidity period - larger spreads possible'
        
        return {
            'level': level,
            'score': liquidity_score,
            'description': description
        }
    
    def assess_volatility_timing(self, hour: int, volatility: str) -> Dict:
        """Assess if current time is good for trading given volatility."""
        # Avoid high volatility during low liquidity periods
        risk_level = 'MEDIUM'
        recommendation = 'Standard timing approach'
        
        if volatility == 'HIGH':
            if hour <= 6 or hour >= 22:  # Low liquidity + high volatility
                risk_level = 'HIGH'
                recommendation = 'Avoid trading during high volatility + low liquidity'
            else:
                risk_level = 'MEDIUM'
                recommendation = 'High volatility during good liquidity hours'
        elif volatility == 'LOW':
            risk_level = 'LOW'
            recommendation = 'Good conditions for trading any time'
        
        return {
            'risk_level': risk_level,
            'recommendation': recommendation,
            'volatility_context': f'{volatility} volatility at {hour:02d}:00'
        }
    
    def calculate_timing_score(self, hour: int, weekday: int, session: str, liquidity: Dict) -> int:
        """Calculate overall timing score (1-10)."""
        score = 5  # Base score
        
        # Add liquidity score impact
        score += (liquidity['score'] - 5) * 0.5
        
        # Session bonuses
        if session in ['EUROPEAN', 'AMERICAN']:
            score += 1
        
        # Avoid weekend trading
        if weekday >= 5:
            score -= 1
        
        # Prime time bonus
        if 9 <= hour <= 17:
            score += 1
        
        return max(1, min(10, int(score)))
    
    def get_timing_recommendation(self, timing_score: int) -> str:
        """Get timing recommendation based on score."""
        if timing_score >= 8:
            return "Excellent timing - peak trading conditions"
        elif timing_score >= 6:
            return "Good timing - suitable for trading"
        elif timing_score >= 4:
            return "Fair timing - proceed with caution"
        else:
            return "Poor timing - consider waiting for better conditions"
    
    def identify_timing_risks(self, hour: int, weekday: int, session: str) -> List[str]:
        """Identify specific timing-related risks."""
        risks = []
        
        if weekday >= 5:
            risks.append("Weekend trading - typically lower volume")
        
        if hour <= 4 or hour >= 22:
            risks.append("Low liquidity hours - wider spreads possible")
        
        if session == 'ASIAN' and weekday == 0:
            risks.append("Monday Asian session - market reopening volatility")
        
        if 12 <= hour <= 14:
            risks.append("European lunch break - reduced activity")
        
        return risks
    
    def validate_action(self, action: str, market_context: Dict, current_price: float) -> Dict:
        """
        Enhanced action validation with comprehensive checks.
        
        Args:
            action: Trading action
            market_context: Sanitized market condition data
            current_price: Current market price
            
        Returns:
            Comprehensive validation result
        """
        try:
            validation = {
                'is_valid': True,
                'confidence_level': 'HIGH',
                'validation_score': 8,  # 1-10 scale
                'issues': [],
                'warnings': [],
                'recommendations': []
            }
            
            # Basic action validation
            if action not in ['BUY', 'SELL', 'HOLD']:
                validation['issues'].append(f"Invalid action: {action}")
                validation['is_valid'] = False
                validation['validation_score'] = 1
                return validation
            
            # Price validation
            if current_price <= 0:
                validation['issues'].append(f"Invalid price: {current_price}")
                validation['is_valid'] = False
                validation['validation_score'] = 1
                return validation
            
            # Risk level validation
            confidence = market_context['confidence']
            fear = market_context['fear']
            volatility = market_context['volatility']
            
            # Confidence checks
            if confidence < 0.3 and action in ['BUY', 'SELL']:
                validation['warnings'].append(f"Low confidence ({confidence:.1%}) for active trade")
                validation['validation_score'] -= 2
                validation['confidence_level'] = 'LOW'
            
            # Fear checks
            if fear > 8.0 and action in ['BUY', 'SELL']:
                validation['warnings'].append(f"High fear level ({fear:.1f}/10) suggests caution")
                validation['validation_score'] -= 1
            
            # Volatility checks
            if volatility == 'HIGH':
                validation['warnings'].append("High volatility increases risk")
                validation['validation_score'] -= 1
                validation['recommendations'].append("Consider smaller position sizes")
            
            # Position concentration check
            position_count = market_context['position_count']
            if position_count > 3 and action in ['BUY', 'SELL']:
                validation['warnings'].append(f"High position concentration ({position_count} positions)")
                validation['validation_score'] -= 1
                validation['recommendations'].append("Consider position limits")
            
            # Market condition coherence
            ta_summary = market_context['ta_summary']
            if len(ta_summary) >= 4:
                momentum = ta_summary[3]
                if action == 'BUY' and momentum < -0.3:
                    validation['warnings'].append("Buying against strong negative momentum")
                    validation['validation_score'] -= 2
                elif action == 'SELL' and momentum > 0.3:
                    validation['warnings'].append("Selling against strong positive momentum")
                    validation['validation_score'] -= 2
            
            # Timing validation
            timing_analysis = self.analyze_timing(market_context)
            if not timing_analysis.get('optimal_entry_window', True):
                validation['warnings'].append("Suboptimal timing conditions")
                validation['validation_score'] -= 1
            
            # Final validation score and confidence
            validation['validation_score'] = max(1, validation['validation_score'])
            
            if validation['validation_score'] >= 8:
                validation['confidence_level'] = 'HIGH'
                validation['recommendation'] = 'PROCEED'
            elif validation['validation_score'] >= 6:
                validation['confidence_level'] = 'MEDIUM'
                validation['recommendation'] = 'PROCEED_WITH_CAUTION'
            elif validation['validation_score'] >= 4:
                validation['confidence_level'] = 'LOW'
                validation['recommendation'] = 'CAUTION'
            else:
                validation['confidence_level'] = 'VERY_LOW'
                validation['recommendation'] = 'AVOID'
            
            # Add specific recommendations
            if validation['validation_score'] < 6:
                validation['recommendations'].append("Consider waiting for better conditions")
            if len(validation['warnings']) > 2:
                validation['recommendations'].append("Multiple risk factors present - reduce exposure")
            
            return validation
            
        except Exception as e:
            self.logger.error(f"Error in action validation: {e}")
            return {
                'is_valid': False,
                'confidence_level': 'ERROR',
                'validation_score': 1,
                'issues': [f"Validation error: {str(e)}"],
                'warnings': [],
                'recommendations': ['Use conservative approach due to validation error'],
                'error': True
            }
    
    def suggest_alternatives(self, action: str, market_context: Dict) -> List[str]:
        """
        Suggest alternative actions with enhanced reasoning.
        
        Args:
            action: Current proposed action
            market_context: Sanitized market condition data
            
        Returns:
            List of alternative action suggestions with reasoning
        """
        try:
            alternatives = []
            
            confidence = market_context['confidence']
            fear = market_context['fear']
            volatility = market_context['volatility']
            position_count = market_context['position_count']
            
            if action == 'BUY':
                if confidence < 0.4:
                    alternatives.append("HOLD - Wait for higher confidence signals before buying")
                if fear > 7.0:
                    alternatives.append("Reduce position size by 50% - High fear suggests smaller positions")
                if volatility == 'HIGH':
                    alternatives.append("Use wider stop losses - High volatility requires more room")
                    alternatives.append("Consider scaling into position - Reduce timing risk")
                if position_count > 3:
                    alternatives.append("Close existing position first - Manage concentration risk")
                
            elif action == 'SELL':
                if confidence < 0.4:
                    alternatives.append("HOLD - Wait for clearer directional signals")
                if fear > 7.0:
                    alternatives.append("Partial position close - Take some profit but keep exposure")
                if volatility == 'HIGH':
                    alternatives.append("Use wider stop losses - Avoid premature stops")
                    alternatives.append("Scale out gradually - Reduce execution risk")
                
            else:  # HOLD
                if confidence > 0.8 and fear < 4.0:
                    alternatives.append("Consider small position - Strong signals support action")
                if volatility == 'LOW' and confidence > 0.6:
                    alternatives.append("Good conditions for position taking - Low risk environment")
                
            # General alternatives based on market conditions
            if volatility == 'HIGH':
                alternatives.append("Wait for volatility to decrease - Better risk/reward when calm")
            
            if position_count == 0 and confidence > 0.6:
                alternatives.append("Consider starting with small position - No current exposure")
            
            # Timing alternatives
            timing_analysis = self.analyze_timing(market_context)
            if not timing_analysis.get('optimal_entry_window', True):
                alternatives.append("Wait for better timing window - Current timing suboptimal")
            
            # Default if no alternatives
            if not alternatives:
                if action == 'HOLD':
                    alternatives.append("Current HOLD approach appears optimal")
                else:
                    alternatives.append("Consider reducing position size - More conservative approach")
                    alternatives.append("Use wider stop losses - Provide more room for market noise")
            
            return alternatives
            
        except Exception as e:
            self.logger.error(f"Error suggesting alternatives: {e}")
            return [f"Alternative analysis error: {str(e)}"]


def main():
    """Test the enhanced position manager with comprehensive validation."""
    print("💰 JARVIS 3.0 - Enhanced Position & Risk Management System")
    print("=" * 70)
    
    try:
        # Initialize enhanced position manager
        position_manager = PositionManager()
        
        # Perform health check
        health = position_manager.perform_health_check()
        print(f"\n🏥 SYSTEM HEALTH CHECK")
        print(f"Overall Status: {health['overall_status']}")
        print(f"Active Positions: {health['metrics']['active_positions']}")
        print(f"Current Balance: ${health['metrics']['current_balance']:,.2f}")
        print(f"Emergency Stop: {health['metrics']['emergency_stop']}")
        
        if health['issues']:
            print(f"Issues: {', '.join(health['issues'])}")
        if health['warnings']:
            print(f"Warnings: {', '.join(health['warnings'])}")
        
        # Test enhanced risk assessment
        print(f"\n🔍 TESTING ENHANCED RISK ASSESSMENT")
        test_market_context = {
            'volatility': 'HIGH',
            'fear': 6.5,
            'confidence': 0.7,
            'position_count': 2,
            'ta_summary': [2.1, -0.8, 1.2, 0.4],
            'historical_performance': {'win_rate': 0.68, 'avg_profit': 0.015}
        }
        
        risk_assessment = position_manager.assess_risk('BUY', test_market_context, 50000.0)
        print(f"Risk Level: {risk_assessment['risk_level']}")
        print(f"Risk Score: {risk_assessment['risk_score']}/10")
        print(f"Max Position Size: {risk_assessment['max_position_size']:.1%}")
        print(f"Recommendation: {risk_assessment['action_recommendation']}")
        
        # Test explicit actions system
        print(f"\n🎯 TESTING EXPLICIT ACTIONS SYSTEM")
        actions_system = ExplicitTradingActions()
        
        explicit_action = actions_system.define_explicit_action(0, 50000.0, test_market_context)
        print(f"Action: {explicit_action['base_action']}")
        print(f"Financial Meaning: {explicit_action['financial_meaning']}")
        print(f"Risk Level: {explicit_action['risk_assessment']['risk_level']}")
        print(f"Validation: {explicit_action['action_validity']['recommendation']}")
        
        # Test position opening with validation
        print(f"\n📊 TESTING POSITION OPENING")
        can_open, reason = position_manager.can_open_position("BTCUSDT", 0.05, 50000.0, test_market_context)
        print(f"Can Open Position: {can_open}")
        print(f"Reason: {reason}")
        
        if can_open:
            strategy_params = {
                'stop_loss': 0.02,
                'profit_target': 0.04,
                'trailing_stop': 0.015
            }
            
            position_id = position_manager.open_position(
                symbol="BTCUSDT",
                side="LONG", 
                size=0.03,  # Smaller size due to risk
                current_price=50000.0,
                strategy_params=strategy_params,
                strategy_mode="test_enhanced",
                market_context=test_market_context
            )
            
            if position_id:
                print(f"✅ Opened position: {position_id}")
                
                # Test position update
                market_data = {"BTCUSDT": 50500.0}
                position_manager.update_positions(market_data)
                print(f"✅ Updated positions with new price data")
                
                # Get comprehensive portfolio summary
                summary = position_manager.get_portfolio_summary()
                print(f"\n📈 PORTFOLIO SUMMARY")
                print(f"Current Balance: ${summary['current_balance']:,.2f}")
                print(f"Total P&L: {summary['total_pnl_percent']:.2f}%")
                print(f"Active Positions: {summary['active_positions']}")
                print(f"Portfolio Health: {summary['risk_metrics']['portfolio_health']}")
        
        print(f"\n🎉 Enhanced Position Manager: FULLY OPERATIONAL!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())