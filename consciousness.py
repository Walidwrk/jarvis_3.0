#!/usr/bin/env python3
"""
JARVIS 3.0 - Adaptive Strategy Consciousness Module
3-Emotion System with 5 Adaptive Trading Strategies

This module implements the consciousness system with 3 core emotions and 5 adaptive
trading strategies that learn and adapt to different market conditions.

Author: JARVIS 3.0 Team
Version: 2.0 (Adaptive Strategy Implementation)
"""

import sqlite3
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List
import json
import os

class StrategyModes:
    """Strategy mode constants for different market conditions."""
    VOLATILE_MARKET = "volatile_market"    # High volatility periods
    TRENDING_MARKET = "trending_market"    # Strong directional moves
    RANGING_MARKET = "ranging_market"      # Sideways/choppy markets
    BREAKOUT_MODE = "breakout_mode"        # Breakout from consolidation
    REVERSAL_MODE = "reversal_mode"        # Trend reversal patterns

class SafetyLimits:
    """Hard safety limits that apply to all strategies."""
    MAX_LOSS_PER_TRADE = 1.0      # 1% maximum loss per trade
    DAILY_DRAWDOWN_LIMIT = 3.0    # 3% daily drawdown limit  
    MAX_TOTAL_DRAWDOWN = 8.0      # 8% maximum total drawdown
    MAX_POSITION_SIZE = 0.20      # 20% maximum position size

class AdaptiveStrategyConsciousness:
    """
    Advanced consciousness system implementing 3-emotion based decision making
    with 5 adaptive trading strategies for different market conditions.
    """
    
    def __init__(self, db_path: str = "data/crypto_data.db", log_path: str = "logs/"):
        """
        Initialize the adaptive strategy consciousness system.
        
        Args:
            db_path: Path to SQLite database file
            log_path: Path to logs directory
        """
        self.db_path = db_path
        self.log_path = log_path
        
        # Core emotions (0.0 to 10.0 scale)
        self.confidence = 5.0    # Influences strategy aggressiveness
        self.fear = 5.0          # Controls risk management across all strategies
        self.greed = 5.0         # Influences profit targets and holding time
        
        # Current state
        self.current_strategy = StrategyModes.RANGING_MARKET
        self.current_market_condition = None
        self.daily_pnl = 0.0
        self.session_id = f"session_{int(time.time())}"
        
        # Performance tracking
        self.strategy_performance = {}
        self.emotion_history = []
        
        # Setup logging
        self._setup_logging()
        
        # Initialize consciousness database
        self._initialize_consciousness_db()
        
        # Load previous strategy performance
        self._load_strategy_performance()
        
        self.logger.info("JARVIS 3.0 Adaptive Strategy Consciousness initialized")
    
    def _setup_logging(self):
        """Setup comprehensive logging system."""
        log_file = os.path.join(self.log_path, "jarvis_consciousness.log")
        
        # Create logs directory if it doesn't exist
        os.makedirs(self.log_path, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('JARVIS_Consciousness')
    
    def _initialize_consciousness_db(self):
        """Initialize consciousness-specific database tables."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Consciousness state table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS consciousness_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER,
                    confidence REAL,
                    fear REAL,
                    greed REAL,
                    current_strategy TEXT,
                    market_condition TEXT,
                    daily_pnl REAL,
                    session_id TEXT
                )
            """)
            
            # Strategy performance table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS strategy_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_mode TEXT,
                    market_condition TEXT,
                    wins INTEGER,
                    losses INTEGER,
                    total_pnl REAL,
                    avg_profit REAL,
                    last_updated INTEGER
                )
            """)
            
            conn.commit()
            conn.close()
            self.logger.info("Consciousness database tables initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize consciousness database: {e}")
            raise
    
    def clean_corrupted_database(self):
        """Clean corrupted database entries and reset if needed."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check for corrupted data and clean it
            cursor.execute("SELECT id, avg_profit FROM strategy_performance")
            corrupted_rows = []
            
            for row in cursor.fetchall():
                try:
                    # Try to convert avg_profit to float
                    float(row[1]) if row[1] is not None else 0.0
                except (ValueError, TypeError):
                    corrupted_rows.append(row[0])
                    self.logger.warning(f"Found corrupted avg_profit data in row {row[0]}")
            
            # Delete corrupted rows
            if corrupted_rows:
                placeholders = ','.join(['?'] * len(corrupted_rows))
                cursor.execute(f"DELETE FROM strategy_performance WHERE id IN ({placeholders})", corrupted_rows)
                self.logger.info(f"Cleaned {len(corrupted_rows)} corrupted database records")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to clean corrupted database: {e}")

    def _load_strategy_performance(self):
        """Load previous strategy performance data with corruption handling."""
        try:
            # First clean any corrupted data
            self.clean_corrupted_database()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM strategy_performance")
            results = cursor.fetchall()
            
            for row in results:
                strategy_mode = row[1]
                if strategy_mode not in self.strategy_performance:
                    self.strategy_performance[strategy_mode] = {}
                
                # Safe float conversion with error handling
                try:
                    avg_profit = float(row[6]) if row[6] is not None else 0.0
                except (ValueError, TypeError):
                    avg_profit = 0.0
                    self.logger.warning(f"Corrupted avg_profit for {strategy_mode}, reset to 0.0")
                
                self.strategy_performance[strategy_mode] = {
                    'wins': int(row[3] or 0),
                    'losses': int(row[4] or 0),
                    'total_pnl': float(row[5] or 0.0),
                    'avg_profit': avg_profit  # Use cleaned value
                }
            
            conn.close()
            self.logger.info(f"Loaded performance data for {len(self.strategy_performance)} strategies")
            
        except Exception as e:
            self.logger.warning(f"Could not load previous strategy performance: {e}")
            self.strategy_performance = {}
    
    def get_recent_data(self, symbol: str, timeframe: str, limit: int = 50) -> pd.DataFrame:
        """
        Get recent market data for analysis.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe interval
            limit: Number of recent candles to retrieve
            
        Returns:
            DataFrame with recent OHLCV data
        """
        table_name = f"{symbol.lower()}_{timeframe}"
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = f"""
                SELECT timestamp, open, high, low, close, volume 
                FROM {table_name} 
                ORDER BY timestamp DESC 
                LIMIT {limit}
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            # Convert timestamp and reverse order (oldest first)
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.iloc[::-1].reset_index(drop=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve data for {symbol} {timeframe}: {e}")
            return pd.DataFrame()
    
    def calculate_atr_volatility(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range volatility."""
        if len(data) < period + 1:
            return 0.0
        
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift(1))
        low_close = np.abs(data['low'] - data['close'].shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean().iloc[-1]
        
        # Normalize ATR as percentage of price
        current_price = data['close'].iloc[-1]
        return (atr / current_price) * 100 if current_price > 0 else 0.0
    
    def calculate_trend_strength(self, data: pd.DataFrame, period: int = 20) -> float:
        """Calculate trend strength using moving average slope."""
        if len(data) < period:
            return 0.0
        
        # Calculate moving average
        ma = data['close'].rolling(window=period).mean()
        
        # Calculate slope of moving average
        if len(ma) < 2:
            return 0.0
        
        # Simple linear regression slope
        y = ma.dropna().values
        x = np.arange(len(y))
        
        if len(x) < 2:
            return 0.0
        
        slope = np.polyfit(x, y, 1)[0]
        
        # Normalize slope relative to price
        current_price = data['close'].iloc[-1]
        normalized_slope = (slope / current_price) * 1000 if current_price > 0 else 0.0
        
        return abs(normalized_slope)
    
    def detect_ranging_market(self, data: pd.DataFrame, period: int = 20) -> bool:
        """Detect if market is range-bound."""
        if len(data) < period:
            return False
        
        recent_data = data.tail(period)
        high_price = recent_data['high'].max()
        low_price = recent_data['low'].min()
        
        # Calculate range as percentage
        price_range = ((high_price - low_price) / low_price) * 100
        
        # Consider ranging if price range is small
        return price_range < 8.0  # Less than 8% range
    
    def detect_breakout_setup(self, data: pd.DataFrame, period: int = 20) -> bool:
        """Detect potential breakout setup."""
        if len(data) < period + 5:
            return False
        
        recent_data = data.tail(period)
        latest_data = data.tail(5)
        
        # Check for consolidation followed by volume increase
        recent_range = (recent_data['high'].max() - recent_data['low'].min()) / recent_data['low'].min()
        recent_volume = recent_data['volume'].mean()
        latest_volume = latest_data['volume'].mean()
        
        # Breakout setup: tight range + volume increase
        tight_range = recent_range < 0.05  # Less than 5% range
        volume_increase = latest_volume > recent_volume * 1.3  # 30% volume increase
        
        return tight_range and volume_increase
    
    def detect_market_condition(self, symbol: str, timeframe: str, market_data: np.ndarray = None) -> str:
        """
        Detect current market condition using simple metrics.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe interval
            market_data: Optional market data array (for testing)
            
        Returns:
            Strategy mode for current market condition
        """
        if market_data is not None:
            # Use provided market data for testing
            # Convert numpy array to DataFrame format
            if len(market_data.shape) == 2 and market_data.shape[1] >= 5:
                df_data = {
                    'timestamp': range(len(market_data)),
                    'open': market_data[:, 1] if market_data.shape[1] >= 6 else market_data[:, 0],
                    'high': market_data[:, 2] if market_data.shape[1] >= 6 else market_data[:, 1],
                    'low': market_data[:, 3] if market_data.shape[1] >= 6 else market_data[:, 2],
                    'close': market_data[:, 4] if market_data.shape[1] >= 6 else market_data[:, 3],
                    'volume': market_data[:, 5] if market_data.shape[1] >= 6 else market_data[:, 4]
                }
                recent_data = pd.DataFrame(df_data)
            else:
                return StrategyModes.RANGING_MARKET
        else:
            # Get recent data for analysis from database
            recent_data = self.get_recent_data(symbol, timeframe, 50)
        
        if recent_data.empty:
            return StrategyModes.RANGING_MARKET
        
        # Calculate simple metrics
        volatility = self.calculate_atr_volatility(recent_data)
        trend_strength = self.calculate_trend_strength(recent_data)
        range_bound = self.detect_ranging_market(recent_data)
        breakout_setup = self.detect_breakout_setup(recent_data)
        
        # Simple classification logic
        if volatility > 4.0:
            return StrategyModes.VOLATILE_MARKET
        elif trend_strength > 0.7:
            return StrategyModes.TRENDING_MARKET
        elif range_bound:
            return StrategyModes.RANGING_MARKET
        elif breakout_setup:
            return StrategyModes.BREAKOUT_MODE
        else:
            return StrategyModes.REVERSAL_MODE
    
    def get_strategy_parameters(self, strategy_mode: str) -> Dict:
        """
        Get base parameters for specific strategy mode.
        
        Args:
            strategy_mode: Strategy mode to get parameters for
            
        Returns:
            Dictionary with strategy parameters
        """
        strategy_configs = {
            StrategyModes.VOLATILE_MARKET: {
                'position_size_multiplier': 0.6,     # Smaller positions
                'trailing_stop_factor': 1.5,         # Tighter trailing stops
                'exit_speed': 'FAST',                 # Quick exits
                'profit_target_multiplier': 0.8      # Lower profit targets
            },
            StrategyModes.TRENDING_MARKET: {
                'position_size_multiplier': 1.2,     # Larger positions
                'trailing_stop_factor': 0.7,         # Looser trailing stops
                'exit_speed': 'SLOW',                 # Hold longer
                'profit_target_multiplier': 1.5      # Higher profit targets
            },
            StrategyModes.RANGING_MARKET: {
                'position_size_multiplier': 1.0,     # Medium positions
                'trailing_stop_factor': 1.0,         # Normal trailing stops
                'exit_speed': 'MEDIUM',               # Medium holding time
                'profit_target_multiplier': 1.0      # Normal profit targets
            },
            StrategyModes.BREAKOUT_MODE: {
                'position_size_multiplier': 1.3,     # Large positions
                'trailing_stop_factor': 0.6,         # Very loose stops
                'exit_speed': 'VERY_SLOW',            # Hold for big moves
                'profit_target_multiplier': 2.0      # High profit targets
            },
            StrategyModes.REVERSAL_MODE: {
                'position_size_multiplier': 0.8,     # Smaller positions
                'trailing_stop_factor': 1.2,         # Tighter stops
                'exit_speed': 'FAST',                 # Quick scalps
                'profit_target_multiplier': 0.9      # Quick profits
            }
        }
        
        return strategy_configs.get(strategy_mode, strategy_configs[StrategyModes.RANGING_MARKET])
    
    def apply_emotional_influence(self, base_strategy_params: Dict) -> Dict:
        """
        Apply emotional influence to base strategy parameters.
        
        Args:
            base_strategy_params: Base parameters from strategy mode
            
        Returns:
            Modified parameters with emotional influence
        """
        # Confidence influences position sizing (0.5 to 1.5 multiplier)
        confidence_factor = 0.5 + (self.confidence / 10.0)  
        
        # Fear influences risk management (high fear = tighter stops)
        # HIGH FEAR = TIGHTER STOPS (smaller trailing distance)
        fear_factor = 0.5 + (self.fear / 10.0)  # Range: 0.5 (low fear) to 1.5 (high fear)
        
        # Greed influences profit targets and holding time
        greed_factor = 0.7 + (self.greed / 10.0)  # 0.7 to 1.7
        
        # Calculate base position size with emotional scaling
        base_position = 0.10  # 10% base position size
        emotional_position_size = (base_strategy_params['position_size_multiplier'] * 
                                  confidence_factor * base_position)
        
        # Apply emotional modifications
        modified_params = {
            'position_size': emotional_position_size,
            'trailing_stop': base_strategy_params['trailing_stop_factor'] * (1.0 + self.fear / 10.0) * 0.01,  # High fear = tighter stops
            'profit_target': base_strategy_params['profit_target_multiplier'] * greed_factor,
            'exit_speed': base_strategy_params['exit_speed']
        }
        
        return modified_params
    
    def enforce_safety_across_strategies(self, strategy_params: Dict) -> Dict:
        """
        Enforce hard safety limits that override any strategy.
        
        Args:
            strategy_params: Strategy parameters to enforce limits on
            
        Returns:
            Parameters with safety limits enforced
        """
        # Enforce maximum position size limit
        strategy_params['position_size'] = min(
            strategy_params['position_size'], 
            SafetyLimits.MAX_POSITION_SIZE
        )
        
        # Enforce minimum stop loss (don't let it be too tight)
        strategy_params['trailing_stop'] = max(
            strategy_params['trailing_stop'],
            0.005  # Minimum 0.5% stop loss
        )
        
        # Additional safety checks for daily drawdown
        if self.daily_pnl < -SafetyLimits.DAILY_DRAWDOWN_LIMIT:
            strategy_params['position_size'] *= 0.5  # Halve position size
            self.logger.warning("Daily drawdown limit reached - reducing position sizes")
        
        # Copy to max_ keys for compatibility
        strategy_params['max_position_size'] = strategy_params['position_size']
        strategy_params['max_stop_loss'] = strategy_params['trailing_stop']
        
        return strategy_params
    
    def track_strategy_performance(self, strategy_mode: str, trade_result: str, profit_loss: float):
        """
        Track performance of strategies for learning.
        
        Args:
            strategy_mode: Strategy that was used
            trade_result: 'WIN' or 'LOSS'
            profit_loss: Profit or loss amount
        """
        # Initialize strategy performance if not exists
        if strategy_mode not in self.strategy_performance:
            self.strategy_performance[strategy_mode] = {
                'wins': 0, 'losses': 0, 'total_pnl': 0.0, 'avg_profit': 0.0
            }
        
        # Update performance metrics
        if trade_result == 'WIN':
            self.strategy_performance[strategy_mode]['wins'] += 1
        else:
            self.strategy_performance[strategy_mode]['losses'] += 1
        
        self.strategy_performance[strategy_mode]['total_pnl'] += profit_loss
        
        # Calculate average profit
        total_trades = (self.strategy_performance[strategy_mode]['wins'] + 
                       self.strategy_performance[strategy_mode]['losses'])
        if total_trades > 0:
            self.strategy_performance[strategy_mode]['avg_profit'] = (
                self.strategy_performance[strategy_mode]['total_pnl'] / total_trades
            )
        
        # Update daily P&L
        self.daily_pnl += profit_loss
        
        # Save to database
        self.save_strategy_performance(strategy_mode)
        
        self.logger.info(f"Strategy {strategy_mode} performance updated: {trade_result}, P&L: {profit_loss:.2f}")
    
    def save_strategy_performance(self, strategy_mode: str = None):
        """Save strategy performance to database with proper error handling."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if strategy_mode:
                # Save specific strategy
                strategies_to_save = [strategy_mode]
            else:
                # Save all strategies
                strategies_to_save = list(self.strategy_performance.keys())
            
            for mode in strategies_to_save:
                if mode in self.strategy_performance:
                    perf = self.strategy_performance[mode]
                    
                    # Delete existing record for this strategy
                    cursor.execute("DELETE FROM strategy_performance WHERE strategy_mode = ?", (mode,))
                    
                    # Insert fresh record
                    cursor.execute("""
                        INSERT INTO strategy_performance 
                        (strategy_mode, market_condition, wins, losses, total_pnl, avg_profit, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        mode,
                        self.current_market_condition or 'general',
                        perf['wins'],
                        perf['losses'],
                        perf['total_pnl'],
                        perf['avg_profit'],
                        int(time.time())
                    ))
            
            conn.commit()
            conn.close()
            self.logger.info(f"Successfully saved performance data for {len(strategies_to_save)} strategies")
            
        except Exception as e:
            self.logger.error(f"Failed to save strategy performance: {e}")
            # Don't crash the system, just log the error
    
    def get_best_performing_strategy(self, market_condition: str) -> str:
        """
        Get the best performing strategy for current market condition.
        
        Args:
            market_condition: Current market condition
            
        Returns:
            Best performing strategy mode
        """
        if not self.strategy_performance:
            return StrategyModes.RANGING_MARKET
        
        best_strategy = StrategyModes.RANGING_MARKET
        best_score = -float('inf')
        
        for strategy_mode, performance in self.strategy_performance.items():
            total_trades = performance['wins'] + performance['losses']
            
            if total_trades >= 5:  # Minimum trades for consideration
                # Calculate performance score (win rate * average profit)
                win_rate = performance['wins'] / total_trades
                avg_profit = performance['avg_profit']
                
                # CRITICAL BUG FIX: Ensure avg_profit is a number
                try:
                    if avg_profit is None:
                        avg_profit = 0.0
                    elif isinstance(avg_profit, str):
                        avg_profit = float(avg_profit)
                    elif not isinstance(avg_profit, (int, float)):
                        avg_profit = 0.0
                    else:
                        avg_profit = float(avg_profit)
                except (ValueError, TypeError):
                    avg_profit = 0.0
                
                # Ensure win_rate is also a number
                try:
                    win_rate = float(win_rate)
                except (ValueError, TypeError):
                    win_rate = 0.0
                
                score = win_rate * avg_profit
                
                if score > best_score:
                    best_score = score
                    best_strategy = strategy_mode
        
        return best_strategy
    
    def select_best_strategy(self, market_condition: str) -> str:
        """
        Select best strategy considering market condition and emotions.
        
        Args:
            market_condition: Current market condition
            
        Returns:
            Selected strategy mode
        """
        # Get the best performing strategy for this market condition
        best_strategy = self.get_best_performing_strategy(market_condition)
        
        # If no performance data, use market condition as default
        if best_strategy == StrategyModes.RANGING_MARKET and market_condition != StrategyModes.RANGING_MARKET:
            best_strategy = market_condition
        
        # Emotional override (high fear = safer strategies)
        if self.fear > 8.0:
            if best_strategy in [StrategyModes.BREAKOUT_MODE, StrategyModes.TRENDING_MARKET]:
                best_strategy = StrategyModes.VOLATILE_MARKET  # Switch to safer
                self.logger.info("High fear detected - switching to safer strategy")
        
        # High confidence = more aggressive strategies allowed
        if self.confidence > 8.0:
            if best_strategy == StrategyModes.RANGING_MARKET:
                best_strategy = StrategyModes.BREAKOUT_MODE  # Switch to aggressive
                self.logger.info("High confidence detected - switching to aggressive strategy")
        
        return best_strategy
    
    def should_switch_strategy(self, new_market_condition: str) -> bool:
        """
        Decide if strategy should change based on market condition.
        
        Args:
            new_market_condition: New detected market condition
            
        Returns:
            True if strategy should switch
        """
        # Switch if market condition significantly changed
        if new_market_condition != self.current_market_condition:
            return True
        
        # Switch if current strategy is performing poorly
        if self.current_strategy in self.strategy_performance:
            perf = self.strategy_performance[self.current_strategy]
            total_trades = perf['wins'] + perf['losses']
            
            if total_trades >= 10:  # Enough data to judge
                win_rate = perf['wins'] / total_trades
                if win_rate < 0.4:  # Less than 40% win rate
                    self.logger.warning(f"Strategy {self.current_strategy} underperforming - considering switch")
                    return True
        
        return False
    
    def analyze_market_and_select_strategy(self, symbol: str, timeframe: str) -> Tuple[str, Dict]:
        """
        Analyze market conditions and select optimal strategy.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe interval
            
        Returns:
            Tuple of (strategy_mode, strategy_parameters)
        """
        # Detect current market condition
        market_condition = self.detect_market_condition(symbol, timeframe)
        
        # Check if strategy should switch
        if self.should_switch_strategy(market_condition):
            new_strategy = self.select_best_strategy(market_condition)
            
            if new_strategy != self.current_strategy:
                self.logger.info(f"Strategy switching: {self.current_strategy} -> {new_strategy}")
                self.current_strategy = new_strategy
        
        # Update current market condition
        self.current_market_condition = market_condition
        
        # Get base strategy parameters
        base_params = self.get_strategy_parameters(self.current_strategy)
        
        # Apply emotional influence
        emotional_params = self.apply_emotional_influence(base_params)
        
        # Enforce safety limits
        final_params = self.enforce_safety_across_strategies(emotional_params)
        
        # Save consciousness state
        self.save_consciousness_state()
        
        return self.current_strategy, final_params
    
    def save_consciousness_state(self):
        """Save current consciousness state to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO consciousness_state 
                (timestamp, confidence, fear, greed, current_strategy, market_condition, daily_pnl, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                int(time.time()), self.confidence, self.fear, self.greed,
                self.current_strategy, self.current_market_condition, self.daily_pnl, self.session_id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to save consciousness state: {e}")
    
    def update_emotions_and_strategy_performance(self, trade_result: str, profit_loss: float):
        """
        Update emotions and track strategy performance based on trade results.
        
        Args:
            trade_result: 'WIN' or 'LOSS'
            profit_loss: Profit or loss amount
        """
        # Track strategy performance
        self.track_strategy_performance(self.current_strategy, trade_result, profit_loss)
        
        # Update emotions based on trade result
        if trade_result == 'WIN':
            # Winning trades increase confidence and greed, reduce fear
            self.confidence = min(10.0, self.confidence + 0.5)
            self.greed = min(10.0, self.greed + 0.3)
            self.fear = max(0.0, self.fear - 0.2)
        else:
            # Losing trades reduce confidence and greed, increase fear
            self.confidence = max(0.0, self.confidence - 0.7)
            self.greed = max(0.0, self.greed - 0.4)
            self.fear = min(10.0, self.fear + 0.5)
        
        # Emotion bounds and normalization
        self.confidence = max(0.0, min(10.0, self.confidence))
        self.fear = max(0.0, min(10.0, self.fear))
        self.greed = max(0.0, min(10.0, self.greed))
        
        # Log emotional state changes
        self.logger.info(f"Emotions updated - Confidence: {self.confidence:.1f}, Fear: {self.fear:.1f}, Greed: {self.greed:.1f}")
        
        # Save state after emotion update
        self.save_consciousness_state()
    
    def get_dynamic_position_parameters(self, symbol: str, timeframe: str, market_data: np.ndarray = None) -> Dict:
        """
        Get dynamic position parameters based on current strategy and emotions.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe interval
            market_data: Optional market data array (for testing)
            
        Returns:
            Dictionary with position parameters
        """
        # Detect market condition (with optional data for testing)
        if market_data is not None:
            market_condition = self.detect_market_condition(symbol, timeframe, market_data)
        else:
            market_condition = self.detect_market_condition(symbol, timeframe)
        
        # Update current market condition and select strategy
        self.current_market_condition = market_condition
        if self.should_switch_strategy(market_condition):
            new_strategy = self.select_best_strategy(market_condition)
            if new_strategy != self.current_strategy:
                self.logger.info(f"Strategy switching: {self.current_strategy} -> {new_strategy}")
                self.current_strategy = new_strategy
        
        # Get base strategy parameters
        base_params = self.get_strategy_parameters(self.current_strategy)
        
        # Apply emotional influence
        emotional_params = self.apply_emotional_influence(base_params)
        
        # Enforce safety limits
        final_params = self.enforce_safety_across_strategies(emotional_params)
        
        # Save consciousness state
        self.save_consciousness_state()
        
        return {
            'strategy_mode': self.current_strategy,
            'market_condition': self.current_market_condition,
            'position_size': final_params['max_position_size'],
            'stop_loss': final_params['max_stop_loss'],
            'profit_target': final_params['profit_target'],
            'trailing_stop': final_params['trailing_stop'],
            'exit_speed': final_params['exit_speed'],
            'emotions': {
                'confidence': self.confidence,
                'fear': self.fear,
                'greed': self.greed
            }
        }
    
    def get_consciousness_summary(self) -> Dict:
        """Get comprehensive consciousness state summary."""
        return {
            'emotions': {
                'confidence': self.confidence,
                'fear': self.fear,
                'greed': self.greed
            },
            'current_strategy': self.current_strategy,
            'market_condition': self.current_market_condition,
            'daily_pnl': self.daily_pnl,
            'strategy_performance': self.strategy_performance,
            'session_id': self.session_id
        }
    
    def update_emotions_after_trade(self, result: str, pnl_percent: float, strategy_mode: str):
        """
        Update emotions after trade for learning (test compatibility method).
        
        Args:
            result: 'WIN' or 'LOSS'
            pnl_percent: P&L percentage
            strategy_mode: Strategy mode used
        """
        self.update_emotions_and_strategy_performance(result, pnl_percent)
    
    def record_strategy_performance(self, strategy_mode: str, result: str, pnl_percent: float):
        """
        Record strategy performance for learning (test compatibility method).
        
        Args:
            strategy_mode: Strategy mode used
            result: 'WIN' or 'LOSS'
            pnl_percent: P&L percentage
        """
        self.track_strategy_performance(strategy_mode, result, pnl_percent)
    
    def get_strategy_performance(self) -> List[Dict]:
        """
        Get strategy performance records (test compatibility method).
        
        Returns:
            List of performance records
        """
        performance_list = []
        for strategy_mode, perf_data in self.strategy_performance.items():
            performance_list.append({
                'strategy_mode': strategy_mode,
                'wins': perf_data['wins'],
                'losses': perf_data['losses'],
                'total_pnl': perf_data['total_pnl'],
                'avg_profit': perf_data['avg_profit']
            })
        return performance_list
    
    def select_optimal_strategy(self, symbol: str, timeframe: str) -> str:
        """
        Select optimal strategy based on market analysis and emotions.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe interval
            
        Returns:
            Optimal strategy mode
        """
        # Detect current market condition
        market_condition = self.detect_market_condition(symbol, timeframe)
        
        # Use existing comprehensive strategy selection logic
        selected_strategy, _ = self.analyze_market_and_select_strategy(symbol, timeframe)
        
        return selected_strategy
    
    def get_best_strategy_for_condition(self, market_condition: str) -> str:
        """
        Get best strategy for market condition (test compatibility method).
        
        Args:
            market_condition: Market condition
            
        Returns:
            Best strategy mode
        """
        return self.get_best_performing_strategy(market_condition)


def main():
    """Main function to demonstrate consciousness system."""
    print("🧠 JARVIS 3.0 - Adaptive Strategy Consciousness System")
    print("=" * 70)
    
    try:
        # Initialize consciousness
        consciousness = AdaptiveStrategyConsciousness()
        
        # Test symbols and timeframes
        test_symbols = ["BTCUSDT", "ETHUSDT"]
        test_timeframes = ["1h", "4h"]
        
        print("\n🎯 TESTING ADAPTIVE STRATEGY CONSCIOUSNESS")
        print("-" * 50)
        
        # Test strategy selection for different symbols/timeframes
        for symbol in test_symbols:
            for timeframe in test_timeframes:
                print(f"\n📊 Analyzing {symbol} {timeframe}:")
                
                # Get dynamic position parameters
                params = consciousness.get_dynamic_position_parameters(symbol, timeframe)
                
                print(f"   Strategy Mode: {params['strategy_mode']}")
                print(f"   Market Condition: {params['market_condition']}")
                print(f"   Position Size: {params['position_size']:.3f}")
                print(f"   Stop Loss: {params['stop_loss']:.3f}")
                print(f"   Profit Target: {params['profit_target']:.3f}")
                print(f"   Emotions - C:{params['emotions']['confidence']:.1f} F:{params['emotions']['fear']:.1f} G:{params['emotions']['greed']:.1f}")
        
        # Simulate some trade results to test learning
        print("\n🔄 SIMULATING TRADE RESULTS FOR LEARNING")
        print("-" * 50)
        
        # Simulate winning trade
        consciousness.update_emotions_and_strategy_performance('WIN', 1.5)
        print("✅ Simulated winning trade (+1.5%)")
        
        # Simulate losing trade
        consciousness.update_emotions_and_strategy_performance('LOSS', -0.8)
        print("❌ Simulated losing trade (-0.8%)")
        
        # Show final consciousness state
        print("\n🎭 FINAL CONSCIOUSNESS STATE")
        print("-" * 40)
        summary = consciousness.get_consciousness_summary()
        
        print(f"Confidence: {summary['emotions']['confidence']:.1f}/10.0")
        print(f"Fear: {summary['emotions']['fear']:.1f}/10.0")
        print(f"Greed: {summary['emotions']['greed']:.1f}/10.0")
        print(f"Current Strategy: {summary['current_strategy']}")
        print(f"Daily P&L: {summary['daily_pnl']:.2f}%")
        
        print("\n🎉 Adaptive Strategy Consciousness System Ready!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 