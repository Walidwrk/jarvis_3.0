#!/usr/bin/env python3
"""
JARVIS 3.0 - Intelligent Memory System
Advanced pattern learning and experience tracking for trading intelligence.

This module implements comprehensive memory capabilities:
- Pattern outcome tracking and learning
- Experience replay for neural network training  
- Long-term pattern analysis
- Historical performance integration
- Conflict pattern learning and intelligence

Author: JARVIS 3.0 Team
Version: 3.4 (Enhanced with Conflict Intelligence)
"""

import sqlite3
import pandas as pd
import numpy as np
import torch
import time
import json
import hashlib
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

class IntelligentMemorySystem:
    """
    Advanced memory system for trading pattern learning and experience tracking.
    Stores and analyzes trading patterns, outcomes, and experiences for intelligent decision making.
    """
    
    def __init__(self, db_path: str = "data/crypto_data.db"):
        """
        Initialize intelligent memory system.
        
        Args:
            db_path: Path to database file
        """
        self.db_path = db_path
        self.setup_logging()
        
        # Memory components
        self.pattern_outcomes = {}           # Pattern hash -> success rate
        self.experience_replay_buffer = []   # For neural network learning
        self.market_condition_memory = {}    # Condition -> best strategies
        self.long_term_patterns = {}         # Weekly/monthly pattern tracking
        self.position_outcome_history = []   # Track position management success
        
        # Performance tracking
        self.pattern_success_rates = {}
        self.strategy_effectiveness = {}
        self.time_based_performance = {}
        
        # Initialize database tables
        self.setup_memory_tables()
        self.load_existing_memory()
        
        # Initialize conflict learning system
        self.conflict_learning = ConflictLearningSystem(self.db_path)
        
        self.logger.info("Intelligent Memory System initialized with conflict learning")
    
    def setup_logging(self):
        """Setup logging for memory system."""
        self.logger = logging.getLogger('JARVIS_Memory')
        
    def setup_memory_tables(self):
        """Create memory tables in database for comprehensive tracking."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Pattern outcome tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pattern_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_hash TEXT,
                    market_condition TEXT,
                    action_taken TEXT,
                    outcome TEXT,
                    profit_loss REAL,
                    confidence_level REAL,
                    timestamp INTEGER,
                    symbol TEXT,
                    timeframe TEXT,
                    ta_features TEXT,
                    duration_minutes INTEGER
                )
            """)
            
            # Experience replay for neural network training
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experience_replay (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    state_vector TEXT,
                    action_taken INTEGER,
                    reward REAL,
                    next_state_vector TEXT,
                    done INTEGER,
                    timestamp INTEGER,
                    symbol TEXT,
                    confidence REAL,
                    market_condition TEXT
                )
            """)
            
            # Long-term pattern tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS long_term_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT,
                    pattern_data TEXT,
                    success_rate REAL,
                    total_occurrences INTEGER,
                    avg_profit REAL,
                    max_profit REAL,
                    max_loss REAL,
                    avg_duration REAL,
                    last_updated INTEGER,
                    symbol TEXT,
                    timeframe TEXT
                )
            """)
            
            # Market condition effectiveness tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_condition_effectiveness (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market_condition TEXT,
                    strategy_mode TEXT,
                    action_taken TEXT,
                    success_count INTEGER,
                    failure_count INTEGER,
                    avg_profit REAL,
                    last_updated INTEGER,
                    symbol TEXT
                )
            """)
            
            conn.commit()
            conn.close()
            self.logger.info("Memory database tables initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to setup memory tables: {e}")
            raise
    
    def load_existing_memory(self):
        """Load existing memory data from database into memory for fast access."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Load pattern success rates
            cursor = conn.cursor()
            cursor.execute("""
                SELECT pattern_hash, 
                       AVG(profit_loss) as avg_profit,
                       COUNT(*) as total_count,
                       SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as wins
                FROM pattern_outcomes 
                WHERE timestamp > ?
                GROUP BY pattern_hash
                HAVING COUNT(*) >= 2
            """, (int(time.time()) - 30*24*3600,))  # Last 30 days
            
            for row in cursor.fetchall():
                pattern_hash, avg_profit, total_count, wins = row
                win_rate = wins / total_count if total_count > 0 else 0
                
                self.pattern_success_rates[pattern_hash] = {
                    'avg_profit': avg_profit,
                    'win_rate': win_rate,
                    'sample_size': total_count,
                    'confidence': min(total_count / 10.0, 1.0),
                    'total_profit': avg_profit * total_count,  # Calculate total profit
                    'wins': wins
                }
            
            # Load market condition effectiveness
            cursor.execute("""
                SELECT market_condition, action_taken, 
                       AVG(profit_loss) as avg_profit,
                       COUNT(*) as total_count,
                       SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as wins
                FROM pattern_outcomes 
                WHERE timestamp > ?
                GROUP BY market_condition, action_taken
                HAVING COUNT(*) >= 3
            """, (int(time.time()) - 14*24*3600,))  # Last 14 days
            
            for row in cursor.fetchall():
                market_condition, action_taken, avg_profit, total_count, wins = row
                key = f"{market_condition}_{action_taken}"
                
                self.market_condition_memory[key] = {
                    'avg_profit': avg_profit,
                    'win_rate': wins / total_count,
                    'sample_size': total_count,
                    'total_profit': avg_profit * total_count,  # Calculate total profit
                    'wins': wins
                }
            
            conn.close()
            
            self.logger.info(f"Loaded {len(self.pattern_success_rates)} pattern memories and {len(self.market_condition_memory)} market condition memories")
            
        except Exception as e:
            self.logger.warning(f"Could not load existing memory data: {e}")
    
    def create_pattern_hash(self, ta_features: np.ndarray) -> str:
        """
        Create unique hash for technical analysis pattern for pattern matching.
        
        Args:
            ta_features: Technical analysis features array
            
        Returns:
            Pattern hash string
        """
        if len(ta_features) == 0:
            return "empty_pattern"
        
        # Discretize TA features for pattern matching
        discretized = []
        for feature in ta_features:
            if feature > 1.5:
                discretized.append('HIGH')
            elif feature < -1.5:
                discretized.append('LOW')
            elif feature > 0.5:
                discretized.append('MID_HIGH')
            elif feature < -0.5:
                discretized.append('MID_LOW')
            else:
                discretized.append('NEUTRAL')
        
        # Create hash from discretized pattern
        pattern_string = '_'.join(discretized)
        return hashlib.md5(pattern_string.encode()).hexdigest()[:12]
    
    def store_trading_experience(self, market_state: Dict, action_taken: str, 
                                outcome: str, profit_loss: float, symbol: str, 
                                timeframe: str, duration_minutes: int = 0) -> bool:
        """
        Store complete trading experience for intelligent learning.
        
        Args:
            market_state: Complete market state information
            action_taken: Action that was taken (BUY/SELL/HOLD)
            outcome: Result of the trade (WIN/LOSS/STOP_LOSS/TAKE_PROFIT)
            profit_loss: Profit or loss amount
            symbol: Trading symbol
            timeframe: Timeframe used
            duration_minutes: How long the position was held
            
        Returns:
            True if stored successfully
        """
        try:
            # Create pattern hash from technical analysis features
            ta_features = market_state.get('ta_features', np.array([]))
            pattern_hash = self.create_pattern_hash(ta_features)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Store pattern outcome
            cursor.execute("""
                INSERT INTO pattern_outcomes 
                (pattern_hash, market_condition, action_taken, outcome, profit_loss, 
                 confidence_level, timestamp, symbol, timeframe, ta_features, duration_minutes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pattern_hash,
                market_state.get('market_condition', 'unknown'),
                action_taken,
                outcome,
                profit_loss,
                market_state.get('confidence', 0.5),
                int(time.time()),
                symbol,
                timeframe,
                json.dumps(ta_features.tolist() if hasattr(ta_features, 'tolist') else list(ta_features)),
                duration_minutes
            ))
            
            # Store for neural network experience replay
            state_vector = market_state.get('state_vector', torch.zeros(76))
            next_state_vector = market_state.get('next_state_vector')
            
            # Convert action to integer
            action_int = {'BUY': 0, 'SELL': 1, 'HOLD': 2}.get(action_taken, 2)
            
            state_json = json.dumps(state_vector.tolist() if hasattr(state_vector, 'tolist') else list(state_vector))
            next_state_json = json.dumps(
                next_state_vector.tolist() if next_state_vector is not None and hasattr(next_state_vector, 'tolist') 
                else []
            )
            
            cursor.execute("""
                INSERT INTO experience_replay 
                (state_vector, action_taken, reward, next_state_vector, done, timestamp, symbol, confidence, market_condition)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                state_json,
                action_int,
                profit_loss,
                next_state_json,
                1 if outcome in ['STOP_LOSS', 'TAKE_PROFIT', 'CLOSE'] else 0,
                int(time.time()),
                symbol,
                market_state.get('confidence', 0.5),
                market_state.get('market_condition', 'unknown')
            ))
            
            conn.commit()
            conn.close()
            
            # Update in-memory pattern tracking
            self.update_pattern_success_rate(pattern_hash, outcome, profit_loss)
            self.update_market_condition_memory(market_state.get('market_condition'), action_taken, profit_loss)
            
            self.logger.debug(f"Stored trading experience: {action_taken} -> {outcome} ({profit_loss:.2f})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store trading experience: {e}")
            return False
    
    def update_pattern_success_rate(self, pattern_hash: str, outcome: str, profit_loss: float):
        """Update in-memory pattern success rate tracking."""
        if pattern_hash not in self.pattern_success_rates:
            self.pattern_success_rates[pattern_hash] = {
                'avg_profit': 0.0,
                'win_rate': 0.5,
                'sample_size': 0,
                'confidence': 0.0,
                'total_profit': 0.0,
                'wins': 0
            }
        
        pattern = self.pattern_success_rates[pattern_hash]
        pattern['sample_size'] += 1
        pattern['total_profit'] += profit_loss
        pattern['avg_profit'] = pattern['total_profit'] / pattern['sample_size']
        
        if profit_loss > 0:
            pattern['wins'] += 1
        
        pattern['win_rate'] = pattern['wins'] / pattern['sample_size']
        pattern['confidence'] = min(pattern['sample_size'] / 10.0, 1.0)
    
    def update_market_condition_memory(self, market_condition: str, action_taken: str, profit_loss: float):
        """Update market condition effectiveness memory."""
        if not market_condition or not action_taken:
            return
            
        key = f"{market_condition}_{action_taken}"
        
        if key not in self.market_condition_memory:
            self.market_condition_memory[key] = {
                'avg_profit': 0.0,
                'win_rate': 0.5,
                'sample_size': 0,
                'total_profit': 0.0,
                'wins': 0
            }
        
        memory = self.market_condition_memory[key]
        memory['sample_size'] += 1
        memory['total_profit'] += profit_loss
        memory['avg_profit'] = memory['total_profit'] / memory['sample_size']
        
        if profit_loss > 0:
            memory['wins'] += 1
        
        memory['win_rate'] = memory['wins'] / memory['sample_size']
    
    def get_pattern_success_rate(self, current_pattern_hash: str) -> Dict:
        """
        Get historical success rate for similar patterns.
        
        Args:
            current_pattern_hash: Hash of current market pattern
            
        Returns:
            Dictionary with success rate information
        """
        # Check in-memory first
        if current_pattern_hash in self.pattern_success_rates:
            return self.pattern_success_rates[current_pattern_hash].copy()
        
        # Query database for this specific pattern
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT AVG(profit_loss), COUNT(*), 
                       SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as wins,
                       MAX(profit_loss) as max_profit,
                       MIN(profit_loss) as max_loss
                FROM pattern_outcomes 
                WHERE pattern_hash = ? AND timestamp > ?
            """, (current_pattern_hash, int(time.time()) - 30*24*3600))  # Last 30 days
            
            result = cursor.fetchone()
            conn.close()
            
            if result and result[1] >= 3:  # At least 3 occurrences
                avg_profit, total_count, wins, max_profit, max_loss = result
                win_rate = wins / total_count
                
                pattern_data = {
                    'avg_profit': avg_profit,
                    'win_rate': win_rate,
                    'sample_size': total_count,
                    'confidence': min(total_count / 10.0, 1.0),
                    'max_profit': max_profit,
                    'max_loss': max_loss,
                    'total_profit': avg_profit * total_count,  # Calculate total profit
                    'wins': wins
                }
                
                # Cache for future use
                self.pattern_success_rates[current_pattern_hash] = pattern_data
                return pattern_data
            
        except Exception as e:
            self.logger.error(f"Error querying pattern success rate: {e}")
        
        # Default for unknown patterns
        return {
            'avg_profit': 0, 
            'win_rate': 0.5, 
            'sample_size': 0, 
            'confidence': 0,
            'max_profit': 0,
            'max_loss': 0
        }
    
    def get_market_condition_effectiveness(self, market_condition: str, action: str) -> Dict:
        """
        Get effectiveness of specific action in specific market condition.
        
        Args:
            market_condition: Current market condition
            action: Proposed action
            
        Returns:
            Effectiveness data for this combination
        """
        key = f"{market_condition}_{action}"
        
        if key in self.market_condition_memory:
            return self.market_condition_memory[key].copy()
        
        return {
            'avg_profit': 0,
            'win_rate': 0.5,
            'sample_size': 0
        }
    
    def get_experience_replay_batch(self, batch_size: int = 32) -> Optional[List[Tuple]]:
        """
        Get batch of experiences for neural network training.
        
        Args:
            batch_size: Number of experiences to return
            
        Returns:
            List of experience tuples or None if insufficient data
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT state_vector, action_taken, reward, next_state_vector, done
                FROM experience_replay 
                WHERE timestamp > ? AND next_state_vector != '[]'
                ORDER BY timestamp DESC
                LIMIT ?
            """, (int(time.time()) - 7*24*3600, batch_size * 10))  # Last 7 days
            
            experiences = cursor.fetchall()
            conn.close()
            
            if len(experiences) < batch_size:
                return None  # Not enough data yet
            
            # Sample random batch
            batch = random.sample(experiences, batch_size)
            
            # Convert to proper format
            formatted_batch = []
            for exp in batch:
                try:
                    state = json.loads(exp[0])
                    next_state = json.loads(exp[3]) if exp[3] else []
                    
                    if len(state) > 0 and len(next_state) > 0:
                        formatted_batch.append((
                            torch.tensor(state, dtype=torch.float32),
                            exp[1],  # action
                            exp[2],  # reward
                            torch.tensor(next_state, dtype=torch.float32),
                            exp[4]   # done
                        ))
                except (json.JSONDecodeError, ValueError):
                    continue
            
            return formatted_batch if len(formatted_batch) >= batch_size // 2 else None
            
        except Exception as e:
            self.logger.error(f"Error getting experience replay batch: {e}")
            return None
    
    def analyze_long_term_patterns(self, symbol: str, timeframe: str) -> Dict:
        """
        Analyze long-term patterns for better strategic decisions.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe to analyze
            
        Returns:
            Long-term pattern analysis
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Analyze patterns by time of day
            cursor.execute("""
                SELECT 
                    CAST(strftime('%H', datetime(timestamp, 'unixepoch')) AS INTEGER) as hour,
                    AVG(profit_loss) as avg_profit,
                    COUNT(*) as trade_count,
                    SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as wins
                FROM pattern_outcomes 
                WHERE symbol = ? AND timeframe = ? AND timestamp > ?
                GROUP BY hour
                HAVING COUNT(*) >= 2
                ORDER BY avg_profit DESC
            """, (symbol, timeframe, int(time.time()) - 30*24*3600))
            
            hourly_performance = {}
            for row in cursor.fetchall():
                hour, avg_profit, trade_count, wins = row
                hourly_performance[hour] = {
                    'avg_profit': avg_profit,
                    'win_rate': wins / trade_count,
                    'trade_count': trade_count
                }
            
            # Analyze patterns by day of week
            cursor.execute("""
                SELECT 
                    CAST(strftime('%w', datetime(timestamp, 'unixepoch')) AS INTEGER) as day_of_week,
                    AVG(profit_loss) as avg_profit,
                    COUNT(*) as trade_count,
                    SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as wins
                FROM pattern_outcomes 
                WHERE symbol = ? AND timeframe = ? AND timestamp > ?
                GROUP BY day_of_week
                HAVING COUNT(*) >= 3
                ORDER BY avg_profit DESC
            """, (symbol, timeframe, int(time.time()) - 30*24*3600))
            
            daily_performance = {}
            day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
            
            for row in cursor.fetchall():
                day_num, avg_profit, trade_count, wins = row
                daily_performance[day_names[day_num]] = {
                    'avg_profit': avg_profit,
                    'win_rate': wins / trade_count,
                    'trade_count': trade_count
                }
            
            conn.close()
            
            return {
                'hourly_performance': hourly_performance,
                'daily_performance': daily_performance,
                'best_hour': max(hourly_performance.keys(), key=lambda x: hourly_performance[x]['avg_profit']) if hourly_performance else None,
                'best_day': max(daily_performance.keys(), key=lambda x: daily_performance[x]['avg_profit']) if daily_performance else None
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing long-term patterns: {e}")
            return {'hourly_performance': {}, 'daily_performance': {}, 'best_hour': None, 'best_day': None}
    
    def get_memory_summary(self) -> Dict:
        """Get comprehensive summary of memory system state."""
        return {
            'pattern_memories': len(self.pattern_success_rates),
            'market_condition_memories': len(self.market_condition_memory),
            'high_confidence_patterns': len([p for p in self.pattern_success_rates.values() if p['confidence'] > 0.7]),
            'successful_patterns': len([p for p in self.pattern_success_rates.values() if p['win_rate'] > 0.6]),
            'memory_database': self.db_path
        }
    
    def cleanup_old_memories(self, days_to_keep: int = 60):
        """Clean up old memory data to maintain performance."""
        try:
            cutoff_timestamp = int(time.time()) - days_to_keep * 24 * 3600
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Clean old pattern outcomes
            cursor.execute("DELETE FROM pattern_outcomes WHERE timestamp < ?", (cutoff_timestamp,))
            pattern_deleted = cursor.rowcount
            
            # Clean old experience replay
            cursor.execute("DELETE FROM experience_replay WHERE timestamp < ?", (cutoff_timestamp,))
            experience_deleted = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Cleaned up {pattern_deleted} old pattern outcomes and {experience_deleted} old experiences")
            
            # Reload memory after cleanup
            self.load_existing_memory()
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old memories: {e}")
    
    def create_enhanced_pattern_hash(self, ta_features: np.ndarray, market_context: Dict) -> str:
        """Enhanced pattern hash that includes market context."""
        if len(ta_features) == 0:
            return "empty_pattern"
        
        # Discretize TA features with more granularity
        discretized = []
        for feature in ta_features:
            if feature > 2.0:
                discretized.append('VERY_HIGH')
            elif feature > 1.0:
                discretized.append('HIGH')
            elif feature > 0.3:
                discretized.append('MID_HIGH')
            elif feature > -0.3:
                discretized.append('NEUTRAL')
            elif feature > -1.0:
                discretized.append('MID_LOW')
            elif feature > -2.0:
                discretized.append('LOW')
            else:
                discretized.append('VERY_LOW')
        
        # Add market context
        volatility = market_context.get('volatility', 'MEDIUM')
        trend_strength = market_context.get('trend_strength', 'WEAK')
        
        # Include volatility and trend in pattern
        discretized.extend([volatility, trend_strength])
        
        # Create hash from enhanced pattern
        pattern_string = '_'.join(discretized)
        return hashlib.md5(pattern_string.encode()).hexdigest()[:16]  # Longer hash

    def get_pattern_confidence_interval(self, pattern_hash: str) -> Dict:
        """Calculate confidence intervals for pattern predictions."""
        if pattern_hash not in self.pattern_success_rates:
            return {'lower': 0.4, 'upper': 0.6, 'confidence': 0.0}
        
        pattern_data = self.pattern_success_rates[pattern_hash]
        sample_size = pattern_data['sample_size']
        win_rate = pattern_data['win_rate']
        
        if sample_size < 5:
            return {'lower': 0.4, 'upper': 0.6, 'confidence': 0.1}
        
        # Calculate 95% confidence interval using normal approximation
        z_score = 1.96  # 95% confidence
        standard_error = np.sqrt((win_rate * (1 - win_rate)) / sample_size)
        
        lower_bound = max(0, win_rate - z_score * standard_error)
        upper_bound = min(1, win_rate + z_score * standard_error)
        
        # Confidence based on sample size and interval width
        interval_width = upper_bound - lower_bound
        confidence = min(sample_size / 20.0, 1.0) * (1 - interval_width)
        
        return {
            'lower': lower_bound,
            'upper': upper_bound,
            'confidence': confidence,
            'sample_size': sample_size
        }
    
    def store_conflict_aware_experience(self, market_state: Dict, action_taken: str, 
                                      outcome: str, profit_loss: float, symbol: str, 
                                      timeframe: str, duration_minutes: int = 0) -> bool:
        """Enhanced experience storage with conflict pattern learning."""
        # Store normal experience
        success = self.store_trading_experience(market_state, action_taken, outcome, 
                                              profit_loss, symbol, timeframe, duration_minutes)
        
        # Additionally store conflict pattern if conflicts exist
        tf_signals = market_state.get('timeframe_signals', {})
        confluence_score = market_state.get('confluence_score', 0.5)
        conflicts = market_state.get('conflicts', [])
        
        if len(conflicts) > 0 and tf_signals and hasattr(self, 'conflict_learning'):
            action_int = {'BUY': 0, 'SELL': 1, 'HOLD': 2}.get(action_taken, 2)
            self.conflict_learning.record_conflict_trade(
                tf_signals, confluence_score, conflicts, action_int, outcome, profit_loss
            )
        
        return success


class ConflictLearningSystem:
    """
    Advanced system to learn which timeframe conflict patterns are profitable.
    Instead of penalizing all conflicts, learns which ones are trading opportunities.
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conflict_patterns = {}  # Pattern hash -> profitability data
        
        # Set up logging first
        self.logger = logging.getLogger('JARVIS_ConflictLearning')
        
        self.setup_conflict_tables()
        self.load_conflict_patterns()
        
        # Pattern profitability thresholds
        self.high_profit_threshold = 0.65  # 65% win rate
        self.avoid_threshold = 0.35        # Below 35% win rate
    
    def setup_conflict_tables(self):
        """Create tables for conflict pattern learning."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conflict_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_hash TEXT UNIQUE,
                    pattern_description TEXT,
                    total_trades INTEGER,
                    winning_trades INTEGER,
                    avg_profit REAL,
                    max_profit REAL,
                    min_loss REAL,
                    confidence_score REAL,
                    last_updated INTEGER
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conflict_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_hash TEXT,
                    timeframe_signals TEXT,
                    action_taken INTEGER,
                    outcome TEXT,
                    profit_loss REAL,
                    timestamp INTEGER,
                    confluence_score REAL,
                    conflict_count INTEGER
                )
            """)
            
            conn.commit()
            conn.close()
            self.logger.info("Conflict learning tables initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to setup conflict tables: {e}")
    
    def encode_conflict_pattern(self, tf_signals: Dict, confluence_score: float) -> str:
        """
        Encode timeframe conflict pattern into a learnable hash.
        """
        pattern_parts = []
        
        # Encode each timeframe direction
        tf_order = ['1d', '4h', '1h', '15m', '5m', '1m']
        for tf in tf_order:
            if tf in tf_signals:
                direction = tf_signals[tf].get('direction', 'NEUTRAL')
                pattern_parts.append(f"{tf}_{direction.lower()}")
            else:
                pattern_parts.append(f"{tf}_missing")
        
        # Add confluence level
        if confluence_score > 0.7:
            confluence_level = "high_confluence"
        elif confluence_score > 0.4:
            confluence_level = "medium_confluence"
        else:
            confluence_level = "low_confluence"
        
        pattern_parts.append(confluence_level)
        
        # Create hash from pattern
        pattern_string = "_".join(pattern_parts)
        pattern_hash = hashlib.md5(pattern_string.encode()).hexdigest()[:12]
        
        return pattern_hash
    
    def record_conflict_trade(self, tf_signals: Dict, confluence_score: float, 
                            conflicts: List, action: int, outcome: str, profit_loss: float):
        """Record a trade with conflict pattern for learning."""
        try:
            pattern_hash = self.encode_conflict_pattern(tf_signals, confluence_score)
            
            # Store trade record
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO conflict_trades 
                (pattern_hash, timeframe_signals, action_taken, outcome, profit_loss, 
                 timestamp, confluence_score, conflict_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pattern_hash,
                json.dumps(tf_signals),
                action,
                outcome,
                profit_loss,
                int(time.time()),
                confluence_score,
                len(conflicts)
            ))
            
            conn.commit()
            conn.close()
            
            # Update pattern statistics
            self.update_conflict_pattern_stats(pattern_hash, outcome, profit_loss)
            
        except Exception as e:
            self.logger.error(f"Failed to record conflict trade: {e}")
    
    def update_conflict_pattern_stats(self, pattern_hash: str, outcome: str, profit_loss: float):
        """Update statistical data for conflict pattern."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get current stats
            cursor.execute("SELECT * FROM conflict_patterns WHERE pattern_hash = ?", (pattern_hash,))
            current = cursor.fetchone()
            
            if current:
                # Update existing pattern
                total_trades = current[3] + 1
                winning_trades = current[4] + (1 if outcome == 'WIN' else 0)
                
                # Update profit statistics
                current_total_profit = current[5] * current[3]  # avg_profit * total_trades
                new_total_profit = current_total_profit + profit_loss
                new_avg_profit = new_total_profit / total_trades
                
                new_max_profit = max(current[6], profit_loss)
                new_min_loss = min(current[7], profit_loss)
                
                # Calculate confidence
                win_rate = winning_trades / total_trades
                sample_confidence = min(total_trades / 20.0, 1.0)
                consistency_score = 1.0 - abs(win_rate - 0.5)
                confidence_score = sample_confidence * consistency_score
                
                cursor.execute("""
                    UPDATE conflict_patterns 
                    SET total_trades = ?, winning_trades = ?, avg_profit = ?, 
                        max_profit = ?, min_loss = ?, confidence_score = ?, last_updated = ?
                    WHERE pattern_hash = ?
                """, (total_trades, winning_trades, new_avg_profit, new_max_profit, 
                     new_min_loss, confidence_score, int(time.time()), pattern_hash))
            
            else:
                # Create new pattern
                winning_trades = 1 if outcome == 'WIN' else 0
                confidence_score = 0.1
                
                cursor.execute("""
                    INSERT INTO conflict_patterns 
                    (pattern_hash, pattern_description, total_trades, winning_trades, 
                     avg_profit, max_profit, min_loss, confidence_score, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (pattern_hash, "Auto-generated conflict pattern", 1, winning_trades,
                     profit_loss, profit_loss, profit_loss, confidence_score, int(time.time())))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to update conflict pattern stats: {e}")
    
    def get_conflict_intelligence(self, tf_signals: Dict, confluence_score: float) -> Dict:
        """Get intelligent analysis of current conflict pattern."""
        pattern_hash = self.encode_conflict_pattern(tf_signals, confluence_score)
        
        if pattern_hash in self.conflict_patterns:
            pattern_data = self.conflict_patterns[pattern_hash]
            
            total_trades = pattern_data['total_trades']
            win_rate = pattern_data['win_rate']
            avg_profit = pattern_data['avg_profit']
            confidence_score = pattern_data['confidence_score']
            
            # Determine intelligence level
            if total_trades >= 10 and confidence_score > 0.7:
                if win_rate >= self.high_profit_threshold:
                    intelligence_level = "HIGH_PROFIT_PATTERN"
                    confidence_multiplier = 1.5
                    recommendation = f"Historical {win_rate:.1%} win rate - STRONG signal"
                elif win_rate <= self.avoid_threshold:
                    intelligence_level = "AVOID_PATTERN"
                    confidence_multiplier = 0.3
                    recommendation = f"Historical {win_rate:.1%} win rate - AVOID trading"
                else:
                    intelligence_level = "NEUTRAL_PATTERN"
                    confidence_multiplier = 1.0
                    recommendation = f"Historical {win_rate:.1%} win rate - Standard confidence"
            else:
                intelligence_level = "LEARNING_PATTERN"
                confidence_multiplier = 0.8
                recommendation = f"Learning pattern ({total_trades} trades) - Reduced confidence"
            
            return {
                'pattern_hash': pattern_hash,
                'intelligence_level': intelligence_level,
                'historical_win_rate': win_rate,
                'historical_avg_profit': avg_profit,
                'confidence_multiplier': confidence_multiplier,
                'recommendation': recommendation,
                'sample_size': total_trades,
                'pattern_confidence': confidence_score
            }
        
        else:
            # Unknown pattern
            return {
                'pattern_hash': pattern_hash,
                'intelligence_level': "UNKNOWN_PATTERN",
                'historical_win_rate': 0.5,
                'historical_avg_profit': 0.0,
                'confidence_multiplier': 0.7,
                'recommendation': "Unknown conflict pattern - Proceed with caution",
                'sample_size': 0,
                'pattern_confidence': 0.0
            }
    
    def load_conflict_patterns(self):
        """Load existing conflict patterns into memory."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM conflict_patterns")
            for row in cursor.fetchall():
                pattern_hash = row[1]
                self.conflict_patterns[pattern_hash] = {
                    'total_trades': row[3],
                    'winning_trades': row[4],
                    'win_rate': row[4] / row[3] if row[3] > 0 else 0.5,
                    'avg_profit': row[5],
                    'max_profit': row[6],
                    'min_loss': row[7],
                    'confidence_score': row[8]
                }
            
            conn.close()
            self.logger.info(f"Loaded {len(self.conflict_patterns)} conflict patterns")
            
        except Exception as e:
            self.logger.warning(f"Could not load conflict patterns: {e}")
            self.conflict_patterns = {}


def main():
    """Test intelligent memory system."""
    print("🧠 JARVIS 3.0 - Intelligent Memory System Test")
    print("=" * 60)
    
    try:
        # Initialize memory system
        memory = IntelligentMemorySystem()
        
        # Test conflict intelligence
        tf_signals = {'1d': {'direction': 'BULLISH'}, '1h': {'direction': 'BEARISH'}}
        conflict_intel = memory.conflict_learning.get_conflict_intelligence(tf_signals, 0.3)
        print(f"✅ Conflict intelligence: {conflict_intel['intelligence_level']}")
        
        # Test pattern creation
        test_ta_features = np.array([2.1, -1.8, 0.3, 1.7, -0.2, 0.8, -1.9, 2.3, 0.1, -0.7, 1.2, -1.1])
        pattern_hash = memory.create_pattern_hash(test_ta_features)
        print(f"✅ Created pattern hash: {pattern_hash}")
        
        # Test storing experience
        market_state = {
            'ta_features': test_ta_features,
            'market_condition': 'trending_market',
            'confidence': 0.75,
            'state_vector': torch.randn(76)
        }
        
        success = memory.store_trading_experience(
            market_state, 'BUY', 'WIN', 2.35, 'BTCUSDT', '1h', 120
        )
        print(f"✅ Stored experience: {success}")
        
        # Test pattern retrieval
        pattern_data = memory.get_pattern_success_rate(pattern_hash)
        print(f"✅ Pattern data: Win rate {pattern_data['win_rate']:.2f}, Sample size {pattern_data['sample_size']}")
        
        # Test market condition effectiveness
        effectiveness = memory.get_market_condition_effectiveness('trending_market', 'BUY')
        print(f"✅ Market condition effectiveness: {effectiveness}")
        
        # Test memory summary
        summary = memory.get_memory_summary()
        print(f"✅ Memory summary: {summary}")
        
        print("\n🎉 Intelligent Memory System: OPERATIONAL")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 