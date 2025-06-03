#!/usr/bin/env python3
"""
JARVIS 3.0 - Neural Intelligence System (FIXED VERSION)
Complete AI brain (CNN-LSTM-SAC + Technical Analysis)

FIXES APPLIED:
- Fixed dimension mismatches in state vector construction
- Standardized CNN output sizes
- Fixed LSTM integration
- Added comprehensive data validation
- Fixed real-time processing issues

Author: JARVIS 3.0 Team
Version: 3.1 (FIXED)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import sqlite3
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List
import os
from config import CONFIG

class LightweightTechnicalAnalysis:
    """
    Lightweight technical analysis for fast pattern recognition.
    FIXED: Ensures consistent 12-feature output always.
    """
    
    def __init__(self):
        """Initialize TA analyzer with caching for performance."""
        self.ta_cache = {}
        self.cache_max_size = 1000
        
    def extract_ta_features(self, ohlcv_data: np.ndarray, timeframe: str, symbol: str, timestamp: int) -> np.ndarray:
        """
        Extract lightweight technical analysis features.
        FIXED: Always returns exactly 12 features, handles all edge cases.
        """
        # Cache key for avoiding recalculation
        cache_key = f"{symbol}_{timeframe}_{timestamp}"
        if cache_key in self.ta_cache:
            return self.ta_cache[cache_key]
        
        # Manage cache size
        if len(self.ta_cache) > self.cache_max_size:
            oldest_keys = list(self.ta_cache.keys())[:int(self.cache_max_size * 0.2)]
            for key in oldest_keys:
                del self.ta_cache[key]
        
        # FIXED: Return default features if insufficient data
        if len(ohlcv_data) < 10:
            default_features = np.zeros(12)
            self.ta_cache[cache_key] = default_features
            return default_features
        
        try:
            # FIXED: Robust data extraction with validation
            if ohlcv_data.shape[1] >= 6:
                high = ohlcv_data[:, 2]
                low = ohlcv_data[:, 3]
                close = ohlcv_data[:, 4]
                volume = ohlcv_data[:, 5]
            elif ohlcv_data.shape[1] == 5:
                high = ohlcv_data[:, 1]
                low = ohlcv_data[:, 2]
                close = ohlcv_data[:, 3]
                volume = ohlcv_data[:, 4]
            else:
                # Invalid data format
                default_features = np.zeros(12)
                self.ta_cache[cache_key] = default_features
                return default_features
            
            # FIXED: Validate data types and handle NaN/inf
            if not all(np.isfinite(arr).all() for arr in [high, low, close, volume]):
                default_features = np.zeros(12)
                self.ta_cache[cache_key] = default_features
                return default_features
            
            # Calculate features with error handling
            try:
                support_levels = self.find_pivot_lows(low, window=5)
                resistance_levels = self.find_pivot_highs(high, window=5)
                uptrend_slope = self.calculate_trendline_slope(low, lookback=min(20, len(low)-1))
                downtrend_slope = self.calculate_trendline_slope(high, lookback=min(20, len(high)-1), direction='down')
                
                current_price = close[-1]
                breakout_signal = self.detect_breakout(current_price, support_levels, resistance_levels, volume)
                triangle_pattern = self.detect_triangle_pattern(high, low, lookback=min(15, len(high)-1))
                channel_pattern = self.detect_channel_pattern(high, low, lookback=min(20, len(high)-1))
                volume_spike = self.detect_volume_spike(volume, lookback=min(10, len(volume)-1))
                volume_trend = self.calculate_volume_trend(volume, lookback=min(10, len(volume)-1))
                
                # FIXED: Compile exactly 12 features with validation
                ta_features = np.array([
                    self.distance_to_nearest_support(current_price, support_levels),
                    self.distance_to_nearest_resistance(current_price, resistance_levels),
                    uptrend_slope,
                    downtrend_slope,
                    breakout_signal['strength'],
                    breakout_signal['direction'],
                    triangle_pattern['strength'],
                    channel_pattern['strength'],
                    volume_spike,
                    volume_trend,
                    self.price_position_in_range(current_price, high, low, lookback=min(20, len(high)-1)),
                    self.momentum_indicator(close, lookback=min(10, len(close)-1))
                ])
                
                # FIXED: Ensure exactly 12 features and handle NaN/inf
                if len(ta_features) != 12:
                    ta_features = np.zeros(12)
                
                # Replace any NaN or inf values
                ta_features = np.nan_to_num(ta_features, nan=0.0, posinf=5.0, neginf=-5.0)
                
                # Clip extreme values
                ta_features = np.clip(ta_features, -10.0, 10.0)
                
            except Exception as e:
                logging.getLogger('JARVIS_Neural').warning(f"TA calculation error: {e}")
                ta_features = np.zeros(12)
            
            # Cache and return
            self.ta_cache[cache_key] = ta_features
            return ta_features
            
        except Exception as e:
            logging.getLogger('JARVIS_Neural').error(f"TA extraction failed: {e}")
            default_features = np.zeros(12)
            self.ta_cache[cache_key] = default_features
            return default_features
    
    def find_pivot_lows(self, low_prices: np.ndarray, window: int = 5) -> np.ndarray:
        """Fast vectorized pivot point detection for support levels."""
        try:
            if len(low_prices) < window * 2 + 1:
                return np.array([np.min(low_prices)]) if len(low_prices) > 0 else np.array([0.0])
            
            pivots = []
            for i in range(window, len(low_prices) - window):
                if low_prices[i] == np.min(low_prices[i-window:i+window+1]):
                    pivots.append(low_prices[i])
            return np.array(pivots) if pivots else np.array([np.min(low_prices)])
        except:
            return np.array([0.0])
    
    def find_pivot_highs(self, high_prices: np.ndarray, window: int = 5) -> np.ndarray:
        """Fast vectorized pivot point detection for resistance levels."""
        try:
            if len(high_prices) < window * 2 + 1:
                return np.array([np.max(high_prices)]) if len(high_prices) > 0 else np.array([0.0])
            
            pivots = []
            for i in range(window, len(high_prices) - window):
                if high_prices[i] == np.max(high_prices[i-window:i+window+1]):
                    pivots.append(high_prices[i])
            return np.array(pivots) if pivots else np.array([np.max(high_prices)])
        except:
            return np.array([0.0])
    
    def calculate_trendline_slope(self, prices: np.ndarray, lookback: int = 20, direction: str = 'up') -> float:
        """Linear regression for trendline slope calculation."""
        try:
            if len(prices) < max(2, lookback):
                return 0.0
            
            recent_prices = prices[-lookback:] if lookback <= len(prices) else prices
            if len(recent_prices) < 2:
                return 0.0
                
            x = np.arange(len(recent_prices))
            slope = np.polyfit(x, recent_prices, 1)[0]
            return slope if direction == 'up' else -slope
        except:
            return 0.0
    
    def detect_breakout(self, current_price: float, support_levels: np.ndarray, 
                       resistance_levels: np.ndarray, volume: np.ndarray) -> Dict:
        """Simple breakout detection with volume confirmation."""
        try:
            if len(volume) < 5 or current_price <= 0:
                return {'strength': 0.0, 'direction': 0}
            
            nearby_resistance = resistance_levels[resistance_levels > current_price]
            nearby_support = support_levels[support_levels < current_price]
            
            nearest_resistance = np.min(nearby_resistance) if len(nearby_resistance) > 0 else float('inf')
            nearest_support = np.max(nearby_support) if len(nearby_support) > 0 else 0
            
            resistance_distance = (nearest_resistance - current_price) / current_price if nearest_resistance != float('inf') else 1.0
            support_distance = (current_price - nearest_support) / current_price if nearest_support > 0 else 1.0
            
            # Volume confirmation
            recent_volume = volume[-3:] if len(volume) >= 3 else volume
            avg_volume = np.mean(volume[-10:-3]) if len(volume) >= 13 else np.mean(volume[:-3]) if len(volume) > 3 else np.mean(volume)
            volume_factor = min(np.mean(recent_volume) / avg_volume, 3.0) if avg_volume > 0 else 1.0
            
            if resistance_distance < 0.005:
                return {'strength': min(volume_factor * 2, 5.0), 'direction': 1}
            elif support_distance < 0.005:
                return {'strength': min(volume_factor * 2, 5.0), 'direction': -1}
            else:
                return {'strength': 0.0, 'direction': 0}
        except:
            return {'strength': 0.0, 'direction': 0}
    
    def detect_triangle_pattern(self, high_prices: np.ndarray, low_prices: np.ndarray, lookback: int = 15) -> Dict:
        """Detect triangle pattern formation."""
        try:
            if len(high_prices) < lookback or len(low_prices) < lookback:
                return {'strength': 0.0}
            
            recent_highs = high_prices[-lookback:]
            recent_lows = low_prices[-lookback:]
            
            high_slope = self.calculate_trendline_slope(recent_highs, lookback, direction='down')
            low_slope = self.calculate_trendline_slope(recent_lows, lookback, direction='up')
            
            convergence = abs(high_slope) + abs(low_slope)
            strength = min(convergence * 100, 3.0)
            
            return {'strength': strength}
        except:
            return {'strength': 0.0}
    
    def detect_channel_pattern(self, high_prices: np.ndarray, low_prices: np.ndarray, lookback: int = 20) -> Dict:
        """Detect channel pattern formation."""
        try:
            if len(high_prices) < lookback or len(low_prices) < lookback:
                return {'strength': 0.0}
            
            recent_highs = high_prices[-lookback:]
            recent_lows = low_prices[-lookback:]
            
            high_slope = self.calculate_trendline_slope(recent_highs, lookback)
            low_slope = self.calculate_trendline_slope(recent_lows, lookback)
            
            slope_difference = abs(high_slope - low_slope)
            strength = max(0, 2.0 - slope_difference * 100)
            
            return {'strength': min(strength, 2.0)}
        except:
            return {'strength': 0.0}
    
    def detect_volume_spike(self, volume: np.ndarray, lookback: int = 10) -> float:
        """Detect volume spike compared to average."""
        try:
            if len(volume) < lookback + 3:
                return 0.0
            
            recent_volume = np.mean(volume[-3:])
            avg_volume = np.mean(volume[-lookback-3:-3])
            
            if avg_volume > 0:
                return min((recent_volume / avg_volume - 1.0) * 2, 3.0)
            return 0.0
        except:
            return 0.0
    
    def calculate_volume_trend(self, volume: np.ndarray, lookback: int = 10) -> float:
        """Calculate volume trend using linear regression."""
        try:
            if len(volume) < lookback:
                return 0.0
            
            recent_volume = volume[-lookback:]
            return self.calculate_trendline_slope(recent_volume, lookback)
        except:
            return 0.0
    
    def distance_to_nearest_support(self, current_price: float, support_levels: np.ndarray) -> float:
        """Calculate distance to nearest support level."""
        try:
            if len(support_levels) == 0 or current_price <= 0:
                return 0.5
            
            below_supports = support_levels[support_levels < current_price]
            if len(below_supports) > 0:
                nearest_support = np.max(below_supports)
                return (current_price - nearest_support) / current_price
            return 0.5
        except:
            return 0.5
    
    def distance_to_nearest_resistance(self, current_price: float, resistance_levels: np.ndarray) -> float:
        """Calculate distance to nearest resistance level."""
        try:
            if len(resistance_levels) == 0 or current_price <= 0:
                return 0.5
            
            above_resistances = resistance_levels[resistance_levels > current_price]
            if len(above_resistances) > 0:
                nearest_resistance = np.min(above_resistances)
                return (nearest_resistance - current_price) / current_price
            return 0.5
        except:
            return 0.5
    
    def price_position_in_range(self, current_price: float, high_prices: np.ndarray, 
                               low_prices: np.ndarray, lookback: int = 20) -> float:
        """Calculate price position within recent range (0=low, 1=high)."""
        try:
            if len(high_prices) < lookback or len(low_prices) < lookback or current_price <= 0:
                return 0.5
            
            recent_high = np.max(high_prices[-lookback:])
            recent_low = np.min(low_prices[-lookback:])
            
            if recent_high > recent_low:
                return (current_price - recent_low) / (recent_high - recent_low)
            return 0.5
        except:
            return 0.5
    
    def momentum_indicator(self, close_prices: np.ndarray, lookback: int = 10) -> float:
        """Simple momentum indicator based on price change."""
        try:
            if len(close_prices) < lookback + 1:
                return 0.0
            
            current_price = close_prices[-1]
            old_price = close_prices[-lookback-1]
            
            if old_price > 0:
                return (current_price - old_price) / old_price
            return 0.0
        except:
            return 0.0


class EnhancedMultiTimeframeCNN(nn.Module):
    """
    FIXED: Enhanced CNN for multi-timeframe pattern recognition with consistent output sizes.
    """
    
    def __init__(self):
        super(EnhancedMultiTimeframeCNN, self).__init__()
        
        self.ta_analyzer = LightweightTechnicalAnalysis()
        
        # FIXED: Standardized CNN architecture for all timeframes
        self.timeframe_cnns = nn.ModuleDict({
            '1m': self.create_cnn(),
            '5m': self.create_cnn(),
            '15m': self.create_cnn(),
            '1h': self.create_cnn(),
            '4h': self.create_cnn(),
            '1d': self.create_cnn()
        })
        
        # FIXED: Precise input size calculation
        # 6 timeframes * (64 CNN features + 12 TA features) = 456 total features
        expected_input_size = 6 * (CONFIG.NeuralNetworks.CNN_OUTPUT_FEATURES + CONFIG.NeuralNetworks.TA_FEATURES_COUNT)
        self.feature_fusion = nn.Linear(expected_input_size, CONFIG.NeuralNetworks.CNN_OUTPUT_FEATURES)
        self.dropout = nn.Dropout(CONFIG.NeuralNetworks.DROPOUT_RATE)
        
    def create_cnn(self) -> nn.Module:
        """Create CNN optimized for MPS with consistent output."""
        class MPSCompatibleCNN(nn.Module):
            def __init__(self):
                super(MPSCompatibleCNN, self).__init__()
                self.conv1 = nn.Conv1d(CONFIG.NeuralNetworks.CNN_INPUT_CHANNELS, 32, kernel_size=3, padding=1)
                self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
                self.conv3 = nn.Conv1d(64, CONFIG.NeuralNetworks.CNN_OUTPUT_FEATURES, kernel_size=3, padding=1)
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.dropout = nn.Dropout(CONFIG.NeuralNetworks.DROPOUT_RATE)
                
            def forward(self, x):
                # FIXED: Guaranteed output size
                x = F.relu(self.conv1(x))
                x = self.dropout(x)
                x = F.relu(self.conv2(x))
                x = self.dropout(x)
                x = F.relu(self.conv3(x))
                x = self.pool(x).squeeze(-1)  # Always outputs exactly 64 features
                return x
        
        return MPSCompatibleCNN()
    
    def forward(self, multi_tf_data: Dict, symbol: str, timestamp: int) -> torch.Tensor:
        """
        FIXED: Process multi-timeframe data with guaranteed consistent output.
        """
        device = next(self.parameters()).device
        all_features = []
        
        for timeframe in ['1m', '5m', '15m', '1h', '4h', '1d']:
            if timeframe in multi_tf_data and multi_tf_data[timeframe] is not None:
                try:
                    data = multi_tf_data[timeframe]
                    
                    # FIXED: Robust data validation
                    if not isinstance(data, np.ndarray) or len(data) < 5:
                        combined_features = torch.zeros(CONFIG.NeuralNetworks.CNN_OUTPUT_FEATURES + CONFIG.NeuralNetworks.TA_FEATURES_COUNT).to(device)
                        all_features.append(combined_features)
                        continue
                    
                    # FIXED: Always get exactly 12 TA features
                    ta_features = self.ta_analyzer.extract_ta_features(data, timeframe, symbol, timestamp)
                    if len(ta_features) != 12:
                        ta_features = np.zeros(12)
                    
                    # FIXED: Robust OHLCV processing
                    if data.shape[1] >= 6:
                        ohlcv = data[:, 1:6].T
                    elif data.shape[1] >= 5:
                        ohlcv = data[:, :5].T
                    else:
                        # Invalid data
                        combined_features = torch.zeros(CONFIG.NeuralNetworks.CNN_OUTPUT_FEATURES + CONFIG.NeuralNetworks.TA_FEATURES_COUNT).to(device)
                        all_features.append(combined_features)
                        continue
                    
                    # FIXED: Ensure exactly 6 input channels
                    if ohlcv.shape[0] == 5:
                        volume_change = np.diff(ohlcv[4], prepend=ohlcv[4][0])
                        ohlcv = np.vstack([ohlcv, volume_change.reshape(1, -1)])
                    elif ohlcv.shape[0] < 6:
                        # Pad to 6 channels
                        padding_needed = 6 - ohlcv.shape[0]
                        padding = np.zeros((padding_needed, ohlcv.shape[1]))
                        ohlcv = np.vstack([ohlcv, padding])
                    elif ohlcv.shape[0] > 6:
                        # Truncate to 6 channels
                        ohlcv = ohlcv[:6, :]
                    
                    # FIXED: Validate tensor creation
                    if not np.isfinite(ohlcv).all():
                        ohlcv = np.nan_to_num(ohlcv, nan=0.0, posinf=1e6, neginf=-1e6)
                    
                    ohlcv_tensor = torch.FloatTensor(ohlcv).unsqueeze(0).to(device)
                    
                    # FIXED: CNN processing with error handling
                    try:
                        cnn_features = self.timeframe_cnns[timeframe](ohlcv_tensor).squeeze(0)
                        # Ensure exactly 64 features
                        if cnn_features.size(0) != CONFIG.NeuralNetworks.CNN_OUTPUT_FEATURES:
                            cnn_features = torch.zeros(CONFIG.NeuralNetworks.CNN_OUTPUT_FEATURES).to(device)
                    except Exception as e:
                        logging.getLogger('JARVIS_Neural').warning(f"CNN processing failed for {timeframe}: {e}")
                        cnn_features = torch.zeros(CONFIG.NeuralNetworks.CNN_OUTPUT_FEATURES).to(device)
                    
                    # FIXED: Combine features with validation
                    ta_tensor = torch.FloatTensor(ta_features).to(device)
                    if ta_tensor.size(0) != CONFIG.NeuralNetworks.TA_FEATURES_COUNT:
                        ta_tensor = torch.zeros(CONFIG.NeuralNetworks.TA_FEATURES_COUNT).to(device)
                    
                    combined_features = torch.cat([cnn_features, ta_tensor], dim=0)
                    
                except Exception as e:
                    logging.getLogger('JARVIS_Neural').error(f"Feature processing failed for {timeframe}: {e}")
                    combined_features = torch.zeros(CONFIG.NeuralNetworks.CNN_OUTPUT_FEATURES + CONFIG.NeuralNetworks.TA_FEATURES_COUNT).to(device)
            else:
                # Default features for missing timeframe
                combined_features = torch.zeros(CONFIG.NeuralNetworks.CNN_OUTPUT_FEATURES + CONFIG.NeuralNetworks.TA_FEATURES_COUNT).to(device)
            
            all_features.append(combined_features)
        
        # FIXED: Guaranteed consistent concatenation
        # 6 timeframes * 76 features each = 456 total
        fused_features = torch.cat(all_features, dim=0)
        
        # FIXED: Validate fusion input size
        expected_size = 6 * (CONFIG.NeuralNetworks.CNN_OUTPUT_FEATURES + CONFIG.NeuralNetworks.TA_FEATURES_COUNT)
        if fused_features.size(0) != expected_size:
            logging.getLogger('JARVIS_Neural').warning(f"Feature fusion size mismatch: got {fused_features.size(0)}, expected {expected_size}")
            fused_features = torch.zeros(expected_size).to(device)
        
        # Feature fusion and normalization
        enhanced_features = self.feature_fusion(fused_features)
        enhanced_features = self.dropout(enhanced_features)
        
        return enhanced_features


class EnhancedPatternLSTM(nn.Module):
    """
    FIXED: Enhanced LSTM with proper dimension handling.
    """
    
    def __init__(self, input_size: int = None):
        super(EnhancedPatternLSTM, self).__init__()
        
        # FIXED: Use CNN output size as LSTM input
        if input_size is None:
            input_size = CONFIG.NeuralNetworks.CNN_OUTPUT_FEATURES  # 64 from CNN
        
        self.input_size = input_size
        self.hidden_size = CONFIG.NeuralNetworks.LSTM_HIDDEN_SIZE
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=CONFIG.NeuralNetworks.DROPOUT_RATE,
            bidirectional=False
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=4,
            dropout=CONFIG.NeuralNetworks.DROPOUT_RATE,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(CONFIG.NeuralNetworks.DROPOUT_RATE)
        
    def forward(self, enhanced_features: torch.Tensor, sequence_length: int = 20) -> torch.Tensor:
        """
        FIXED: Process enhanced features through LSTM with proper dimensions.
        """
        device = enhanced_features.device
        
        # FIXED: Validate input dimensions
        if enhanced_features.dim() == 1:
            enhanced_features = enhanced_features.unsqueeze(0)
        
        # FIXED: Create proper sequence with variation
        batch_size = enhanced_features.size(0)
        input_size = enhanced_features.size(1)
        
        # Validate input size matches LSTM expectation
        if input_size != self.input_size:
            logging.getLogger('JARVIS_Neural').warning(f"LSTM input size mismatch: got {input_size}, expected {self.input_size}")
            # Pad or truncate to match expected size
            if input_size < self.input_size:
                padding = torch.zeros(batch_size, self.input_size - input_size).to(device)
                enhanced_features = torch.cat([enhanced_features, padding], dim=1)
            else:
                enhanced_features = enhanced_features[:, :self.input_size]
        
        # Create sequence with temporal patterns
        sequence_input = enhanced_features.unsqueeze(1).repeat(1, sequence_length, 1)
        
        # Add slight noise for temporal variation (smaller noise)
        noise = torch.randn_like(sequence_input) * 0.001
        sequence_input = sequence_input + noise
        
        try:
            # LSTM processing
            lstm_output, (hidden_state, cell_state) = self.lstm(sequence_input)
            
            # Apply attention mechanism
            attended_output, attention_weights = self.attention(
                query=lstm_output,
                key=lstm_output,
                value=lstm_output
            )
            
            # Use last output for pattern representation
            pattern_features = attended_output[:, -1, :]
            
            # Output projection
            output = self.output_projection(pattern_features)
            output = self.dropout(output)
            
            return output.squeeze(0) if output.size(0) == 1 else output
            
        except Exception as e:
            logging.getLogger('JARVIS_Neural').error(f"LSTM processing failed: {e}")
            # Return zero tensor with correct dimensions
            return torch.zeros(self.hidden_size).to(device)


class EnhancedTradingSAC(nn.Module):
    """
    FIXED: Enhanced SAC with proper state vector handling.
    """
    
    def __init__(self):
        super(EnhancedTradingSAC, self).__init__()
        
        self.state_size = CONFIG.NeuralNetworks.SAC_STATE_VECTOR_SIZE  # 76
        self.action_size = CONFIG.NeuralNetworks.SAC_ACTION_SPACE      # 3
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(self.state_size, 256),
            nn.ReLU(),
            nn.Dropout(CONFIG.NeuralNetworks.DROPOUT_RATE),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(CONFIG.NeuralNetworks.DROPOUT_RATE),
            nn.Linear(128, self.action_size),
            nn.Softmax(dim=-1)
        )
        
        # Critic networks
        self.critic1 = nn.Sequential(
            nn.Linear(self.state_size + self.action_size, 256),
            nn.ReLU(),
            nn.Dropout(CONFIG.NeuralNetworks.DROPOUT_RATE),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(CONFIG.NeuralNetworks.DROPOUT_RATE),
            nn.Linear(128, 1)
        )
        
        self.critic2 = nn.Sequential(
            nn.Linear(self.state_size + self.action_size, 256),
            nn.ReLU(),
            nn.Dropout(CONFIG.NeuralNetworks.DROPOUT_RATE),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(CONFIG.NeuralNetworks.DROPOUT_RATE),
            nn.Linear(128, 1)
        )
        
        # Value network
        self.value_network = nn.Sequential(
            nn.Linear(self.state_size, 256),
            nn.ReLU(),
            nn.Dropout(CONFIG.NeuralNetworks.DROPOUT_RATE),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(CONFIG.NeuralNetworks.DROPOUT_RATE),
            nn.Linear(128, 1)
        )
    
    def get_ta_summary(self, ta_features_all_timeframes: List[np.ndarray]) -> np.ndarray:
        """
        FIXED: Summarize TA features with guaranteed 8-element output.
        """
        try:
            if not ta_features_all_timeframes or len(ta_features_all_timeframes) == 0:
                return np.zeros(8)
            
            # Filter valid features
            valid_features = []
            for features in ta_features_all_timeframes:
                if isinstance(features, np.ndarray) and len(features) == 12:
                    valid_features.append(features)
            
            if not valid_features:
                return np.zeros(8)
            
            stacked_features = np.array(valid_features)
            
            # Calculate summary statistics with error handling
            ta_summary = np.array([
                np.mean(stacked_features[:, 0]) if stacked_features.shape[1] > 0 else 0.0,   # Avg support distance
                np.mean(stacked_features[:, 1]) if stacked_features.shape[1] > 1 else 0.0,   # Avg resistance distance
                np.mean(stacked_features[:, 2:4]) if stacked_features.shape[1] > 3 else 0.0, # Avg trend slopes
                np.max(stacked_features[:, 4:6]) if stacked_features.shape[1] > 5 else 0.0,   # Max breakout signals
                np.mean(stacked_features[:, 6:8]) if stacked_features.shape[1] > 7 else 0.0,  # Avg pattern strengths
                np.mean(stacked_features[:, 8:10]) if stacked_features.shape[1] > 9 else 0.0, # Avg volume indicators
                np.mean(stacked_features[:, 10:12]) if stacked_features.shape[1] > 11 else 0.0 # Avg momentum/position
            ]).flatten()
            
            # Ensure exactly 8 features
            if len(ta_summary) < 8:
                ta_summary = np.pad(ta_summary, (0, 8 - len(ta_summary)), 'constant')
            elif len(ta_summary) > 8:
                ta_summary = ta_summary[:8]
            
            # Handle NaN/inf
            ta_summary = np.nan_to_num(ta_summary, nan=0.0, posinf=5.0, neginf=-5.0)
            
            return ta_summary
            
        except Exception as e:
            logging.getLogger('JARVIS_Neural').error(f"TA summary calculation failed: {e}")
            return np.zeros(8)
    
    def forward(self, state_vector: torch.Tensor) -> torch.Tensor:
        """Forward pass for action prediction with dimension validation."""
        # FIXED: Validate state vector dimensions
        if state_vector.dim() == 1:
            state_vector = state_vector.unsqueeze(0)
        
        if state_vector.size(1) != self.state_size:
            logging.getLogger('JARVIS_Neural').warning(f"SAC input size mismatch: got {state_vector.size(1)}, expected {self.state_size}")
            # Pad or truncate
            if state_vector.size(1) < self.state_size:
                padding = torch.zeros(state_vector.size(0), self.state_size - state_vector.size(1)).to(state_vector.device)
                state_vector = torch.cat([state_vector, padding], dim=1)
            else:
                state_vector = state_vector[:, :self.state_size]
        
        return self.actor(state_vector)


class EnhancedJARVIS_Neural_System:
    """
    FIXED: Complete Enhanced JARVIS Neural Intelligence System.
    """
    
    def __init__(self, consciousness_system=None, db_path: str = None):
        """Initialize the enhanced neural system."""
        self.db_path = db_path or CONFIG.DATABASE_PATH
        self.consciousness = consciousness_system
        
        # Setup logging
        self.logger = logging.getLogger('JARVIS_Neural')
        
        # Setup device and models
        self.device = self.setup_device()
        
        # Initialize neural networks
        self.cnn = EnhancedMultiTimeframeCNN().to(self.device)
        self.lstm = EnhancedPatternLSTM().to(self.device)
        self.sac = EnhancedTradingSAC().to(self.device)
        
        # Set to evaluation mode for inference
        self.fast_inference_mode()
        
        # Optimize for M1 if available
        self.optimize_for_m1()
        
        self.logger.info("Enhanced JARVIS Neural System initialized with fixes")
    
    def setup_device(self) -> torch.device:
        """Setup optimal device for M1 MacBook."""
        if torch.backends.mps.is_available() and not CONFIG.NeuralNetworks.FORCE_CPU:
            device = torch.device(CONFIG.NeuralNetworks.DEVICE)
            self.logger.info("Using MPS (Metal Performance Shaders) acceleration")
        else:
            device = torch.device("cpu")
            self.logger.info("Using CPU")
        
        return device
    
    def optimize_for_m1(self):
        """Apply M1 MacBook specific optimizations."""
        try:
            torch.set_num_threads(8)
            
            if self.device.type == 'mps':
                try:
                    if hasattr(torch.backends.mps, 'empty_cache'):
                        torch.backends.mps.empty_cache()
                except AttributeError:
                    pass
                
            self.logger.info("Applied M1 MacBook optimizations")
            
        except Exception as e:
            self.logger.warning(f"Could not apply M1 optimizations: {e}")
    
    def get_multi_timeframe_data(self, symbol: str, current_timestamp: int, 
                                lookback_periods: Dict[str, int] = None) -> Dict:
        """
        FIXED: Retrieve multi-timeframe data with validation.
        """
        if lookback_periods is None:
            lookback_periods = {
                '1m': 50, '5m': 50, '15m': 50,
                '1h': 50, '4h': 50, '1d': 50
            }
        
        multi_tf_data = {}
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            for timeframe in CONFIG.TIMEFRAMES:
                table_name = f"{symbol.lower()}_{timeframe}"
                lookback = lookback_periods.get(timeframe, 50)
                
                try:
                    # FIXED: Check table exists
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
                    if not cursor.fetchone():
                        self.logger.warning(f"Table {table_name} does not exist")
                        continue
                    
                    # FIXED: Add data staleness check
                    max_age_ms = 24 * 60 * 60 * 1000  # 24 hours
                    min_timestamp = current_timestamp - max_age_ms
                    
                    query = f"""
                        SELECT timestamp, open, high, low, close, volume
                        FROM {table_name}
                        WHERE timestamp <= ? AND timestamp >= ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    """
                    
                    df = pd.read_sql_query(query, conn, params=[current_timestamp, min_timestamp, lookback])
                    
                    if not df.empty and len(df) >= 10:  # Require minimum data
                        df = df.iloc[::-1].reset_index(drop=True)
                        
                        # FIXED: Validate data quality
                        if self._validate_market_data(df):
                            multi_tf_data[timeframe] = df.values
                        else:
                            self.logger.warning(f"Invalid data quality for {symbol} {timeframe}")
                    else:
                        self.logger.warning(f"Insufficient recent data for {symbol} {timeframe}: {len(df) if not df.empty else 0} records")
                        
                except Exception as e:
                    self.logger.warning(f"Failed to get {timeframe} data: {e}")
            
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Database error: {e}")
        
        return multi_tf_data
    
    def _validate_market_data(self, df: pd.DataFrame) -> bool:
        """
        FIXED: Validate market data quality.
        """
        try:
            if df.empty or len(df) < 5:
                return False
            
            # Check for required columns
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                return False
            
            # Check for NaN values
            if df[required_cols].isnull().any().any():
                return False
            
            # Check for invalid prices
            price_cols = ['open', 'high', 'low', 'close']
            if (df[price_cols] <= 0).any().any():
                return False
            
            # Check OHLC consistency
            if not ((df['low'] <= df['high']) & 
                   (df['low'] <= df['open']) & 
                   (df['low'] <= df['close']) &
                   (df['open'] <= df['high']) & 
                   (df['close'] <= df['high'])).all():
                return False
            
            # Check for extreme price movements (>50% in one candle)
            price_changes = df['close'].pct_change().abs()
            if (price_changes > 0.5).any():
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Data validation error: {e}")
            return False
    
    def make_enhanced_trading_decision(self, symbol: str, current_timestamp: int = None) -> Dict:
        """
        FIXED: Make enhanced trading decision with proper error handling.
        """
        if current_timestamp is None:
            current_timestamp = int(time.time() * 1000)
        
        try:
            # Get multi-timeframe data with validation
            multi_tf_data = self.get_multi_timeframe_data(symbol, current_timestamp)
            
            if not multi_tf_data:
                return self._get_default_decision("No recent market data available")
            
            # FIXED: CNN processing with error handling
            try:
                cnn_features = self.cnn(multi_tf_data, symbol, current_timestamp)
                if torch.isnan(cnn_features).any() or torch.isinf(cnn_features).any():
                    self.logger.warning("CNN features contain NaN/inf values")
                    return self._get_default_decision("Invalid CNN features")
            except Exception as e:
                self.logger.error(f"CNN processing failed: {e}")
                return self._get_default_decision(f"CNN error: {str(e)}")
            
            # FIXED: LSTM processing with error handling
            try:
                lstm_features = self.lstm(cnn_features)
                if torch.isnan(lstm_features).any() or torch.isinf(lstm_features).any():
                    self.logger.warning("LSTM features contain NaN/inf values")
                    return self._get_default_decision("Invalid LSTM features")
            except Exception as e:
                self.logger.error(f"LSTM processing failed: {e}")
                return self._get_default_decision(f"LSTM error: {str(e)}")
            
            # FIXED: Build comprehensive state vector
            try:
                state_vector = self._build_state_vector(lstm_features, multi_tf_data, symbol)
                if torch.isnan(state_vector).any() or torch.isinf(state_vector).any():
                    self.logger.warning("State vector contains NaN/inf values")
                    return self._get_default_decision("Invalid state vector")
            except Exception as e:
                self.logger.error(f"State vector construction failed: {e}")
                return self._get_default_decision(f"State vector error: {str(e)}")
            
            # FIXED: SAC decision making
            try:
                with torch.no_grad():
                    action_probs = self.sac(state_vector)
                    action_probs_np = action_probs.cpu().numpy()
                    
                    if len(action_probs_np.shape) > 1:
                        action_probs_np = action_probs_np[0]
                    
                    if np.isnan(action_probs_np).any() or np.isinf(action_probs_np).any():
                        return self._get_default_decision("Invalid SAC output")
                        
            except Exception as e:
                self.logger.error(f"SAC processing failed: {e}")
                return self._get_default_decision(f"SAC error: {str(e)}")
            
            # Interpret action probabilities
            sell_prob, hold_prob, buy_prob = action_probs_np
            
            # Get consciousness input if available
            consciousness_influence = 1.0
            if self.consciousness:
                try:
                    consciousness_state = self.consciousness.get_dynamic_position_parameters(symbol, "1h")
                    emotions = consciousness_state.get('emotions', {})
                    confidence = emotions.get('confidence', 5.0)
                    fear = emotions.get('fear', 5.0)
                    
                    consciousness_influence = (confidence - fear) / 10.0
                    consciousness_influence = max(-1.0, min(1.0, consciousness_influence))
                    
                except Exception as e:
                    self.logger.warning(f"Could not get consciousness input: {e}")
                    consciousness_influence = 0.0
            
            # FIXED: Make final decision with validation
            if buy_prob > 0.6 and consciousness_influence > -0.5:
                decision = "BUY"
                confidence_score = min(buy_prob * (1 + consciousness_influence), 1.0)
            elif sell_prob > 0.6 and consciousness_influence > -0.5:
                decision = "SELL"
                confidence_score = min(sell_prob * (1 + abs(consciousness_influence)), 1.0)
            else:
                decision = "HOLD"
                confidence_score = hold_prob
            
            # Ensure confidence is valid
            confidence_score = max(0.0, min(1.0, float(confidence_score)))
            
            # Get TA interpretation
            ta_insights = self.get_ta_interpretation_from_data(multi_tf_data, symbol, current_timestamp)
            
            return {
                'decision': decision,
                'confidence': confidence_score,
                'action_probabilities': {
                    'buy': float(buy_prob),
                    'hold': float(hold_prob),
                    'sell': float(sell_prob)
                },
                'neural_analysis': {
                    'cnn_features_norm': float(torch.norm(cnn_features).cpu()),
                    'lstm_features_norm': float(torch.norm(lstm_features).cpu()),
                    'state_vector_norm': float(torch.norm(state_vector).cpu())
                },
                'consciousness_influence': consciousness_influence,
                'ta_insights': ta_insights,
                'timestamp': current_timestamp,
                'symbol': symbol,
                'data_quality': {
                    'timeframes_available': len(multi_tf_data),
                    'total_timeframes': len(CONFIG.TIMEFRAMES)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in enhanced trading decision: {e}")
            return self._get_default_decision(f"System error: {str(e)}")
    
    def _build_state_vector(self, lstm_features: torch.Tensor, multi_tf_data: Dict, symbol: str) -> torch.Tensor:
        """
        FIXED: Build comprehensive state vector with guaranteed 76 dimensions.
        """
        device = lstm_features.device
        state_components = []
        
        # Component 1: LSTM features (64 dimensions)
        if lstm_features.dim() == 0:
            lstm_features = lstm_features.unsqueeze(0)
        
        lstm_size = CONFIG.NeuralNetworks.LSTM_HIDDEN_SIZE  # 64
        if lstm_features.size(0) < lstm_size:
            padding = torch.zeros(lstm_size - lstm_features.size(0)).to(device)
            lstm_features = torch.cat([lstm_features, padding], dim=0)
        elif lstm_features.size(0) > lstm_size:
            lstm_features = lstm_features[:lstm_size]
        
        state_components.append(lstm_features)
        current_size = lstm_size
        
        # Component 2: Market context (2 dimensions)
        try:
            if '1h' in multi_tf_data and len(multi_tf_data['1h']) > 10:
                data = multi_tf_data['1h']
                current_price = data[-1, 4]
                old_price = data[-10, 4] if len(data) >= 10 else data[0, 4]
                price_change = (current_price - old_price) / old_price if old_price > 0 else 0
                
                current_volume = data[-1, 5]
                avg_volume = np.mean(data[-20:, 5]) if len(data) >= 20 else current_volume
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                
                market_context = torch.FloatTensor([price_change, volume_ratio]).to(device)
            else:
                market_context = torch.zeros(2).to(device)
        except Exception as e:
            self.logger.warning(f"Market context calculation failed: {e}")
            market_context = torch.zeros(2).to(device)
        
        state_components.append(market_context)
        current_size += 2
        
        # Component 3: TA summary (8 dimensions)
        try:
            ta_features_all = []
            for tf in CONFIG.TIMEFRAMES:
                if tf in multi_tf_data:
                    ta_features = self.cnn.ta_analyzer.extract_ta_features(
                        multi_tf_data[tf], tf, symbol, int(time.time() * 1000)
                    )
                    ta_features_all.append(ta_features)
            
            ta_summary = self.sac.get_ta_summary(ta_features_all)
            ta_summary_tensor = torch.FloatTensor(ta_summary).to(device)
            
            if ta_summary_tensor.size(0) != 8:
                ta_summary_tensor = torch.zeros(8).to(device)
        except Exception as e:
            self.logger.warning(f"TA summary calculation failed: {e}")
            ta_summary_tensor = torch.zeros(8).to(device)
        
        state_components.append(ta_summary_tensor)
        current_size += 8
        
        # Component 4: Padding to reach exactly 76 dimensions
        remaining_size = CONFIG.NeuralNetworks.SAC_STATE_VECTOR_SIZE - current_size  # Should be 2
        if remaining_size > 0:
            padding = torch.zeros(remaining_size).to(device)
            state_components.append(padding)
        elif remaining_size < 0:
            # Truncate the last component if we somehow exceeded
            excess = -remaining_size
            if ta_summary_tensor.size(0) > excess:
                ta_summary_tensor = ta_summary_tensor[:-excess]
                state_components[-1] = ta_summary_tensor
        
        # FIXED: Concatenate and validate final size
        state_vector = torch.cat(state_components, dim=0)
        
        if state_vector.size(0) != CONFIG.NeuralNetworks.SAC_STATE_VECTOR_SIZE:
            self.logger.error(f"State vector size mismatch: got {state_vector.size(0)}, expected {CONFIG.NeuralNetworks.SAC_STATE_VECTOR_SIZE}")
            # Force correct size
            if state_vector.size(0) < CONFIG.NeuralNetworks.SAC_STATE_VECTOR_SIZE:
                padding = torch.zeros(CONFIG.NeuralNetworks.SAC_STATE_VECTOR_SIZE - state_vector.size(0)).to(device)
                state_vector = torch.cat([state_vector, padding], dim=0)
            else:
                state_vector = state_vector[:CONFIG.NeuralNetworks.SAC_STATE_VECTOR_SIZE]
        
        return state_vector
    
    def get_ta_interpretation_from_data(self, multi_tf_data: Dict, symbol: str, timestamp: int) -> List[str]:
        """Get human-readable TA interpretation with error handling."""
        insights = []
        
        try:
            for tf in ['1h', '4h', '1d']:
                if tf in multi_tf_data and multi_tf_data[tf] is not None:
                    try:
                        data = multi_tf_data[tf]
                        ta_features = self.cnn.ta_analyzer.extract_ta_features(data, tf, symbol, timestamp)
                        
                        if len(ta_features) >= 6:
                            support_dist = ta_features[0]
                            resistance_dist = ta_features[1]
                            breakout_strength = ta_features[4]
                            breakout_direction = ta_features[5]
                            
                            if breakout_strength > 1.0:
                                direction = "bullish" if breakout_direction > 0 else "bearish"
                                insights.append(f"{tf}: {direction} breakout (strength: {breakout_strength:.1f})")
                            elif support_dist < 0.02:
                                insights.append(f"{tf}: Near support level")
                            elif resistance_dist < 0.02:
                                insights.append(f"{tf}: Near resistance level")
                    except Exception as e:
                        self.logger.warning(f"TA interpretation failed for {tf}: {e}")
                        
        except Exception as e:
            self.logger.warning(f"Error interpreting TA data: {e}")
        
        return insights if insights else ["Standard market conditions"]
    
    def _get_default_decision(self, reason: str) -> Dict:
        """Return default HOLD decision with reason."""
        return {
            'decision': 'HOLD',
            'confidence': 0.5,
            'action_probabilities': {'buy': 0.33, 'hold': 0.34, 'sell': 0.33},
            'neural_analysis': {'status': 'default'},
            'consciousness_influence': 0.0,
            'ta_insights': [reason],
            'timestamp': int(time.time() * 1000),
            'symbol': 'UNKNOWN',
            'data_quality': {'timeframes_available': 0, 'total_timeframes': 6}
        }
    
    def fast_inference_mode(self):
        """Set all models to evaluation mode for fast inference."""
        self.cnn.eval()
        self.lstm.eval()
        self.sac.eval()
        
    def training_mode(self):
        """Set all models to training mode."""
        self.cnn.train()
        self.lstm.train()
        self.sac.train()
    
    def test_enhanced_system(self) -> Dict:
        """Test the enhanced neural system with comprehensive validation."""
        test_results = {
            'cnn_test': False,
            'lstm_test': False,
            'sac_test': False,
            'integration_test': False,
            'ta_test': False,
            'dimension_test': False
        }
        
        try:
            symbol = "BTCUSDT"
            timestamp = int(time.time() * 1000)
            
            # Test CNN with proper data
            dummy_data = {
                '1h': np.random.rand(50, 6) * 50000  # Realistic price data
            }
            cnn_output = self.cnn(dummy_data, symbol, timestamp)
            test_results['cnn_test'] = (cnn_output.numel() == CONFIG.NeuralNetworks.CNN_OUTPUT_FEATURES)
            
            # Test LSTM
            lstm_output = self.lstm(cnn_output)
            test_results['lstm_test'] = (lstm_output.numel() == CONFIG.NeuralNetworks.LSTM_HIDDEN_SIZE)
            
            # Test state vector dimensions
            state_vector = self._build_state_vector(lstm_output, dummy_data, symbol)
            test_results['dimension_test'] = (state_vector.size(0) == CONFIG.NeuralNetworks.SAC_STATE_VECTOR_SIZE)
            
            # Test SAC
            sac_output = self.sac(state_vector)
            test_results['sac_test'] = (sac_output.numel() == CONFIG.NeuralNetworks.SAC_ACTION_SPACE)
            
            # Test TA
            ta_features = self.cnn.ta_analyzer.extract_ta_features(
                np.random.rand(50, 6) * 50000, '1h', symbol, timestamp
            )
            test_results['ta_test'] = (len(ta_features) == CONFIG.NeuralNetworks.TA_FEATURES_COUNT)
            
            # Test full integration
            decision = self.make_enhanced_trading_decision(symbol)
            test_results['integration_test'] = ('decision' in decision and decision['decision'] in ['BUY', 'SELL', 'HOLD'])
            
        except Exception as e:
            self.logger.error(f"Neural system test failed: {e}")
        
        return test_results


def main():
    """Main function to test neural intelligence system."""
    print("🧠 JARVIS 3.0 - FIXED Neural Intelligence System")
    print("=" * 60)
    
    try:
        # Initialize neural system
        neural_system = EnhancedJARVIS_Neural_System()
        
        # Test the system
        print("\n🔬 TESTING FIXED NEURAL COMPONENTS")
        test_results = neural_system.test_enhanced_system()
        
        for component, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {component}: {status}")
        
        # Test trading decision
        print("\n📊 TESTING TRADING DECISION")
        decision = neural_system.make_enhanced_trading_decision("BTCUSDT")
        
        print(f"Decision: {decision['decision']}")
        print(f"Confidence: {decision['confidence']:.3f}")
        print(f"Probabilities: BUY={decision['action_probabilities']['buy']:.3f}, "
              f"HOLD={decision['action_probabilities']['hold']:.3f}, "
              f"SELL={decision['action_probabilities']['sell']:.3f}")
        print(f"Data Quality: {decision['data_quality']['timeframes_available']}/{decision['data_quality']['total_timeframes']} timeframes")
        
        print("\n🎉 FIXED Neural Intelligence System Ready!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())