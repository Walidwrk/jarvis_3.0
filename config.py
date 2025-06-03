#!/usr/bin/env python3
"""
JARVIS 3.0 - Enhanced Configuration & Settings (FIXED)
Central configuration management with comprehensive validation and error checking.

ENHANCEMENTS ADDED:
- Configuration validation on startup
- Environment variable support with fallbacks
- Dynamic parameter adjustment capabilities
- Configuration file backup and versioning
- Runtime configuration validation
- Parameter bounds checking
- Performance optimization settings
- Enhanced error handling and logging

Author: JARVIS 3.0 Team
Version: 1.1 (ENHANCED - Validated Configuration)
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import threading

class ConfigurationError(Exception):
    """Exception raised for configuration validation errors."""
    pass

@dataclass
class ValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

class ConfigValidator:
    """Comprehensive configuration validator."""
    
    @staticmethod
    def validate_trading_params(config) -> ValidationResult:
        """Validate trading configuration parameters."""
        result = ValidationResult(is_valid=True)
        
        # Position size validation
        if not (0.001 <= config.Trading.MIN_POSITION_SIZE <= config.Trading.MAX_POSITION_SIZE <= 1.0):
            result.errors.append("Invalid position size bounds")
            result.is_valid = False
        
        if config.Trading.DEFAULT_POSITION_SIZE > config.Trading.MAX_POSITION_SIZE:
            result.errors.append("Default position size exceeds maximum")
            result.is_valid = False
        
        # Stop loss validation
        if not (0.001 <= config.Trading.MIN_STOP_LOSS <= config.Trading.MAX_STOP_LOSS <= 0.5):
            result.errors.append("Invalid stop loss bounds")
            result.is_valid = False
        
        if config.Trading.DEFAULT_STOP_LOSS > config.Trading.MAX_STOP_LOSS:
            result.errors.append("Default stop loss exceeds maximum")
            result.is_valid = False
        
        # Take profit validation
        if not (0.005 <= config.Trading.MIN_TAKE_PROFIT <= config.Trading.MAX_TAKE_PROFIT <= 1.0):
            result.errors.append("Invalid take profit bounds")
            result.is_valid = False
        
        # Risk/reward ratio check
        if config.Trading.DEFAULT_TAKE_PROFIT < config.Trading.DEFAULT_STOP_LOSS:
            result.warnings.append("Default take profit is less than stop loss (negative risk/reward)")
        
        return result
    
    @staticmethod
    def validate_risk_limits(config) -> ValidationResult:
        """Validate risk management limits."""
        result = ValidationResult(is_valid=True)
        
        # Loss limits validation
        if not (0.001 <= config.RiskLimits.MAX_LOSS_PER_TRADE <= 0.1):
            result.errors.append("Max loss per trade must be between 0.1% and 10%")
            result.is_valid = False
        
        if not (0.01 <= config.RiskLimits.DAILY_DRAWDOWN_LIMIT <= 0.2):
            result.errors.append("Daily drawdown limit must be between 1% and 20%")
            result.is_valid = False
        
        if not (0.05 <= config.RiskLimits.MAX_TOTAL_DRAWDOWN <= 0.5):
            result.errors.append("Max total drawdown must be between 5% and 50%")
            result.is_valid = False
        
        # Position limits validation
        if not (1 <= config.RiskLimits.MAX_CONCURRENT_POSITIONS <= 10):
            result.errors.append("Max concurrent positions must be between 1 and 10")
            result.is_valid = False
        
        if not (0.05 <= config.RiskLimits.MAX_SYMBOL_EXPOSURE <= 1.0):
            result.errors.append("Max symbol exposure must be between 5% and 100%")
            result.is_valid = False
        
        # Emergency stop validation
        if not (0.05 <= config.RiskLimits.EMERGENCY_STOP_LOSS <= 0.8):
            result.errors.append("Emergency stop loss must be between 5% and 80%")
            result.is_valid = False
        
        # Logical consistency checks
        if config.RiskLimits.DAILY_DRAWDOWN_LIMIT >= config.RiskLimits.MAX_TOTAL_DRAWDOWN:
            result.warnings.append("Daily drawdown limit should be less than total drawdown limit")
        
        if config.RiskLimits.MAX_LOSS_PER_TRADE * config.RiskLimits.MAX_CONCURRENT_POSITIONS > config.RiskLimits.DAILY_DRAWDOWN_LIMIT:
            result.warnings.append("Combined maximum losses could exceed daily limit")
        
        return result
    
    @staticmethod
    def validate_neural_networks(config) -> ValidationResult:
        """Validate neural network configuration."""
        result = ValidationResult(is_valid=True)
        
        # Device validation
        if config.NeuralNetworks.DEVICE not in ['cpu', 'mps', 'cuda']:
            result.errors.append("Invalid device specification")
            result.is_valid = False
        
        # Model dimensions validation
        if not (1 <= config.NeuralNetworks.CNN_INPUT_CHANNELS <= 20):
            result.errors.append("CNN input channels must be between 1 and 20")
            result.is_valid = False
        
        if not (16 <= config.NeuralNetworks.CNN_OUTPUT_FEATURES <= 512):
            result.errors.append("CNN output features must be between 16 and 512")
            result.is_valid = False
        
        if not (16 <= config.NeuralNetworks.LSTM_HIDDEN_SIZE <= 512):
            result.errors.append("LSTM hidden size must be between 16 and 512")
            result.is_valid = False
        
        if not (32 <= config.NeuralNetworks.SAC_STATE_VECTOR_SIZE <= 512):
            result.errors.append("SAC state vector size must be between 32 and 512")
            result.is_valid = False
        
        # Training parameters validation
        if not (1e-6 <= config.NeuralNetworks.LEARNING_RATE <= 1e-1):
            result.errors.append("Learning rate must be between 1e-6 and 1e-1")
            result.is_valid = False
        
        if not (1 <= config.NeuralNetworks.BATCH_SIZE <= 1024):
            result.errors.append("Batch size must be between 1 and 1024")
            result.is_valid = False
        
        if not (1 <= config.NeuralNetworks.EPOCHS <= 1000):
            result.errors.append("Epochs must be between 1 and 1000")
            result.is_valid = False
        
        if not (0.0 <= config.NeuralNetworks.DROPOUT_RATE <= 0.9):
            result.errors.append("Dropout rate must be between 0.0 and 0.9")
            result.is_valid = False
        
        # Performance warnings
        if config.NeuralNetworks.BATCH_SIZE > 256:
            result.warnings.append("Large batch size may cause memory issues")
        
        if config.NeuralNetworks.LEARNING_RATE > 1e-2:
            result.warnings.append("High learning rate may cause training instability")
        
        return result
    
    @staticmethod
    def validate_file_paths(config) -> ValidationResult:
        """Validate file paths and directory structure."""
        result = ValidationResult(is_valid=True)
        
        # Check if base directory exists
        if not os.path.exists(config.BASE_DIR):
            result.errors.append(f"Base directory does not exist: {config.BASE_DIR}")
            result.is_valid = False
        
        # Check critical directories
        critical_dirs = [config.DATA_DIR, config.LOGS_DIR]
        for dir_path in critical_dirs:
            if not os.path.exists(dir_path):
                try:
                    os.makedirs(dir_path, exist_ok=True)
                    result.suggestions.append(f"Created missing directory: {dir_path}")
                except Exception as e:
                    result.errors.append(f"Cannot create directory {dir_path}: {e}")
                    result.is_valid = False
        
        # Check database path
        db_dir = os.path.dirname(config.DATABASE_PATH)
        if not os.path.exists(db_dir):
            try:
                os.makedirs(db_dir, exist_ok=True)
                result.suggestions.append(f"Created database directory: {db_dir}")
            except Exception as e:
                result.errors.append(f"Cannot create database directory: {e}")
                result.is_valid = False
        
        return result


class JARVISConfig:
    """Enhanced central configuration class with validation and dynamic updates."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern implementation."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """Initialize configuration with validation."""
        if self._initialized:
            return
        
        # Version and metadata
        self.VERSION = "3.1"
        self.BUILD_DATE = datetime.now().isoformat()
        self.CONFIG_VERSION = "1.1"
        
        # Environment detection
        self.ENVIRONMENT = os.getenv('JARVIS_ENV', 'development')
        self.DEBUG_MODE = os.getenv('JARVIS_DEBUG', 'false').lower() == 'true'
        
        # Base paths with environment variable support
        self.BASE_DIR = os.getenv('JARVIS_BASE_DIR', os.path.dirname(os.path.abspath(__file__)))
        self.DATA_DIR = os.getenv('JARVIS_DATA_DIR', os.path.join(self.BASE_DIR, "data"))
        self.MODELS_DIR = os.getenv('JARVIS_MODELS_DIR', os.path.join(self.BASE_DIR, "models"))
        self.LOGS_DIR = os.getenv('JARVIS_LOGS_DIR', os.path.join(self.BASE_DIR, "logs"))
        self.REPORTS_DIR = os.getenv('JARVIS_REPORTS_DIR', os.path.join(self.BASE_DIR, "reports"))
        self.TESTS_DIR = os.getenv('JARVIS_TESTS_DIR', os.path.join(self.BASE_DIR, "tests"))
        
        # Database configuration with environment support
        self.DATABASE_PATH = os.getenv('JARVIS_DB_PATH', os.path.join(self.DATA_DIR, "crypto_data.db"))
        self.MEMORY_DATABASE = self.DATABASE_PATH  # Same database for memory system
        
        # Trading symbols and timeframes with validation
        default_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOTUSDT']
        symbols_env = os.getenv('JARVIS_SYMBOLS', ','.join(default_symbols))
        self.SYMBOLS = [s.strip().upper() for s in symbols_env.split(',') if s.strip()]
        
        default_timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        timeframes_env = os.getenv('JARVIS_TIMEFRAMES', ','.join(default_timeframes))
        self.TIMEFRAMES = [tf.strip().lower() for tf in timeframes_env.split(',') if tf.strip()]
        
        # Initialize nested configuration classes
        self.Trading = self.TradingConfig()
        self.RiskLimits = self.RiskLimitsConfig()
        self.Consciousness = self.ConsciousnessConfig()
        self.NeuralNetworks = self.NeuralNetworksConfig()
        self.Memory = self.MemoryConfig()
        self.DataManager = self.DataManagerConfig()
        self.Logging = self.LoggingConfig()
        self.StrategyModes = self.StrategyModesConfig()
        self.PerformanceTargets = self.PerformanceTargetsConfig()
        
        # Validation and setup
        self._validation_results = {}
        self._last_validation = None
        
        # Setup logging
        self._setup_logging()
        
        # Validate configuration
        self.validate_configuration()
        
        # Ensure directories exist
        self.ensure_directories()
        
        self._initialized = True
    
    def _setup_logging(self):
        """Setup configuration logging."""
        self.logger = logging.getLogger('JARVIS_Config')
    
    class TradingConfig:
        """Enhanced trading configuration with environment variable support."""
        
        def __init__(self):
            # Position sizing with bounds
            self.DEFAULT_POSITION_SIZE = float(os.getenv('JARVIS_DEFAULT_POSITION_SIZE', '0.10'))
            self.MIN_POSITION_SIZE = float(os.getenv('JARVIS_MIN_POSITION_SIZE', '0.01'))
            self.MAX_POSITION_SIZE = float(os.getenv('JARVIS_MAX_POSITION_SIZE', '0.20'))
            
            # Stop loss configuration
            self.DEFAULT_STOP_LOSS = float(os.getenv('JARVIS_DEFAULT_STOP_LOSS', '0.015'))
            self.MIN_STOP_LOSS = float(os.getenv('JARVIS_MIN_STOP_LOSS', '0.005'))
            self.MAX_STOP_LOSS = float(os.getenv('JARVIS_MAX_STOP_LOSS', '0.050'))
            
            # Take profit configuration
            self.DEFAULT_TAKE_PROFIT = float(os.getenv('JARVIS_DEFAULT_TAKE_PROFIT', '0.025'))
            self.MIN_TAKE_PROFIT = float(os.getenv('JARVIS_MIN_TAKE_PROFIT', '0.010'))
            self.MAX_TAKE_PROFIT = float(os.getenv('JARVIS_MAX_TAKE_PROFIT', '0.100'))
            
            # Trailing stop
            self.TRAILING_STOP_FACTOR = float(os.getenv('JARVIS_TRAILING_STOP_FACTOR', '0.7'))
            
            # Advanced trading parameters
            self.SLIPPAGE_TOLERANCE = float(os.getenv('JARVIS_SLIPPAGE_TOLERANCE', '0.001'))
            self.MAX_SPREAD_TOLERANCE = float(os.getenv('JARVIS_MAX_SPREAD_TOLERANCE', '0.002'))
            self.ORDER_TIMEOUT_SECONDS = int(os.getenv('JARVIS_ORDER_TIMEOUT', '30'))
    
    class RiskLimitsConfig:
        """Enhanced risk management limits with validation."""
        
        def __init__(self):
            # Hard safety limits
            self.MAX_LOSS_PER_TRADE = float(os.getenv('JARVIS_MAX_LOSS_PER_TRADE', '0.01'))
            self.DAILY_DRAWDOWN_LIMIT = float(os.getenv('JARVIS_DAILY_DRAWDOWN_LIMIT', '0.03'))
            self.MAX_TOTAL_DRAWDOWN = float(os.getenv('JARVIS_MAX_TOTAL_DRAWDOWN', '0.08'))
            
            # Portfolio limits
            self.MAX_CONCURRENT_POSITIONS = int(os.getenv('JARVIS_MAX_CONCURRENT_POSITIONS', '3'))
            self.MAX_SYMBOL_EXPOSURE = float(os.getenv('JARVIS_MAX_SYMBOL_EXPOSURE', '0.15'))
            
            # Emergency stops
            self.EMERGENCY_STOP_LOSS = float(os.getenv('JARVIS_EMERGENCY_STOP_LOSS', '0.10'))
            self.MAX_CONSECUTIVE_LOSSES = int(os.getenv('JARVIS_MAX_CONSECUTIVE_LOSSES', '5'))
            
            # Advanced risk parameters
            self.CORRELATION_LIMIT = float(os.getenv('JARVIS_CORRELATION_LIMIT', '0.7'))
            self.VOLATILITY_ADJUSTMENT_FACTOR = float(os.getenv('JARVIS_VOLATILITY_ADJUSTMENT', '1.5'))
            self.MARGIN_SAFETY_BUFFER = float(os.getenv('JARVIS_MARGIN_SAFETY_BUFFER', '0.2'))
    
    class ConsciousnessConfig:
        """Enhanced consciousness system configuration."""
        
        def __init__(self):
            # Emotion boundaries (0.0 to 10.0)
            self.MIN_EMOTION = float(os.getenv('JARVIS_MIN_EMOTION', '0.0'))
            self.MAX_EMOTION = float(os.getenv('JARVIS_MAX_EMOTION', '10.0'))
            self.NEUTRAL_EMOTION = float(os.getenv('JARVIS_NEUTRAL_EMOTION', '5.0'))
            
            # Emotion update rates for wins
            self.WIN_CONFIDENCE_BOOST = float(os.getenv('JARVIS_WIN_CONFIDENCE_BOOST', '0.5'))
            self.WIN_FEAR_REDUCTION = float(os.getenv('JARVIS_WIN_FEAR_REDUCTION', '0.2'))
            self.WIN_GREED_INCREASE = float(os.getenv('JARVIS_WIN_GREED_INCREASE', '0.3'))
            
            # Emotion update rates for losses
            self.LOSS_CONFIDENCE_DROP = float(os.getenv('JARVIS_LOSS_CONFIDENCE_DROP', '0.7'))
            self.LOSS_FEAR_INCREASE = float(os.getenv('JARVIS_LOSS_FEAR_INCREASE', '0.5'))
            self.LOSS_GREED_REDUCTION = float(os.getenv('JARVIS_LOSS_GREED_REDUCTION', '0.4'))
            
            # Strategy performance thresholds
            self.MIN_TRADES_FOR_ANALYSIS = int(os.getenv('JARVIS_MIN_TRADES_ANALYSIS', '10'))
            self.POOR_PERFORMANCE_THRESHOLD = float(os.getenv('JARVIS_POOR_PERFORMANCE_THRESHOLD', '0.4'))
            self.SWITCH_STRATEGY_THRESHOLD = float(os.getenv('JARVIS_SWITCH_STRATEGY_THRESHOLD', '0.35'))
            
            # Advanced consciousness parameters
            self.EMOTION_DECAY_RATE = float(os.getenv('JARVIS_EMOTION_DECAY_RATE', '0.01'))
            self.MEMORY_INFLUENCE_FACTOR = float(os.getenv('JARVIS_MEMORY_INFLUENCE_FACTOR', '0.3'))
            self.ADAPTATION_SPEED = float(os.getenv('JARVIS_ADAPTATION_SPEED', '0.1'))
    
    class NeuralNetworksConfig:
        """Enhanced neural network configuration with dynamic optimization."""
        
        def __init__(self):
            # Device configuration with automatic detection
            default_device = "mps" if self._detect_mps() else ("cuda" if self._detect_cuda() else "cpu")
            self.DEVICE = os.getenv('JARVIS_DEVICE', default_device)
            self.FORCE_CPU = os.getenv('JARVIS_FORCE_CPU', 'false').lower() == 'true'
            
            # Model dimensions
            self.CNN_INPUT_CHANNELS = int(os.getenv('JARVIS_CNN_INPUT_CHANNELS', '6'))
            self.CNN_OUTPUT_FEATURES = int(os.getenv('JARVIS_CNN_OUTPUT_FEATURES', '64'))
            self.LSTM_HIDDEN_SIZE = int(os.getenv('JARVIS_LSTM_HIDDEN_SIZE', '64'))
            self.SAC_STATE_VECTOR_SIZE = int(os.getenv('JARVIS_SAC_STATE_VECTOR_SIZE', '76'))
            self.SAC_ACTION_SPACE = int(os.getenv('JARVIS_SAC_ACTION_SPACE', '3'))
            
            # Training parameters
            self.LEARNING_RATE = float(os.getenv('JARVIS_LEARNING_RATE', '0.0003'))
            self.BATCH_SIZE = int(os.getenv('JARVIS_BATCH_SIZE', '32'))
            self.EPOCHS = int(os.getenv('JARVIS_EPOCHS', '50'))
            self.DROPOUT_RATE = float(os.getenv('JARVIS_DROPOUT_RATE', '0.2'))
            
            # Technical analysis features
            self.TA_FEATURES_COUNT = int(os.getenv('JARVIS_TA_FEATURES_COUNT', '12'))
            
            # Advanced neural network parameters
            self.GRADIENT_CLIP_VALUE = float(os.getenv('JARVIS_GRADIENT_CLIP_VALUE', '1.0'))
            self.WEIGHT_DECAY = float(os.getenv('JARVIS_WEIGHT_DECAY', '1e-5'))
            self.SCHEDULER_PATIENCE = int(os.getenv('JARVIS_SCHEDULER_PATIENCE', '10'))
            self.EARLY_STOPPING_PATIENCE = int(os.getenv('JARVIS_EARLY_STOPPING_PATIENCE', '20'))
        
        @staticmethod
        def _detect_mps() -> bool:
            """Detect MPS (Apple Silicon) availability."""
            try:
                import torch
                return torch.backends.mps.is_available()
            except:
                return False
        
        @staticmethod
        def _detect_cuda() -> bool:
            """Detect CUDA availability."""
            try:
                import torch
                return torch.cuda.is_available()
            except:
                return False
    
    class MemoryConfig:
        """Enhanced memory system configuration."""
        
        def __init__(self):
            # Pattern storage
            self.MAX_PATTERN_MEMORIES = int(os.getenv('JARVIS_MAX_PATTERN_MEMORIES', '10000'))
            self.PATTERN_SIMILARITY_THRESHOLD = float(os.getenv('JARVIS_PATTERN_SIMILARITY_THRESHOLD', '0.85'))
            
            # Experience replay
            self.EXPERIENCE_REPLAY_SIZE = int(os.getenv('JARVIS_EXPERIENCE_REPLAY_SIZE', '1000'))
            self.MIN_EXPERIENCES_FOR_REPLAY = int(os.getenv('JARVIS_MIN_EXPERIENCES_FOR_REPLAY', '100'))
            
            # Memory cleanup
            self.MEMORY_CLEANUP_INTERVAL = int(os.getenv('JARVIS_MEMORY_CLEANUP_INTERVAL', '1000'))
            self.MEMORY_RETENTION_DAYS = int(os.getenv('JARVIS_MEMORY_RETENTION_DAYS', '90'))
            
            # Advanced memory parameters
            self.MEMORY_COMPRESSION_THRESHOLD = int(os.getenv('JARVIS_MEMORY_COMPRESSION_THRESHOLD', '50000'))
            self.PATTERN_VALIDATION_SAMPLES = int(os.getenv('JARVIS_PATTERN_VALIDATION_SAMPLES', '5'))
            self.CONFIDENCE_THRESHOLD = float(os.getenv('JARVIS_CONFIDENCE_THRESHOLD', '0.7'))
    
    class DataManagerConfig:
        """Enhanced data management configuration."""
        
        def __init__(self):
            # Data collection
            self.BINANCE_API_LIMIT = int(os.getenv('JARVIS_BINANCE_API_LIMIT', '1000'))
            self.DATA_UPDATE_INTERVAL = int(os.getenv('JARVIS_DATA_UPDATE_INTERVAL', '60'))
            
            # Data retention
            self.MIN_HISTORICAL_DAYS = int(os.getenv('JARVIS_MIN_HISTORICAL_DAYS', '365'))
            self.MAX_HISTORICAL_DAYS = int(os.getenv('JARVIS_MAX_HISTORICAL_DAYS', '730'))
            
            # Data quality
            self.MIN_VOLUME_THRESHOLD = float(os.getenv('JARVIS_MIN_VOLUME_THRESHOLD', '1000'))
            self.MAX_PRICE_DEVIATION = float(os.getenv('JARVIS_MAX_PRICE_DEVIATION', '0.1'))
            
            # API management
            self.API_RETRY_ATTEMPTS = int(os.getenv('JARVIS_API_RETRY_ATTEMPTS', '3'))
            self.API_TIMEOUT_SECONDS = int(os.getenv('JARVIS_API_TIMEOUT_SECONDS', '15'))
            self.RATE_LIMIT_DELAY = float(os.getenv('JARVIS_RATE_LIMIT_DELAY', '0.1'))
            
            # Data validation
            self.QUALITY_SCORE_THRESHOLD = float(os.getenv('JARVIS_QUALITY_SCORE_THRESHOLD', '0.5'))
            self.STALENESS_WARNING_MINUTES = int(os.getenv('JARVIS_STALENESS_WARNING_MINUTES', '60'))
    
    class LoggingConfig:
        """Enhanced logging configuration."""
        
        def __init__(self):
            # Log levels
            self.CONSOLE_LOG_LEVEL = os.getenv('JARVIS_CONSOLE_LOG_LEVEL', 'INFO')
            self.FILE_LOG_LEVEL = os.getenv('JARVIS_FILE_LOG_LEVEL', 'DEBUG')
            
            # Log rotation
            self.MAX_LOG_SIZE = int(os.getenv('JARVIS_MAX_LOG_SIZE', str(10 * 1024 * 1024)))
            self.BACKUP_COUNT = int(os.getenv('JARVIS_BACKUP_COUNT', '5'))
            
            # Log files
            self.MAIN_LOG = os.getenv('JARVIS_MAIN_LOG', 'jarvis_main.log')
            self.TRADING_LOG = os.getenv('JARVIS_TRADING_LOG', 'jarvis_trading.log')
            self.CONSCIOUSNESS_LOG = os.getenv('JARVIS_CONSCIOUSNESS_LOG', 'jarvis_consciousness.log')
            self.NEURAL_LOG = os.getenv('JARVIS_NEURAL_LOG', 'jarvis_neural.log')
            
            # Advanced logging
            self.STRUCTURED_LOGGING = os.getenv('JARVIS_STRUCTURED_LOGGING', 'true').lower() == 'true'
            self.LOG_COMPRESSION = os.getenv('JARVIS_LOG_COMPRESSION', 'true').lower() == 'true'
            self.PERFORMANCE_LOGGING = os.getenv('JARVIS_PERFORMANCE_LOGGING', 'true').lower() == 'true'
    
    class StrategyModesConfig:
        """Enhanced strategy modes configuration."""
        
        def __init__(self):
            # Strategy mode identifiers
            self.VOLATILE_MARKET = "volatile_market"
            self.TRENDING_MARKET = "trending_market"
            self.RANGING_MARKET = "ranging_market"
            self.BREAKOUT_MODE = "breakout_mode"
            self.REVERSAL_MODE = "reversal_mode"
            
            # Enhanced strategy parameters
            self.STRATEGY_CONFIGS = {
                self.VOLATILE_MARKET: {
                    'position_size_multiplier': float(os.getenv('JARVIS_VOLATILE_POSITION_MULT', '0.6')),
                    'trailing_stop_factor': float(os.getenv('JARVIS_VOLATILE_TRAILING_STOP', '1.5')),
                    'exit_speed': os.getenv('JARVIS_VOLATILE_EXIT_SPEED', 'FAST'),
                    'profit_target_multiplier': float(os.getenv('JARVIS_VOLATILE_PROFIT_MULT', '0.8')),
                    'risk_tolerance': 'LOW',
                    'entry_confidence_threshold': 0.8
                },
                self.TRENDING_MARKET: {
                    'position_size_multiplier': float(os.getenv('JARVIS_TRENDING_POSITION_MULT', '1.2')),
                    'trailing_stop_factor': float(os.getenv('JARVIS_TRENDING_TRAILING_STOP', '0.7')),
                    'exit_speed': os.getenv('JARVIS_TRENDING_EXIT_SPEED', 'SLOW'),
                    'profit_target_multiplier': float(os.getenv('JARVIS_TRENDING_PROFIT_MULT', '1.5')),
                    'risk_tolerance': 'MEDIUM',
                    'entry_confidence_threshold': 0.6
                },
                self.RANGING_MARKET: {
                    'position_size_multiplier': float(os.getenv('JARVIS_RANGING_POSITION_MULT', '1.0')),
                    'trailing_stop_factor': float(os.getenv('JARVIS_RANGING_TRAILING_STOP', '1.0')),
                    'exit_speed': os.getenv('JARVIS_RANGING_EXIT_SPEED', 'MEDIUM'),
                    'profit_target_multiplier': float(os.getenv('JARVIS_RANGING_PROFIT_MULT', '1.0')),
                    'risk_tolerance': 'MEDIUM',
                    'entry_confidence_threshold': 0.7
                },
                self.BREAKOUT_MODE: {
                    'position_size_multiplier': float(os.getenv('JARVIS_BREAKOUT_POSITION_MULT', '1.3')),
                    'trailing_stop_factor': float(os.getenv('JARVIS_BREAKOUT_TRAILING_STOP', '0.6')),
                    'exit_speed': os.getenv('JARVIS_BREAKOUT_EXIT_SPEED', 'VERY_SLOW'),
                    'profit_target_multiplier': float(os.getenv('JARVIS_BREAKOUT_PROFIT_MULT', '2.0')),
                    'risk_tolerance': 'HIGH',
                    'entry_confidence_threshold': 0.5
                },
                self.REVERSAL_MODE: {
                    'position_size_multiplier': float(os.getenv('JARVIS_REVERSAL_POSITION_MULT', '0.8')),
                    'trailing_stop_factor': float(os.getenv('JARVIS_REVERSAL_TRAILING_STOP', '1.2')),
                    'exit_speed': os.getenv('JARVIS_REVERSAL_EXIT_SPEED', 'FAST'),
                    'profit_target_multiplier': float(os.getenv('JARVIS_REVERSAL_PROFIT_MULT', '0.9')),
                    'risk_tolerance': 'LOW',
                    'entry_confidence_threshold': 0.8
                }
            }
    
    class PerformanceTargetsConfig:
        """Enhanced performance targets configuration."""
        
        def __init__(self):
            # Win rate targets
            self.MINIMUM_WIN_RATE = float(os.getenv('JARVIS_MINIMUM_WIN_RATE', '0.55'))
            self.TARGET_WIN_RATE = float(os.getenv('JARVIS_TARGET_WIN_RATE', '0.65'))
            self.EXCELLENT_WIN_RATE = float(os.getenv('JARVIS_EXCELLENT_WIN_RATE', '0.75'))
            
            # Profit targets
            self.MINIMUM_DAILY_RETURN = float(os.getenv('JARVIS_MINIMUM_DAILY_RETURN', '0.005'))
            self.TARGET_DAILY_RETURN = float(os.getenv('JARVIS_TARGET_DAILY_RETURN', '0.015'))
            self.EXCELLENT_DAILY_RETURN = float(os.getenv('JARVIS_EXCELLENT_DAILY_RETURN', '0.025'))
            
            # Risk metrics
            self.TARGET_SHARPE_RATIO = float(os.getenv('JARVIS_TARGET_SHARPE_RATIO', '2.0'))
            self.MAX_DRAWDOWN_TOLERANCE = float(os.getenv('JARVIS_MAX_DRAWDOWN_TOLERANCE', '0.05'))
            
            # Advanced performance metrics
            self.TARGET_PROFIT_FACTOR = float(os.getenv('JARVIS_TARGET_PROFIT_FACTOR', '1.5'))
            self.TARGET_CALMAR_RATIO = float(os.getenv('JARVIS_TARGET_CALMAR_RATIO', '3.0'))
            self.MAX_CONSECUTIVE_LOSSES = int(os.getenv('JARVIS_MAX_CONSECUTIVE_LOSSES_TARGET', '3'))
    
    def validate_configuration(self) -> ValidationResult:
        """Comprehensive configuration validation."""
        self.logger.info("Validating configuration...")
        
        overall_result = ValidationResult(is_valid=True)
        
        # Validate each configuration section
        validators = [
            ("file_paths", ConfigValidator.validate_file_paths),
            ("trading_params", ConfigValidator.validate_trading_params),
            ("risk_limits", ConfigValidator.validate_risk_limits),
            ("neural_networks", ConfigValidator.validate_neural_networks),
        ]
        
        for section_name, validator_func in validators:
            try:
                section_result = validator_func(self)
                self._validation_results[section_name] = section_result
                
                if not section_result.is_valid:
                    overall_result.is_valid = False
                    overall_result.errors.extend([f"{section_name}: {error}" for error in section_result.errors])
                
                overall_result.warnings.extend([f"{section_name}: {warning}" for warning in section_result.warnings])
                overall_result.suggestions.extend([f"{section_name}: {suggestion}" for suggestion in section_result.suggestions])
                
            except Exception as e:
                error_msg = f"Validation failed for {section_name}: {e}"
                overall_result.errors.append(error_msg)
                overall_result.is_valid = False
                self.logger.error(error_msg)
        
        # Additional cross-section validations
        self._validate_cross_section_consistency(overall_result)
        
        self._last_validation = datetime.now()
        
        # Log validation results
        if overall_result.is_valid:
            self.logger.info("Configuration validation passed")
            if overall_result.warnings:
                self.logger.warning(f"Configuration warnings: {len(overall_result.warnings)}")
        else:
            self.logger.error(f"Configuration validation failed: {len(overall_result.errors)} errors")
            for error in overall_result.errors:
                self.logger.error(f"  - {error}")
        
        if not overall_result.is_valid:
            raise ConfigurationError(f"Configuration validation failed: {overall_result.errors}")
        
        return overall_result
    
    def _validate_cross_section_consistency(self, result: ValidationResult):
        """Validate consistency across configuration sections."""
        # Check if daily drawdown limit allows for multiple max losses
        max_total_loss = (self.RiskLimits.MAX_LOSS_PER_TRADE * 
                         self.RiskLimits.MAX_CONCURRENT_POSITIONS)
        
        if max_total_loss > self.RiskLimits.DAILY_DRAWDOWN_LIMIT:
            result.warnings.append("Maximum concurrent losses could exceed daily drawdown limit")
        
        # Check if neural network state vector size is consistent
        expected_size = (self.NeuralNetworks.CNN_OUTPUT_FEATURES * len(self.TIMEFRAMES) + 
                        self.NeuralNetworks.TA_FEATURES_COUNT)
        
        if expected_size > self.NeuralNetworks.SAC_STATE_VECTOR_SIZE:
            result.warnings.append("SAC state vector size may be too small for CNN output")
        
        # Check if symbol count and position limits are reasonable
        if len(self.SYMBOLS) > self.RiskLimits.MAX_CONCURRENT_POSITIONS:
            result.suggestions.append("Consider increasing max concurrent positions for symbol diversification")
    
    def ensure_directories(self):
        """Ensure all required directories exist with proper permissions."""
        directories = [
            self.DATA_DIR,
            self.MODELS_DIR,
            self.LOGS_DIR,
            self.REPORTS_DIR,
            self.TESTS_DIR
        ]
        
        for directory in directories:
            try:
                Path(directory).mkdir(parents=True, exist_ok=True)
                # Test write permissions
                test_file = os.path.join(directory, '.jarvis_write_test')
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
                self.logger.debug(f"Directory validated: {directory}")
            except Exception as e:
                error_msg = f"Cannot create or write to directory {directory}: {e}"
                self.logger.error(error_msg)
                raise ConfigurationError(error_msg)
    
    def get_log_path(self, log_name: str) -> str:
        """Get full path for log file with validation."""
        if not log_name:
            raise ValueError("Log name cannot be empty")
        
        # Ensure log name has .log extension
        if not log_name.endswith('.log'):
            log_name += '.log'
        
        return os.path.join(self.LOGS_DIR, log_name)
    
    def get_model_path(self, model_name: str) -> str:
        """Get full path for model file with validation."""
        if not model_name:
            raise ValueError("Model name cannot be empty")
        
        return os.path.join(self.MODELS_DIR, model_name)
    
    def get_report_path(self, report_name: str) -> str:
        """Get full path for report file with timestamp."""
        if not report_name:
            raise ValueError("Report name cannot be empty")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{report_name}"
        
        return os.path.join(self.REPORTS_DIR, filename)
    
    def update_parameter(self, section: str, parameter: str, value: Any) -> bool:
        """
        Dynamically update configuration parameter with validation.
        
        Args:
            section: Configuration section name
            parameter: Parameter name
            value: New parameter value
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            if not hasattr(self, section):
                self.logger.error(f"Configuration section '{section}' does not exist")
                return False
            
            section_obj = getattr(self, section)
            
            if not hasattr(section_obj, parameter):
                self.logger.error(f"Parameter '{parameter}' does not exist in section '{section}'")
                return False
            
            # Store old value for rollback
            old_value = getattr(section_obj, parameter)
            
            # Set new value
            setattr(section_obj, parameter, value)
            
            # Validate configuration with new value
            try:
                validation_result = self.validate_configuration()
                if validation_result.is_valid:
                    self.logger.info(f"Updated {section}.{parameter}: {old_value} -> {value}")
                    return True
                else:
                    # Rollback on validation failure
                    setattr(section_obj, parameter, old_value)
                    self.logger.error(f"Parameter update failed validation, rolled back: {validation_result.errors}")
                    return False
            except Exception as e:
                # Rollback on exception
                setattr(section_obj, parameter, old_value)
                self.logger.error(f"Parameter update caused error, rolled back: {e}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to update parameter {section}.{parameter}: {e}")
            return False
    
    def export_configuration(self, include_sensitive: bool = False) -> Dict:
        """
        Export configuration to dictionary for backup or transfer.
        
        Args:
            include_sensitive: Whether to include sensitive configuration
            
        Returns:
            Configuration dictionary
        """
        config_dict = {
            'version': self.VERSION,
            'config_version': self.CONFIG_VERSION,
            'exported_at': datetime.now().isoformat(),
            'environment': self.ENVIRONMENT,
            'symbols': self.SYMBOLS,
            'timeframes': self.TIMEFRAMES,
        }
        
        # Export each configuration section
        sections = ['Trading', 'RiskLimits', 'Consciousness', 'NeuralNetworks', 
                   'Memory', 'DataManager', 'Logging', 'PerformanceTargets']
        
        for section_name in sections:
            if hasattr(self, section_name):
                section_obj = getattr(self, section_name)
                section_dict = {}
                
                for attr_name in dir(section_obj):
                    if not attr_name.startswith('_'):
                        attr_value = getattr(section_obj, attr_name)
                        if not callable(attr_value):
                            section_dict[attr_name] = attr_value
                
                config_dict[section_name.lower()] = section_dict
        
        return config_dict
    
    def save_configuration_backup(self, backup_path: Optional[str] = None) -> str:
        """
        Save configuration backup to file.
        
        Args:
            backup_path: Optional custom backup path
            
        Returns:
            Path to saved backup file
        """
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"jarvis_config_backup_{timestamp}.json"
            backup_path = os.path.join(self.DATA_DIR, backup_filename)
        
        try:
            config_dict = self.export_configuration()
            
            with open(backup_path, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
            
            self.logger.info(f"Configuration backup saved: {backup_path}")
            return backup_path
            
        except Exception as e:
            error_msg = f"Failed to save configuration backup: {e}"
            self.logger.error(error_msg)
            raise ConfigurationError(error_msg)
    
    def get_system_info(self) -> Dict:
        """Get comprehensive system configuration info."""
        info = {
            'version': self.VERSION,
            'config_version': self.CONFIG_VERSION,
            'build_date': self.BUILD_DATE,
            'environment': self.ENVIRONMENT,
            'debug_mode': self.DEBUG_MODE,
            'base_dir': self.BASE_DIR,
            'database_path': self.DATABASE_PATH,
            'symbols': self.SYMBOLS,
            'symbols_count': len(self.SYMBOLS),
            'timeframes': self.TIMEFRAMES,
            'timeframes_count': len(self.TIMEFRAMES),
            'neural_device': self.NeuralNetworks.DEVICE,
            'max_position_size': self.Trading.MAX_POSITION_SIZE,
            'daily_drawdown_limit': self.RiskLimits.DAILY_DRAWDOWN_LIMIT,
            'target_win_rate': self.PerformanceTargets.TARGET_WIN_RATE,
            'last_validation': self._last_validation.isoformat() if self._last_validation else None,
            'validation_status': all(result.is_valid for result in self._validation_results.values())
        }
        
        # Add hardware info if available
        try:
            import torch
            info['torch_version'] = torch.__version__
            info['cuda_available'] = torch.cuda.is_available()
            info['mps_available'] = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        except ImportError:
            info['torch_status'] = 'Not installed'
        
        return info
    
    def get_validation_report(self) -> Dict:
        """Get detailed validation report."""
        return {
            'last_validation': self._last_validation.isoformat() if self._last_validation else None,
            'overall_valid': all(result.is_valid for result in self._validation_results.values()),
            'section_results': {
                section: {
                    'valid': result.is_valid,
                    'errors': result.errors,
                    'warnings': result.warnings,
                    'suggestions': result.suggestions
                }
                for section, result in self._validation_results.items()
            }
        }


# =============================================================================
# CONVENIENCE ALIASES FOR EASY ACCESS
# =============================================================================

# Create global config instance
CONFIG = JARVISConfig()

# Easy access aliases
SYMBOLS = CONFIG.SYMBOLS
TIMEFRAMES = CONFIG.TIMEFRAMES
DATABASE_PATH = CONFIG.DATABASE_PATH
MODELS_DIR = CONFIG.MODELS_DIR
LOGS_DIR = CONFIG.LOGS_DIR


def main():
    """Display current configuration with validation status."""
    print("🔧 JARVIS 3.0 - Enhanced System Configuration")
    print("=" * 60)
    
    try:
        # Display system information
        info = CONFIG.get_system_info()
        print(f"\n📋 SYSTEM INFORMATION")
        print(f"Version: {info['version']}")
        print(f"Config Version: {info['config_version']}")
        print(f"Environment: {info['environment']}")
        print(f"Neural Device: {info['neural_device']}")
        print(f"Validation Status: {'✅ VALID' if info['validation_status'] else '❌ INVALID'}")
        
        # Display validation report
        validation_report = CONFIG.get_validation_report()
        print(f"\n🔍 VALIDATION REPORT")
        
        for section, result in validation_report['section_results'].items():
            status = "✅ VALID" if result['valid'] else "❌ INVALID"
            print(f"  {section.title()}: {status}")
            
            if result['errors']:
                print(f"    Errors: {len(result['errors'])}")
            if result['warnings']:
                print(f"    Warnings: {len(result['warnings'])}")
            if result['suggestions']:
                print(f"    Suggestions: {len(result['suggestions'])}")
        
        # Display key configuration values
        print(f"\n⚙️ KEY CONFIGURATION")
        print(f"Symbols: {len(CONFIG.SYMBOLS)} ({', '.join(CONFIG.SYMBOLS)})")
        print(f"Timeframes: {len(CONFIG.TIMEFRAMES)} ({', '.join(CONFIG.TIMEFRAMES)})")
        print(f"Max Position Size: {CONFIG.Trading.MAX_POSITION_SIZE:.1%}")
        print(f"Daily Drawdown Limit: {CONFIG.RiskLimits.DAILY_DRAWDOWN_LIMIT:.1%}")
        print(f"Target Win Rate: {CONFIG.PerformanceTargets.TARGET_WIN_RATE:.1%}")
        print(f"Neural State Vector Size: {CONFIG.NeuralNetworks.SAC_STATE_VECTOR_SIZE}")
        
        # Display directory information
        print(f"\n📁 DIRECTORIES")
        print(f"Base: {CONFIG.BASE_DIR}")
        print(f"Data: {CONFIG.DATA_DIR}")
        print(f"Models: {CONFIG.MODELS_DIR}")
        print(f"Logs: {CONFIG.LOGS_DIR}")
        print(f"Database: {CONFIG.DATABASE_PATH}")
        
        # Save configuration backup
        backup_path = CONFIG.save_configuration_backup()
        print(f"\n💾 Configuration backup saved: {backup_path}")
        
        print(f"\n🎉 Enhanced Configuration System: OPERATIONAL!")
        return 0
        
    except ConfigurationError as e:
        print(f"\n❌ Configuration Error: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ System Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())