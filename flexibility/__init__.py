"""
Flexibility Services Module
Italian BESS flexibility services implementation
"""

# Import from the main flexibility_market.py file in root
from flexibility_market import FlexibilityMarket, FlexibilityService, BatteryState, ServiceType
from italian_market_config import ItalianMarketConfig, ITALIAN_CONFIG

__all__ = [
    'FlexibilityMarket',
    'FlexibilityService', 
    'BatteryState',
    'ServiceType',
    'ItalianMarketConfig',
    'ITALIAN_CONFIG'
]

__version__ = '1.0.0'
__author__ = 'Battery Flexibility MILP Comparison Project'