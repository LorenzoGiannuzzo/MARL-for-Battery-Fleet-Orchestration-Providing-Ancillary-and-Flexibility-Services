"""
Italian Market Configuration
Centralized configuration for Italian electricity and flexibility markets
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class MarketHours:
    """Italian market timing configuration"""
    day_ahead_closure: int = 12      # 12:00 CET for next day
    intraday_closure: int = 1        # 1 hour before delivery
    msd_closure: int = 1             # MSD closure time
    real_time_start: int = 0         # Real-time market start


@dataclass
class TechnicalRequirements:
    """Technical requirements for BESS participation in Italian markets"""
    min_capacity_mw: float = 1.0           # Minimum capacity for market participation
    min_response_accuracy: float = 0.95    # 95% response accuracy required
    max_response_deviation: float = 0.05   # 5% maximum deviation from setpoint
    min_availability: float = 0.98         # 98% minimum availability
    
    # Response time requirements (seconds)
    fcr_response_time: int = 30            # FCR ≤ 30 seconds
    afrr_response_time: int = 200          # aFRR ≤ 200 seconds  
    mfrr_response_time: int = 900          # mFRR ≤ 15 minutes
    
    # Capacity requirements (MW)
    fcr_min_capacity: float = 1.0
    afrr_min_capacity: float = 1.0
    mfrr_min_capacity: float = 1.0
    
    # Duration requirements (hours)
    fcr_min_duration: int = 4
    afrr_min_duration: int = 4
    mfrr_min_duration: int = 1


@dataclass
class GridConnection:
    """Grid connection requirements for Italian TSO (Terna)"""
    voltage_levels_kv: List[int] = None    # Acceptable connection voltages
    power_factor_range: Tuple[float, float] = (0.95, 1.0)
    frequency_range_hz: Tuple[float, float] = (49.5, 50.5)
    voltage_tolerance: float = 0.10        # ±10% voltage tolerance
    
    def __post_init__(self):
        if self.voltage_levels_kv is None:
            self.voltage_levels_kv = [132, 220, 380, 400]  # Standard Italian voltage levels


@dataclass
class MeteringCommunication:
    """Metering and communication requirements"""
    metering_resolution_seconds: int = 1   # 1-second resolution required
    communication_protocol: str = "IEC 61850"  # Required communication standard
    data_retention_years: int = 5          # Years to retain operational data
    backup_communication: bool = True      # Backup communication required
    cybersecurity_standard: str = "IEC 62351"  # Cybersecurity requirements


class ItalianMarketConfig:
    """
    Comprehensive configuration for Italian electricity markets
    Includes PUN (Prezzo Unico Nazionale) and MSD (Mercato Servizi Dispacciamento)
    """
    
    def __init__(self):
        # Market operators and institutions
        self.institutions = {
            'market_operator': 'GME',    # Gestore dei Mercati Energetici
            'tso': 'Terna',             # Transmission System Operator  
            'regulator': 'ARERA',       # Autorità di Regolazione per Energia Reti e Ambiente
            'energy_authority': 'MITE', # Ministero della Transizione Ecologica
        }
        
        # Market timing
        self.market_hours = MarketHours()
        
        # Technical requirements
        self.technical_req = TechnicalRequirements()
        
        # Grid connection
        self.grid_connection = GridConnection()
        
        # Metering and communication
        self.metering_comm = MeteringCommunication()
        
        # ARERA tariff structure (updated values based on recent regulations)
        self.arera_flexibility_tariffs = {
            'FCR': {
                'capacity_price_base': 50.0,      # EUR/MW/h - calibrated to FCR Cooperation EU 2024 (Magnus Energy 2025)
                'energy_price_base': 100.0,       # EUR/MWh - FCR bundled in capacity, placeholder
                'seasonal_multiplier': {
                    'winter': 1.2,    # Dec, Jan, Feb
                    'spring': 1.0,    # Mar, Apr, May
                    'summer': 0.9,    # Jun, Jul, Aug
                    'autumn': 1.1,    # Sep, Oct, Nov
                },
                'hourly_multiplier': {
                    'peak': 1.3,      # 7-9, 18-21
                    'mid': 1.0,       # 10-17, 22-23
                    'off_peak': 0.8,  # 0-6
                }
            },
            'aFRR': {
                'capacity_price_base': 25.0,      # EUR/MW/h - calibrated to Italian aFRR benchmark 2022-2024
                'energy_price_base': 250.0,       # EUR/MWh - calibrated to ACER CHEST Italy aFRR 2022 (412 EUR/MWh)
                'seasonal_multiplier': {
                    'winter': 1.15,
                    'spring': 1.0,
                    'summer': 0.85,
                    'autumn': 1.05,
                },
                'hourly_multiplier': {
                    'peak': 1.25,
                    'mid': 1.0,
                    'off_peak': 0.85,
                }
            },
            'mFRR': {
                'capacity_price_base': 12.0,      # EUR/MW/h - calibrated to European mFRR/MARI benchmark
                'energy_price_base': 180.0,       # EUR/MWh - calibrated to European mFRR activation prices
                'seasonal_multiplier': {
                    'winter': 1.1,
                    'spring': 1.0,
                    'summer': 0.9,
                    'autumn': 1.0,
                },
                'hourly_multiplier': {
                    'peak': 1.2,
                    'mid': 1.0,
                    'off_peak': 0.9,
                }
            }
        }

        # BSP (Balancing Service Provider) costs
        self.bsp_costs = {
            'qualification_fee_annual': 5000.0,    # EUR/year
            'metering_cost_annual': 2000.0,        # EUR/year
            'communication_cost_annual': 1500.0,   # EUR/year
            'certification_cost': 3000.0,          # EUR one-time
            'audit_cost_annual': 1000.0,           # EUR/year
            'insurance_cost_annual': 2500.0,       # EUR/year
        }

        # Historical activation probabilities (based on Terna data 2020-2023)
        self.historical_activation_prob = {
            'FCR': {
                'mean': 0.10,
                'std': 0.08,
                'seasonal_variation': {
                    'winter': 1.3,    # Higher activation in winter
                    'spring': 0.9,
                    'summer': 0.8,    # Lower activation in summer
                    'autumn': 1.1,
                }
            },
            'aFRR': {
                'mean': 0.30,
                'std': 0.12,
                'seasonal_variation': {
                    'winter': 1.2,
                    'spring': 1.0,
                    'summer': 0.9,
                    'autumn': 1.0,
                }
            },
            'mFRR': {
                'mean': 0.20,
                'std': 0.15,
                'seasonal_variation': {
                    'winter': 1.15,
                    'spring': 1.0,
                    'summer': 0.95,
                    'autumn': 1.05,
                }
            }
        }
        
        # PUN (Prezzo Unico Nazionale) historical statistics
        self.pun_statistics = {
            'mean_price_2023': 127.0,        # EUR/MWh
            'std_price_2023': 45.0,          # EUR/MWh
            'min_price_2023': 15.0,          # EUR/MWh
            'max_price_2023': 350.0,         # EUR/MWh
            'negative_price_hours': 12,      # Hours with negative prices in 2023
        }
        
        # Market zones (Italy is divided into zones)
        self.market_zones = {
            'NORD': 'Northern Italy',
            'CNOR': 'Central-Northern Italy', 
            'CSUD': 'Central-Southern Italy',
            'SUD': 'Southern Italy',
            'SICI': 'Sicily',
            'SARD': 'Sardinia',
            'ROSN': 'Rossano',
            'FOGN': 'Foggia',
            'BRNN': 'Brindisi',
            'PRGP': 'Priolo Gargallo',
        }
        
        # Renewable energy integration parameters
        self.renewable_integration = {
            'solar_peak_hours': [11, 12, 13, 14, 15],
            'wind_variability_factor': 0.3,
            'renewable_penetration_2023': 0.41,  # 41% renewable penetration
            'flexibility_need_multiplier': 1.5,   # Increased flexibility need
        }
    
    def get_seasonal_multiplier(self, month: int, service_type: str) -> float:
        """
        Get seasonal price multiplier for given month and service type
        
        Args:
            month: Month (1-12)
            service_type: 'FCR', 'aFRR', or 'mFRR'
            
        Returns:
            Seasonal multiplier
        """
        if month in [12, 1, 2]:
            season = 'winter'
        elif month in [3, 4, 5]:
            season = 'spring'
        elif month in [6, 7, 8]:
            season = 'summer'
        else:  # [9, 10, 11]
            season = 'autumn'
            
        return self.arera_flexibility_tariffs[service_type]['seasonal_multiplier'][season]
    
    def get_hourly_multiplier(self, hour: int, service_type: str) -> float:
        """
        Get hourly price multiplier for given hour and service type
        
        Args:
            hour: Hour of day (0-23)
            service_type: 'FCR', 'aFRR', or 'mFRR'
            
        Returns:
            Hourly multiplier
        """
        if hour in [7, 8, 9, 18, 19, 20, 21]:
            period = 'peak'
        elif hour in [0, 1, 2, 3, 4, 5, 6]:
            period = 'off_peak'
        else:
            period = 'mid'
            
        return self.arera_flexibility_tariffs[service_type]['hourly_multiplier'][period]
    
    def get_activation_probability(self, service_type: str, month: int) -> float:
        """
        Get activation probability for service type and month
        
        Args:
            service_type: 'FCR', 'aFRR', or 'mFRR'
            month: Month (1-12)
            
        Returns:
            Activation probability [0,1]
        """
        base_prob = self.historical_activation_prob[service_type]['mean']
        
        if month in [12, 1, 2]:
            season = 'winter'
        elif month in [3, 4, 5]:
            season = 'spring'
        elif month in [6, 7, 8]:
            season = 'summer'
        else:
            season = 'autumn'
            
        seasonal_mult = self.historical_activation_prob[service_type]['seasonal_variation'][season]
        
        # Add some random variation
        std = self.historical_activation_prob[service_type]['std']
        variation = np.random.normal(0, std * 0.5)  # Reduced variation
        
        prob = base_prob * seasonal_mult + variation
        return np.clip(prob, 0.05, 0.8)  # Reasonable bounds
    
    def get_total_bsp_annual_cost(self) -> float:
        """Get total annual BSP costs"""
        return sum(self.bsp_costs.values())
    
    def is_peak_renewable_hour(self, hour: int) -> bool:
        """Check if hour is peak renewable generation"""
        return hour in self.renewable_integration['solar_peak_hours']
    
    def get_flexibility_need_factor(self, hour: int, month: int) -> float:
        """
        Get flexibility need factor based on renewable integration
        Higher values indicate more flexibility needed
        """
        base_factor = 1.0
        
        # Higher flexibility need during renewable peak hours
        if self.is_peak_renewable_hour(hour):
            base_factor *= self.renewable_integration['flexibility_need_multiplier']
        
        # Seasonal variation in renewable output
        if month in [5, 6, 7, 8]:  # High solar months
            base_factor *= 1.2
        elif month in [11, 12, 1, 2]:  # High wind months
            base_factor *= 1.1
            
        return base_factor


# Global configuration instance
ITALIAN_CONFIG = ItalianMarketConfig()