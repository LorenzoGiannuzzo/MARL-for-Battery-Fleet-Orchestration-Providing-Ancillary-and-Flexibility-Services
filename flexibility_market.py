"""
Flexibility Market Models for Italian BESS Services
Implements ARERA-compliant flexibility services (FCR, aFRR, mFRR)
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum
import numpy as np


class ServiceType(Enum):
    """Italian flexibility service types according to ARERA"""
    FCR = "FCR"      # Frequency Containment Reserve (Primary)
    AFRR = "aFRR"    # Automatic Frequency Restoration Reserve (Secondary)
    MFRR = "mFRR"    # Manual Frequency Restoration Reserve (Tertiary)


@dataclass
class FlexibilityService:
    """Data model for flexibility service according to Italian regulations"""
    service_type: ServiceType
    capacity_price: float      # EUR/MW/h - capacity payment
    energy_price: float        # EUR/MWh - energy payment when activated
    activation_probability: float  # [0,1] - probability of activation based on historical data
    response_time: int         # seconds - maximum response time per ARERA
    min_capacity: float        # MW - minimum capacity requirement
    max_duration: int          # hours - maximum service duration
    min_duration: int          # hours - minimum service duration
    
    def __post_init__(self):
        """Validate service parameters according to ARERA regulations"""
        if self.service_type == ServiceType.FCR:
            assert self.response_time <= 30, "FCR response time must be ≤ 30 seconds"
            assert self.min_capacity >= 1.0, "FCR minimum capacity is 1 MW"
        elif self.service_type == ServiceType.AFRR:
            assert self.response_time <= 200, "aFRR response time must be ≤ 200 seconds"  
            assert self.min_capacity >= 1.0, "aFRR minimum capacity is 1 MW"
        elif self.service_type == ServiceType.MFRR:
            assert self.response_time <= 900, "mFRR response time must be ≤ 15 minutes"
            assert self.min_capacity >= 1.0, "mFRR minimum capacity is 1 MW"


@dataclass
class BatteryState:
    """Current state of the BESS for flexibility service evaluation"""
    soc: float                 # [0,1] - State of Charge
    available_power: float     # MW - currently available power capacity
    reserved_fcr: float        # MW - power reserved for FCR service
    reserved_afrr: float       # MW - power reserved for aFRR service  
    reserved_mfrr: float       # MW - power reserved for mFRR service
    degradation_cycles: float # cumulative equivalent cycles


class FlexibilityMarket:
    """
    Italian flexibility market model implementing ARERA regulations
    Manages FCR, aFRR, and mFRR services with realistic pricing and constraints
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """Initialize with ARERA-compliant default parameters"""
        # Set random seed for deterministic behavior
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # ====================================================================
        # CALIBRATED ARERA tariff structure (EUR/MW/h for capacity, EUR/MWh for energy)
        #
        # Values calibrated on:
        #   - Magnus Energy 'Developments in balancing and capacity markets', Nov 2025
        #     (European FCR Cooperation 2024 average ~50 EUR/MW/h)
        #   - ACER CHEST database, 'Volume-weighted aFRR prices Italy 2022'
        #     (Italian aFRR upward: 411.99 EUR/MWh)
        #   - Terna 'Report on Balancing 2022-2023' (download.terna.it)
        #   - ARERA TIDE Delibera 345/2023/R/eel
        #
        # Note: FCR is treated as a forward-looking market-based service per the
        # TIDE roadmap (full implementation by 2029). In current Italian practice,
        # FCR is a mandatory non-remunerated obligation for conventional units.
        # See accompanying file 'Italian_BESS_Markets_Reference.xlsx' for full
        # documentation of the calibration choices.
        # ====================================================================
        self.arera_tariffs = {
            ServiceType.FCR: {
                'capacity_base': 50.0,    # EUR/MW/h - aligned to FCR Cooperation EU 2024
                'energy_base': 100.0,     # EUR/MWh - placeholder (bundled in capacity in practice)
                'response_time': 30,      # seconds
                'min_capacity': 1.0,      # MW
                'max_duration': 24,       # hours
                'min_duration': 4,        # hours
            },
            ServiceType.AFRR: {
                'capacity_base': 25.0,    # EUR/MW/h - Italian aFRR benchmark 2022-2024
                'energy_base': 250.0,     # EUR/MWh - ACER CHEST Italy upward 2022 ~412, conservative
                'response_time': 200,     # seconds
                'min_capacity': 1.0,      # MW
                'max_duration': 24,       # hours
                'min_duration': 4,        # hours
            },
            ServiceType.MFRR: {
                'capacity_base': 12.0,    # EUR/MW/h - European mFRR/MARI benchmark
                'energy_base': 180.0,     # EUR/MWh - European mFRR activation prices
                'response_time': 900,     # seconds (15 minutes)
                'min_capacity': 1.0,      # MW
                'max_duration': 24,       # hours
                'min_duration': 1,        # hours
            }
        }

        # BSP (Balancing Service Provider) costs according to ARERA
        self.bsp_costs = {
            'qualification_fee': 5000.0,     # EUR/year - BSP qualification
            'metering_cost': 2000.0,          # EUR/year - metering requirements
            'communication_cost': 1500.0,    # EUR/year - communication systems
        }

        # ====================================================================
        # CALIBRATED historical activation probabilities
        # Source: Terna 'Report on Balancing 2022-2023', cross-referenced with
        # ENTSO-E aFRR activation data for Italy 2023-2024.
        # ====================================================================
        self.historical_probabilities = {
            ServiceType.FCR: 0.10,    # FCR: continuous low-volume activation; capacity payment dominates
            ServiceType.AFRR: 0.30,   # aFRR: most-activated service in Italy due to renewable integration
            ServiceType.MFRR: 0.20,   # mFRR: activated less than aFRR; matches Terna 2022-2023 statistics
        }
        
        # Seasonal and hourly multipliers for realistic pricing
        self.price_multipliers = {
            'peak_hours': [7, 8, 9, 18, 19, 20, 21],  # Peak demand hours
            'peak_multiplier': 1.3,
            'off_peak_multiplier': 0.8,
            'weekend_multiplier': 0.9,
        }
    
    def get_flexibility_opportunities(self, hour: int, day_of_week: int = 1) -> List[FlexibilityService]:
        """
        Get available flexibility service opportunities for given hour
        
        Args:
            hour: Hour of day (0-23)
            day_of_week: Day of week (0=Monday, 6=Sunday)
            
        Returns:
            List of available FlexibilityService objects
        """
        opportunities = []
        
        # Determine pricing multipliers
        is_peak = hour in self.price_multipliers['peak_hours']
        is_weekend = day_of_week >= 5
        
        multiplier = 1.0
        if is_peak:
            multiplier *= self.price_multipliers['peak_multiplier']
        else:
            multiplier *= self.price_multipliers['off_peak_multiplier']
            
        if is_weekend:
            multiplier *= self.price_multipliers['weekend_multiplier']
        
        # Add deterministic variation based on hour and day to simulate market conditions
        # This replaces the random variation to ensure reproducibility
        variation_seed = (hour * 7 + day_of_week) % 100
        price_variation = 0.9 + (variation_seed / 100.0) * 0.2  # Maps to [0.9, 1.1]
        final_multiplier = multiplier * price_variation
        
        # Create service opportunities for each type
        for service_type in ServiceType:
            tariff = self.arera_tariffs[service_type]
            
            service = FlexibilityService(
                service_type=service_type,
                capacity_price=tariff['capacity_base'] * final_multiplier,
                energy_price=tariff['energy_base'] * final_multiplier,
                activation_probability=self.historical_probabilities[service_type],
                response_time=tariff['response_time'],
                min_capacity=tariff['min_capacity'],
                max_duration=tariff['max_duration'],
                min_duration=tariff['min_duration']
            )
            
            opportunities.append(service)
        
        return opportunities
    
    def calculate_capacity_revenue(self, service: FlexibilityService, capacity_mw: float) -> float:
        """
        Calculate capacity revenue for providing flexibility service
        
        Args:
            service: FlexibilityService object
            capacity_mw: Capacity provided in MW
            
        Returns:
            Revenue in EUR/h
        """
        if capacity_mw < service.min_capacity:
            return 0.0
            
        return service.capacity_price * capacity_mw
    
    def calculate_activation_revenue(self, service: FlexibilityService, energy_mwh: float) -> float:
        """
        Calculate energy revenue when flexibility service is activated
        
        Args:
            service: FlexibilityService object  
            energy_mwh: Energy provided in MWh
            
        Returns:
            Revenue in EUR
        """
        if energy_mwh <= 0:
            return 0.0
            
        return service.energy_price * energy_mwh
    
    def check_service_constraints(self, service: FlexibilityService, bess_state: BatteryState) -> bool:
        """
        Check if BESS can provide the requested flexibility service
        
        Args:
            service: FlexibilityService to check
            bess_state: Current BESS state
            
        Returns:
            True if service can be provided, False otherwise
        """
        # Check minimum capacity requirement
        if bess_state.available_power < service.min_capacity:
            return False
            
        # Check if enough power is available after existing reservations
        total_reserved = (bess_state.reserved_fcr + 
                         bess_state.reserved_afrr + 
                         bess_state.reserved_mfrr)
        
        remaining_power = bess_state.available_power - total_reserved
        
        if remaining_power < service.min_capacity:
            return False
            
        # Check SOC constraints for different services
        if service.service_type == ServiceType.FCR:
            # FCR requires symmetric capability (charge/discharge)
            # Need sufficient SOC margin for both directions
            if bess_state.soc < 0.2 or bess_state.soc > 0.8:
                return False
                
        elif service.service_type == ServiceType.AFRR:
            # aFRR typically requires good SOC flexibility
            if bess_state.soc < 0.15 or bess_state.soc > 0.85:
                return False
                
        elif service.service_type == ServiceType.MFRR:
            # mFRR is more flexible with SOC requirements
            if bess_state.soc < 0.1 or bess_state.soc > 0.9:
                return False
        
        return True
    
    def calculate_expected_revenue(self, service: FlexibilityService, capacity_mw: float, 
                                 duration_hours: int = 1) -> Dict[str, float]:
        """
        Calculate expected total revenue including capacity and activation payments
        
        Args:
            service: FlexibilityService object
            capacity_mw: Capacity to provide in MW
            duration_hours: Service duration in hours
            
        Returns:
            Dictionary with revenue breakdown
        """
        if capacity_mw < service.min_capacity:
            return {
                'capacity_revenue': 0.0,
                'expected_energy_revenue': 0.0,
                'total_expected_revenue': 0.0
            }
        
        # Capacity revenue (guaranteed)
        capacity_revenue = self.calculate_capacity_revenue(service, capacity_mw) * duration_hours
        
        # Expected energy revenue (probabilistic)
        # Assume average activation of 50% of reserved capacity when activated
        expected_energy_mwh = (capacity_mw * 0.5 * duration_hours * 
                              service.activation_probability)
        expected_energy_revenue = self.calculate_activation_revenue(service, expected_energy_mwh)
        
        total_expected = capacity_revenue + expected_energy_revenue
        
        return {
            'capacity_revenue': capacity_revenue,
            'expected_energy_revenue': expected_energy_revenue,
            'total_expected_revenue': total_expected
        }
    
    def get_bsp_annual_costs(self) -> float:
        """
        Get total annual BSP (Balancing Service Provider) costs
        
        Returns:
            Total annual costs in EUR
        """
        return sum(self.bsp_costs.values())
    
    def simulate_service_activation(self, service: FlexibilityService, capacity_mw: float) -> Dict[str, float]:
        """
        Simulate whether a flexibility service gets activated and calculate actual revenue
        
        Args:
            service: FlexibilityService object
            capacity_mw: Reserved capacity in MW
            
        Returns:
            Dictionary with activation results
        """
        # Determine if service is activated based on probability
        is_activated = np.random.random() < service.activation_probability
        
        if not is_activated:
            return {
                'activated': False,
                'energy_provided': 0.0,
                'energy_revenue': 0.0,
                'capacity_revenue': self.calculate_capacity_revenue(service, capacity_mw)
            }
        
        # If activated, determine energy provided (random between 20% and 80% of capacity)
        activation_factor = np.random.uniform(0.2, 0.8)
        energy_provided = capacity_mw * activation_factor  # MWh for 1 hour
        
        energy_revenue = self.calculate_activation_revenue(service, energy_provided)
        capacity_revenue = self.calculate_capacity_revenue(service, capacity_mw)
        
        return {
            'activated': True,
            'energy_provided': energy_provided,
            'energy_revenue': energy_revenue,
            'capacity_revenue': capacity_revenue
        }


# Configuration constants for Italian market
ITALIAN_MARKET_CONFIG = {
    'market_operator': 'GME',  # Gestore dei Mercati Energetici
    'tso': 'Terna',           # Transmission System Operator
    'regulator': 'ARERA',     # Autorità di Regolazione per Energia Reti e Ambiente
    
    # Market timing (Italian time zone)
    'market_closure_day_ahead': 12,  # 12:00 CET for next day
    'market_closure_intraday': 1,    # 1 hour before delivery
    
    # Minimum technical requirements
    'min_bess_capacity': 1.0,        # MW minimum for market participation
    'min_response_accuracy': 0.95,   # 95% response accuracy required
    'max_response_deviation': 0.05,  # 5% maximum deviation from setpoint
    
    # Grid connection requirements
    'voltage_levels': [132, 220, 380, 400],  # kV - acceptable connection voltages
    'power_factor_range': (0.95, 1.0),       # Acceptable power factor range
    
    # Measurement and communication
    'metering_resolution': 1,         # 1-second resolution required
    'communication_protocol': 'IEC 61850',  # Required communication standard
    'data_retention_years': 5,        # Years to retain operational data
}