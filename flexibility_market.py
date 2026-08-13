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
    FCR = "FCR"          # Frequency Containment Reserve (Primary, symmetric)
    AFRR = "aFRR"        # legacy symmetric aFRR (kept for back-compat)
    MFRR = "mFRR"        # legacy symmetric mFRR (kept for back-compat)
    AFRR_UP = "aFRR_up"  # aFRR upward (BSP discharges on activation)
    AFRR_DN = "aFRR_dn"  # aFRR downward (BSP charges on activation)
    MFRR_UP = "mFRR_up"  # mFRR upward
    MFRR_DN = "mFRR_dn"  # mFRR downward


# Helper sets for direction-aware logic
DIRECTIONAL_SERVICES_UP = {ServiceType.AFRR_UP, ServiceType.MFRR_UP}
DIRECTIONAL_SERVICES_DN = {ServiceType.AFRR_DN, ServiceType.MFRR_DN}
DIRECTIONAL_SERVICES = DIRECTIONAL_SERVICES_UP | DIRECTIONAL_SERVICES_DN


@dataclass
class FlexibilityService:
    """Data model for flexibility service according to Italian regulations.

    Two pairs of prices are tracked:
      - capacity_price / energy_price: the FORECAST values that MILP and PPO
        observe at decision time. These are noisy (true * (1 + e), with e
        sampled from the uniform forecast-error distribution defined in the
        owning FlexibilityMarket).
      - true_capacity_price / true_energy_price: the actual settlement prices
        used when computing the realized payment. Equal to the forecast values
        when `flex_price_error_pct == 0` (deterministic regime).
    """
    service_type: ServiceType
    capacity_price: float      # EUR/MW/h - capacity payment (FORECAST visible to optimizer)
    energy_price: float        # EUR/MWh  - energy payment when activated (FORECAST)
    activation_probability: float  # [0,1] - probability of activation based on historical data
    response_time: int         # seconds - maximum response time per ARERA
    min_capacity: float        # MW - minimum capacity requirement
    max_duration: int          # hours - maximum service duration
    min_duration: int          # hours - minimum service duration
    # Realized settlement prices (used to compute realized payment ex-post)
    true_capacity_price: Optional[float] = None  # EUR/MW/h - true settlement price
    true_energy_price: Optional[float] = None    # EUR/MWh  - true settlement price
    # June 2026: hourly probability that a bid for this service WINS the
    # capacity auction. Models the fact that the BSP does not always succeed
    # in clearing the aFRR/FCR/mFRR auction: the bid is made (and the
    # capacity is committed) every hour the optimiser decides to participate,
    # but only with probability `award_probability` is the bid actually
    # accepted and the corresponding revenue paid. When the bid is rejected,
    # no capacity or energy revenue accrues for that hour, even though the
    # decision committed power that could have been used for arbitrage.
    # Default 1.0 reproduces the legacy "always-wins" behaviour.
    award_probability: float = 1.0

    def __post_init__(self):
        """Validate service parameters according to ARERA regulations"""
        if self.service_type == ServiceType.FCR:
            assert self.response_time <= 30, "FCR response time must be \u2264 30 seconds"
            assert self.min_capacity >= 1.0, "FCR minimum capacity is 1 MW"
        elif self.service_type in (ServiceType.AFRR, ServiceType.AFRR_UP, ServiceType.AFRR_DN):
            assert self.response_time <= 200, "aFRR response time must be \u2264 200 seconds"
            assert self.min_capacity >= 1.0, "aFRR minimum capacity is 1 MW"
        elif self.service_type in (ServiceType.MFRR, ServiceType.MFRR_UP, ServiceType.MFRR_DN):
            assert self.response_time <= 900, "mFRR response time must be \u2264 15 minutes"
            assert self.min_capacity >= 1.0, "mFRR minimum capacity is 1 MW"

        # If no true price was supplied, default to the forecast value
        # (deterministic regime, equivalent to the previous behaviour)
        if self.true_capacity_price is None:
            self.true_capacity_price = self.capacity_price
        if self.true_energy_price is None:
            self.true_energy_price = self.energy_price


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

    def __init__(self, random_seed: Optional[int] = None,
                 flex_price_error_pct: float = 0.05):
        """Initialize with ARERA-compliant default parameters.

        Args:
            random_seed: seed for deterministic activation sampling.
            flex_price_error_pct: half-width of the uniform forecast error on
                flexibility tariffs (capacity and energy). Default 5%. The forecast
                that is visible to MILP/PPO at decision time is true_price * (1 + e)
                with e ~ U[-flex_price_error_pct, +flex_price_error_pct]. The
                realized payment uses the TRUE price. Set to 0.0 to disable the
                stochastic flexibility-price model (e.g. for ablation studies).
        """
        # Set random seed for deterministic behavior
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)

        # Forecast-error bound for flexibility tariffs (see docstring)
        self.flex_price_error_pct = float(flex_price_error_pct)

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
                # INERT metadata: min_duration is carried on the service
                # object but never read by any constraint. The binding sustain
                # window is market_constants.DEFAULT_SUSTAIN.fcr = 0.25 h
                # (SO GL Art. 156). Do not read this value as the model's
                # requirement.
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

            # True settlement prices: deterministic from the hour/day multipliers
            true_capacity = tariff['capacity_base'] * final_multiplier
            true_energy = tariff['energy_base'] * final_multiplier

            # Forecast prices visible to the optimizer at decision time: noisy
            # with a uniform error of half-width flex_price_error_pct around the
            # true value. INDEPENDENT samples for capacity and energy.
            if self.flex_price_error_pct > 0.0:
                e_cap = np.random.uniform(-self.flex_price_error_pct,
                                          +self.flex_price_error_pct)
                e_en = np.random.uniform(-self.flex_price_error_pct,
                                         +self.flex_price_error_pct)
                fc_capacity = true_capacity * (1.0 + e_cap)
                fc_energy = true_energy * (1.0 + e_en)
            else:
                fc_capacity = true_capacity
                fc_energy = true_energy

            # Award probability: hour-of-day pattern reflecting demand for
            # ancillary services on the Italian balancing market.
            # Night hours (1-5): system stable, few bidders compete, award_prob high
            # Peak hours (18-21): system stressed, many bidders compete, award_prob low
            # Mid-day / off-peak: intermediate.
            # FCR (high-value, fastest service) is the most competitive -> lowest awards.
            # mFRR (slowest, lower-value) is the least competitive -> highest awards.
            award_prob = self._compute_award_probability(hour, day_of_week,
                                                          service_type)

            service = FlexibilityService(
                service_type=service_type,
                capacity_price=fc_capacity,        # FORECAST (visible to optimizer)
                energy_price=fc_energy,            # FORECAST (visible to optimizer)
                activation_probability=self.historical_probabilities[service_type],
                response_time=tariff['response_time'],
                min_capacity=tariff['min_capacity'],
                max_duration=tariff['max_duration'],
                min_duration=tariff['min_duration'],
                true_capacity_price=true_capacity,  # TRUE settlement price
                true_energy_price=true_energy,      # TRUE settlement price
                award_probability=award_prob,
            )

            opportunities.append(service)

        return opportunities

    def _compute_award_probability(self, hour: int, day_of_week: int,
                                    service_type: 'ServiceType') -> float:
        """Hourly probability that a bid for `service_type` is accepted.

        Pattern is intentionally simple and reproducible: a piecewise hourly
        profile shifted per service to reflect competitive intensity.

          Hour band       Base level
          ----------      ----------
          00-05 (night)   0.80   (low demand, high acceptance)
          06-09 (ramp)    0.55
          10-15 (midday)  0.65
          16-17 (rise)    0.45
          18-21 (peak)    0.30   (high demand, low acceptance)
          22-23 (settle)  0.60

        Service offset (added to the base level, clipped to [0.05, 0.95]):
          FCR  : -0.15   (most competitive, hardest to win)
          aFRR :  0.00   (reference)
          mFRR : +0.10   (least competitive, easiest to win)

        Weekend factor: +0.05 (slightly easier to win on weekends, less
        industrial load creates less ancillary demand pressure).
        """
        if 0 <= hour <= 5:
            base = 0.80
        elif 6 <= hour <= 9:
            base = 0.55
        elif 10 <= hour <= 15:
            base = 0.65
        elif 16 <= hour <= 17:
            base = 0.45
        elif 18 <= hour <= 21:
            base = 0.30
        else:  # 22-23
            base = 0.60

        offset = {
            ServiceType.FCR:  -0.15,
            ServiceType.AFRR:  0.00,
            ServiceType.MFRR: +0.10,
        }[service_type]

        weekend_bonus = 0.05 if day_of_week >= 5 else 0.0
        return float(np.clip(base + offset + weekend_bonus, 0.05, 0.95))

    def calculate_capacity_revenue(self, service: FlexibilityService, capacity_mw: float) -> float:
        """
        Calculate capacity revenue for providing flexibility service.

        Uses the TRUE settlement capacity price (post-clearing reality).
        The forecast price visible to the optimizer (service.capacity_price)
        is what informed the decision to reserve, but the payment is
        calculated against service.true_capacity_price.
        """
        if capacity_mw < service.min_capacity:
            return 0.0

        true_price = service.true_capacity_price
        if true_price is None:  # backward compatibility
            true_price = service.capacity_price
        return true_price * capacity_mw

    def calculate_activation_revenue(self, service: FlexibilityService, energy_mwh: float) -> float:
        """
        Calculate energy revenue when flexibility service is activated.

        Uses the TRUE settlement energy price.
        """
        if energy_mwh <= 0:
            return 0.0

        true_price = service.true_energy_price
        if true_price is None:  # backward compatibility
            true_price = service.energy_price
        return true_price * energy_mwh
    
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