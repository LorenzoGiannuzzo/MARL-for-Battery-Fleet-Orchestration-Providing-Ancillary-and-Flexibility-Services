"""
Property-based test for revenue calculation accuracy
Task 1.2: Write property test for revenue calculation accuracy
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from flexibility_market import (
    FlexibilityMarket, FlexibilityService, BatteryState, ServiceType
)
from italian_market_config import ITALIAN_CONFIG


class TestRevenueCalculationAccuracy:
    """Property-based test for revenue calculation accuracy"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.market = FlexibilityMarket()
        self.config = ITALIAN_CONFIG
    
    @given(
        service_type=st.sampled_from([ServiceType.FCR, ServiceType.AFRR, ServiceType.MFRR]),
        capacity_price=st.floats(min_value=10.0, max_value=200.0),
        energy_price=st.floats(min_value=50.0, max_value=300.0),
        activation_probability=st.floats(min_value=0.05, max_value=0.8),
        capacity_mw=st.floats(min_value=0.0, max_value=10.0),
        energy_mwh=st.floats(min_value=0.0, max_value=20.0),
        duration_hours=st.integers(min_value=1, max_value=24),
        month=st.integers(min_value=1, max_value=12),
        hour=st.integers(min_value=0, max_value=23)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_revenue_calculation_accuracy(self, service_type, capacity_price, energy_price, 
                                                 activation_probability, capacity_mw, energy_mwh, 
                                                 duration_hours, month, hour):
        """
        Property test: For any combination of services and quantities, the revenue calculation 
        should correctly apply ARERA tariffs, GME settlement rules, and include all applicable 
        costs and fees.
        
        Feature: battery-flexibility-milp-comparison, Property 3: Revenue Calculation Accuracy
        Validates: Requirements 1.5, 5.4, 5.3
        """
        # Create flexibility service with generated parameters
        # Ensure response times comply with ARERA regulations
        if service_type == ServiceType.FCR:
            response_time = 30
            min_capacity = 1.0
        elif service_type == ServiceType.AFRR:
            response_time = 200
            min_capacity = 1.0
        else:  # MFRR
            response_time = 900
            min_capacity = 1.0
        
        service = FlexibilityService(
            service_type=service_type,
            capacity_price=capacity_price,
            energy_price=energy_price,
            activation_probability=activation_probability,
            response_time=response_time,
            min_capacity=min_capacity,
            max_duration=24,
            min_duration=1 if service_type == ServiceType.MFRR else 4
        )
        
        # Property 1: Capacity revenue calculation accuracy
        capacity_revenue = self.market.calculate_capacity_revenue(service, capacity_mw)
        
        if capacity_mw >= service.min_capacity:
            expected_capacity_revenue = service.capacity_price * capacity_mw
            assert abs(capacity_revenue - expected_capacity_revenue) < 0.01, \
                f"Capacity revenue calculation incorrect: expected {expected_capacity_revenue}, got {capacity_revenue}"
        else:
            # Below minimum capacity should return zero revenue
            assert capacity_revenue == 0.0, \
                f"Revenue should be zero for capacity {capacity_mw} below minimum {service.min_capacity}"
        
        # Property 2: Energy revenue calculation accuracy
        energy_revenue = self.market.calculate_activation_revenue(service, energy_mwh)
        
        if energy_mwh > 0:
            expected_energy_revenue = service.energy_price * energy_mwh
            assert abs(energy_revenue - expected_energy_revenue) < 0.01, \
                f"Energy revenue calculation incorrect: expected {expected_energy_revenue}, got {energy_revenue}"
        else:
            # Zero or negative energy should return zero revenue
            assert energy_revenue == 0.0, \
                f"Revenue should be zero for energy {energy_mwh} <= 0"
        
        # Property 3: Expected revenue calculation accuracy (includes probability)
        if capacity_mw >= service.min_capacity:
            revenue_breakdown = self.market.calculate_expected_revenue(service, capacity_mw, duration_hours)
            
            # Verify capacity revenue component
            expected_capacity_total = service.capacity_price * capacity_mw * duration_hours
            assert abs(revenue_breakdown['capacity_revenue'] - expected_capacity_total) < 0.01, \
                f"Expected capacity revenue incorrect: expected {expected_capacity_total}, got {revenue_breakdown['capacity_revenue']}"
            
            # Verify expected energy revenue component (probabilistic)
            expected_energy_mwh = capacity_mw * 0.5 * duration_hours * service.activation_probability
            expected_energy_total = service.energy_price * expected_energy_mwh
            assert abs(revenue_breakdown['expected_energy_revenue'] - expected_energy_total) < 0.01, \
                f"Expected energy revenue incorrect: expected {expected_energy_total}, got {revenue_breakdown['expected_energy_revenue']}"
            
            # Verify total expected revenue
            expected_total = expected_capacity_total + expected_energy_total
            assert abs(revenue_breakdown['total_expected_revenue'] - expected_total) < 0.01, \
                f"Total expected revenue incorrect: expected {expected_total}, got {revenue_breakdown['total_expected_revenue']}"
        
        # Property 4: ARERA tariff application accuracy
        # Test that market-generated services use correct ARERA base tariffs with multipliers
        market_opportunities = self.market.get_flexibility_opportunities(hour, 1)  # Monday
        market_service = next(s for s in market_opportunities if s.service_type == service_type)
        
        # Base tariff should be applied with appropriate multipliers
        base_capacity = self.market.arera_tariffs[service_type]['capacity_base']
        base_energy = self.market.arera_tariffs[service_type]['energy_base']
        
        # Market prices should be within reasonable bounds of base prices (0.5x to 2.0x due to multipliers)
        assert 0.5 * base_capacity <= market_service.capacity_price <= 2.0 * base_capacity, \
            f"Market capacity price {market_service.capacity_price} outside reasonable bounds of base {base_capacity}"
        assert 0.5 * base_energy <= market_service.energy_price <= 2.0 * base_energy, \
            f"Market energy price {market_service.energy_price} outside reasonable bounds of base {base_energy}"
        
        # Property 5: GME settlement rule compliance (hourly granularity)
        # Revenue calculations should be consistent with hourly settlement periods
        hourly_capacity_revenue = self.market.calculate_capacity_revenue(service, capacity_mw)
        multi_hour_capacity_revenue = self.market.calculate_expected_revenue(service, capacity_mw, duration_hours)['capacity_revenue']
        
        expected_multi_hour = hourly_capacity_revenue * duration_hours
        assert abs(multi_hour_capacity_revenue - expected_multi_hour) < 0.01, \
            f"Multi-hour capacity revenue {multi_hour_capacity_revenue} not consistent with hourly rate {hourly_capacity_revenue}"
        
        # Property 6: BSP cost inclusion accuracy
        bsp_costs = self.market.get_bsp_annual_costs()
        expected_bsp_costs = sum(self.market.bsp_costs.values())
        
        assert abs(bsp_costs - expected_bsp_costs) < 0.01, \
            f"BSP cost calculation incorrect: expected {expected_bsp_costs}, got {bsp_costs}"
        assert bsp_costs > 0, "BSP costs should be positive"
        
        # Property 7: Seasonal and hourly multiplier application
        # Test that config-based multipliers are applied correctly
        seasonal_mult = self.config.get_seasonal_multiplier(month, service_type.value)
        hourly_mult = self.config.get_hourly_multiplier(hour, service_type.value)
        
        assert 0.5 <= seasonal_mult <= 1.5, f"Seasonal multiplier {seasonal_mult} outside reasonable bounds"
        assert 0.5 <= hourly_mult <= 2.0, f"Hourly multiplier {hourly_mult} outside reasonable bounds"
        
        # Property 8: Revenue calculation monotonicity
        # Revenue should increase monotonically with capacity and energy (when above minimum)
        if capacity_mw >= service.min_capacity:
            higher_capacity = capacity_mw + 1.0
            higher_capacity_revenue = self.market.calculate_capacity_revenue(service, higher_capacity)
            assert higher_capacity_revenue >= capacity_revenue, \
                f"Revenue should increase with capacity: {capacity_revenue} -> {higher_capacity_revenue}"
        
        if energy_mwh > 0:
            higher_energy = energy_mwh + 1.0
            higher_energy_revenue = self.market.calculate_activation_revenue(service, higher_energy)
            assert higher_energy_revenue >= energy_revenue, \
                f"Revenue should increase with energy: {energy_revenue} -> {higher_energy_revenue}"
        
        # Property 9: Revenue calculation precision and numerical stability
        # Test that calculations maintain precision for small and large values
        small_capacity = 0.001  # Very small capacity
        large_capacity = 1000.0  # Very large capacity
        
        if small_capacity >= service.min_capacity:
            small_revenue = self.market.calculate_capacity_revenue(service, small_capacity)
            expected_small = service.capacity_price * small_capacity
            assert abs(small_revenue - expected_small) < 1e-6, "Precision lost for small values"
        
        large_revenue = self.market.calculate_capacity_revenue(service, large_capacity)
        expected_large = service.capacity_price * large_capacity
        relative_error = abs(large_revenue - expected_large) / expected_large
        assert relative_error < 1e-10, "Precision lost for large values"
        
        # Property 10: Activation probability bounds validation
        # Ensure activation probabilities are within valid bounds and affect expected revenue correctly
        assert 0.0 <= service.activation_probability <= 1.0, \
            f"Activation probability {service.activation_probability} outside [0,1] bounds"
        
        if capacity_mw >= service.min_capacity:
            # Test with zero activation probability
            zero_prob_service = FlexibilityService(
                service_type=service_type,
                capacity_price=capacity_price,
                energy_price=energy_price,
                activation_probability=0.0,
                response_time=response_time,
                min_capacity=min_capacity,
                max_duration=24,
                min_duration=1 if service_type == ServiceType.MFRR else 4
            )
            
            zero_prob_revenue = self.market.calculate_expected_revenue(zero_prob_service, capacity_mw, duration_hours)
            assert zero_prob_revenue['expected_energy_revenue'] == 0.0, \
                "Expected energy revenue should be zero with zero activation probability"
            assert zero_prob_revenue['capacity_revenue'] > 0.0, \
                "Capacity revenue should still be positive with zero activation probability"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])