"""
Property-based test for ARERA compliance
Task 1.1: Write property test for ARERA compliance
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings
from flexibility_market import (
    FlexibilityMarket, FlexibilityService, BatteryState, ServiceType
)


class TestARERACompliance:
    """Property-based test for ARERA compliance"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.market = FlexibilityMarket()
    
    @given(
        hour=st.integers(min_value=0, max_value=23),
        day_of_week=st.integers(min_value=0, max_value=6),
        soc=st.floats(min_value=0.0, max_value=1.0),
        available_power=st.floats(min_value=0.0, max_value=10.0),
        reserved_fcr=st.floats(min_value=0.0, max_value=5.0),
        reserved_afrr=st.floats(min_value=0.0, max_value=5.0),
        reserved_mfrr=st.floats(min_value=0.0, max_value=5.0),
        degradation_cycles=st.floats(min_value=0.0, max_value=1000.0)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_arera_compliance(self, hour, day_of_week, soc, available_power, 
                                     reserved_fcr, reserved_afrr, reserved_mfrr, degradation_cycles):
        """
        Property test: For any flexibility service request, the BESS response should comply 
        with all ARERA technical parameters including capacity limits, duration constraints, 
        and activation procedures.
        
        Feature: battery-flexibility-milp-comparison, Property 2: ARERA Compliance
        Validates: Requirements 1.1
        """
        # Create battery state with generated values
        battery_state = BatteryState(
            soc=soc,
            available_power=available_power,
            reserved_fcr=reserved_fcr,
            reserved_afrr=reserved_afrr,
            reserved_mfrr=reserved_mfrr,
            degradation_cycles=degradation_cycles
        )
        
        # Get flexibility opportunities for the given hour
        opportunities = self.market.get_flexibility_opportunities(hour, day_of_week)
        
        # Test ARERA compliance for all service opportunities
        for service in opportunities:
            # Property 1: Response time compliance
            if service.service_type == ServiceType.FCR:
                assert service.response_time <= 30, f"FCR response time {service.response_time}s exceeds ARERA limit of 30s"
            elif service.service_type == ServiceType.AFRR:
                assert service.response_time <= 200, f"aFRR response time {service.response_time}s exceeds ARERA limit of 200s"
            elif service.service_type == ServiceType.MFRR:
                assert service.response_time <= 900, f"mFRR response time {service.response_time}s exceeds ARERA limit of 900s"
            
            # Property 2: Minimum capacity compliance
            assert service.min_capacity >= 1.0, f"Service {service.service_type} min capacity {service.min_capacity}MW below ARERA minimum of 1MW"
            
            # Property 3: Pricing compliance (must be positive)
            assert service.capacity_price > 0, f"Service {service.service_type} capacity price {service.capacity_price} must be positive"
            assert service.energy_price > 0, f"Service {service.service_type} energy price {service.energy_price} must be positive"
            
            # Property 4: Activation probability bounds
            assert 0 <= service.activation_probability <= 1, f"Service {service.service_type} activation probability {service.activation_probability} must be in [0,1]"
            
            # Property 5: Duration constraints compliance
            assert service.min_duration >= 1, f"Service {service.service_type} min duration {service.min_duration}h must be >= 1h"
            assert service.max_duration >= service.min_duration, f"Service {service.service_type} max duration must be >= min duration"
            
            # Property 6: Service constraint checking compliance
            constraint_result = self.market.check_service_constraints(service, battery_state)
            
            # If constraints are satisfied, verify the logic is correct
            if constraint_result:
                # Must have sufficient available power after reservations
                total_reserved = reserved_fcr + reserved_afrr + reserved_mfrr
                remaining_power = available_power - total_reserved
                assert remaining_power >= service.min_capacity, "Constraint check passed but insufficient power available"
                
                # SOC constraints must be satisfied for the service type
                if service.service_type == ServiceType.FCR:
                    assert 0.2 <= soc <= 0.8, "FCR constraint check passed but SOC outside [0.2, 0.8] range"
                elif service.service_type == ServiceType.AFRR:
                    assert 0.15 <= soc <= 0.85, "aFRR constraint check passed but SOC outside [0.15, 0.85] range"
                elif service.service_type == ServiceType.MFRR:
                    assert 0.1 <= soc <= 0.9, "mFRR constraint check passed but SOC outside [0.1, 0.9] range"
            
            # Property 7: Revenue calculation compliance
            if available_power >= service.min_capacity:
                # Test capacity revenue calculation
                capacity_revenue = self.market.calculate_capacity_revenue(service, service.min_capacity)
                expected_capacity_revenue = service.capacity_price * service.min_capacity
                assert abs(capacity_revenue - expected_capacity_revenue) < 0.01, "Capacity revenue calculation incorrect"
                
                # Test energy revenue calculation
                test_energy = 1.0  # 1 MWh
                energy_revenue = self.market.calculate_activation_revenue(service, test_energy)
                expected_energy_revenue = service.energy_price * test_energy
                assert abs(energy_revenue - expected_energy_revenue) < 0.01, "Energy revenue calculation incorrect"
            
            # Property 8: Below minimum capacity should return zero revenue
            below_min_capacity = service.min_capacity * 0.5
            zero_revenue = self.market.calculate_capacity_revenue(service, below_min_capacity)
            assert zero_revenue == 0.0, "Revenue should be zero for capacity below minimum"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])