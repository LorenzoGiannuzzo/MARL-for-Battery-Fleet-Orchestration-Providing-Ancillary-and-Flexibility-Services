#!/usr/bin/env python3
"""
Direct runner for the ARERA compliance property test
"""

import sys
import traceback
from test_flexibility_market import TestFlexibilityMarket

def run_property_test():
    """Run the ARERA compliance property test directly"""
    try:
        # Create test instance
        test_instance = TestFlexibilityMarket()
        test_instance.setup_method()
        
        print("Running ARERA compliance property test...")
        
        # Run the property test
        test_instance.test_property_arera_compliance(
            hour=12, day_of_week=1, soc=0.5, available_power=2.0,
            reserved_fcr=0.0, reserved_afrr=0.0, reserved_mfrr=0.0,
            degradation_cycles=100.0
        )
        
        print("✓ Property test passed with sample inputs")
        
        # Run with Hypothesis
        from hypothesis import given, strategies as st, settings
        
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
        def hypothesis_test(hour, day_of_week, soc, available_power, 
                           reserved_fcr, reserved_afrr, reserved_mfrr, degradation_cycles):
            test_instance.test_property_arera_compliance(
                hour, day_of_week, soc, available_power,
                reserved_fcr, reserved_afrr, reserved_mfrr, degradation_cycles
            )
        
        print("Running property test with Hypothesis (100 examples)...")
        hypothesis_test()
        print("✓ All property tests passed!")
        
        return True
        
    except Exception as e:
        print(f"✗ Property test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_property_test()
    sys.exit(0 if success else 1)