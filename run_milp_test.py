#!/usr/bin/env python3
"""
Simple test runner for MILP constraint satisfaction test
"""

import sys
import traceback
from test_milp_constraint_satisfaction import TestMILPConstraintSatisfaction

def main():
    """Run the MILP constraint satisfaction test"""
    try:
        test_instance = TestMILPConstraintSatisfaction()
        print("Running MILP constraint satisfaction test...")
        
        # Run a simple test case
        test_instance.test_milp_solution_satisfies_all_constraints(
            time_horizon=2,
            capacity_mwh=4.0,
            max_power_mw=2.0,
            efficiency=0.95,
            soc_min=0.1,
            soc_max=0.9,
            initial_soc=0.5,
            energy_prices=[50.0, 100.0]
        )
        
        print("✓ Test passed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)