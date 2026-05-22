"""
Extended Action Space for PPO with Flexibility Services
Implements the extended action space from 21 to 31 discrete actions
Maintains compatibility with existing arbitrage actions
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum


class ActionType(Enum):
    """Types of actions in the extended action space"""
    ARBITRAGE = "arbitrage"
    FCR = "fcr"
    AFRR = "afrr"
    MFRR = "mfrr"


@dataclass
class DecodedAction:
    """Decoded action with all components"""
    action_type: ActionType
    arbitrage_power: float      # MW - positive for charge, negative for discharge
    fcr_percentage: float       # Percentage of available capacity for FCR (0.0-1.0)
    afrr_percentage: float      # Percentage of available capacity for aFRR (0.0-1.0)
    mfrr_percentage: float      # Percentage of available capacity for mFRR (0.0-1.0)
    
    def __post_init__(self):
        """Validate action parameters"""
        if not 0.0 <= self.fcr_percentage <= 1.0:
            raise ValueError(f"FCR percentage must be between 0 and 1, got {self.fcr_percentage}")
        if not 0.0 <= self.afrr_percentage <= 1.0:
            raise ValueError(f"aFRR percentage must be between 0 and 1, got {self.afrr_percentage}")
        if not 0.0 <= self.mfrr_percentage <= 1.0:
            raise ValueError(f"mFRR percentage must be between 0 and 1, got {self.mfrr_percentage}")


class ExtendedActionSpace:
    """
    Extended action space for PPO with flexibility services
    
    Action Space Structure (31 total actions):
    - Arbitrage: 21 actions (indices 0-20) from -2MW to +2MW in 0.2MW steps
    - FCR: 4 actions (indices 21-24) for 0%, 25%, 50%, 75% of available capacity
    - aFRR: 3 actions (indices 25-27) for 0%, 50%, 100% of available capacity  
    - mFRR: 3 actions (indices 28-30) for 0%, 50%, 100% of available capacity
    """
    
    def __init__(self, battery_power: float = 2.0):
        """
        Initialize extended action space
        
        Args:
            battery_power: Maximum battery power in MW
        """
        self.battery_power = battery_power
        
        # Action space configuration
        self.n_arbitrage_actions = 21
        self.n_fcr_actions = 4
        self.n_afrr_actions = 3
        self.n_mfrr_actions = 3
        
        self.total_actions = (self.n_arbitrage_actions + self.n_fcr_actions + 
                             self.n_afrr_actions + self.n_mfrr_actions)
        
        # Arbitrage action values (same as original PPO)
        self.arbitrage_values = np.linspace(-battery_power, battery_power, self.n_arbitrage_actions)
        
        # Flexibility service percentages
        self.fcr_percentages = [0.0, 0.25, 0.50, 0.75]
        self.afrr_percentages = [0.0, 0.50, 1.0]
        self.mfrr_percentages = [0.0, 0.50, 1.0]
        
        # Action boundaries for decoding
        self.arbitrage_end = self.n_arbitrage_actions
        self.fcr_end = self.arbitrage_end + self.n_fcr_actions
        self.afrr_end = self.fcr_end + self.n_afrr_actions
        self.mfrr_end = self.afrr_end + self.n_mfrr_actions
        
    def get_action_space_size(self) -> int:
        """Get total number of actions in the extended space"""
        return self.total_actions
    
    def decode_action(self, action_idx: int) -> DecodedAction:
        """
        Decode action index into structured action components
        
        Args:
            action_idx: Action index (0-30)
            
        Returns:
            DecodedAction with all components
        """
        if not 0 <= action_idx < self.total_actions:
            raise ValueError(f"Action index {action_idx} out of range [0, {self.total_actions-1}]")
        
        # Initialize all components
        arbitrage_power = 0.0
        fcr_percentage = 0.0
        afrr_percentage = 0.0
        mfrr_percentage = 0.0
        action_type = ActionType.ARBITRAGE
        
        if action_idx < self.arbitrage_end:
            # Arbitrage action (0-20)
            arbitrage_power = float(self.arbitrage_values[action_idx])
            action_type = ActionType.ARBITRAGE
            
        elif action_idx < self.fcr_end:
            # FCR action (21-24)
            fcr_idx = action_idx - self.arbitrage_end
            fcr_percentage = self.fcr_percentages[fcr_idx]
            action_type = ActionType.FCR
            
        elif action_idx < self.afrr_end:
            # aFRR action (25-27)
            afrr_idx = action_idx - self.fcr_end
            afrr_percentage = self.afrr_percentages[afrr_idx]
            action_type = ActionType.AFRR
            
        else:
            # mFRR action (28-30)
            mfrr_idx = action_idx - self.afrr_end
            mfrr_percentage = self.mfrr_percentages[mfrr_idx]
            action_type = ActionType.MFRR
        
        return DecodedAction(
            action_type=action_type,
            arbitrage_power=arbitrage_power,
            fcr_percentage=fcr_percentage,
            afrr_percentage=afrr_percentage,
            mfrr_percentage=mfrr_percentage
        )
    
    def encode_action(self, decoded_action: DecodedAction) -> int:
        """
        Encode structured action back to action index
        
        Args:
            decoded_action: DecodedAction to encode
            
        Returns:
            Action index (0-30)
        """
        if decoded_action.action_type == ActionType.ARBITRAGE:
            # Find closest arbitrage action
            power_diff = np.abs(self.arbitrage_values - decoded_action.arbitrage_power)
            return int(np.argmin(power_diff))
            
        elif decoded_action.action_type == ActionType.FCR:
            # Find closest FCR percentage
            if decoded_action.fcr_percentage in self.fcr_percentages:
                fcr_idx = self.fcr_percentages.index(decoded_action.fcr_percentage)
                return self.arbitrage_end + fcr_idx
            else:
                # Find closest percentage
                perc_diff = np.abs(np.array(self.fcr_percentages) - decoded_action.fcr_percentage)
                fcr_idx = int(np.argmin(perc_diff))
                return self.arbitrage_end + fcr_idx
                
        elif decoded_action.action_type == ActionType.AFRR:
            # Find closest aFRR percentage
            if decoded_action.afrr_percentage in self.afrr_percentages:
                afrr_idx = self.afrr_percentages.index(decoded_action.afrr_percentage)
                return self.fcr_end + afrr_idx
            else:
                # Find closest percentage
                perc_diff = np.abs(np.array(self.afrr_percentages) - decoded_action.afrr_percentage)
                afrr_idx = int(np.argmin(perc_diff))
                return self.fcr_end + afrr_idx
                
        elif decoded_action.action_type == ActionType.MFRR:
            # Find closest mFRR percentage
            if decoded_action.mfrr_percentage in self.mfrr_percentages:
                mfrr_idx = self.mfrr_percentages.index(decoded_action.mfrr_percentage)
                return self.afrr_end + mfrr_idx
            else:
                # Find closest percentage
                perc_diff = np.abs(np.array(self.mfrr_percentages) - decoded_action.mfrr_percentage)
                mfrr_idx = int(np.argmin(perc_diff))
                return self.afrr_end + mfrr_idx
        
        raise ValueError(f"Unknown action type: {decoded_action.action_type}")
    
    def get_action_description(self, action_idx: int) -> str:
        """
        Get human-readable description of action
        
        Args:
            action_idx: Action index
            
        Returns:
            String description of the action
        """
        decoded = self.decode_action(action_idx)
        
        if decoded.action_type == ActionType.ARBITRAGE:
            direction = "Charge" if decoded.arbitrage_power > 0 else "Discharge"
            return f"{direction} {abs(decoded.arbitrage_power):.1f} MW"
            
        elif decoded.action_type == ActionType.FCR:
            return f"Reserve {decoded.fcr_percentage*100:.0f}% capacity for FCR"
            
        elif decoded.action_type == ActionType.AFRR:
            return f"Reserve {decoded.afrr_percentage*100:.0f}% capacity for aFRR"
            
        elif decoded.action_type == ActionType.MFRR:
            return f"Reserve {decoded.mfrr_percentage*100:.0f}% capacity for mFRR"
        
        return "Unknown action"
    
    def get_backward_compatible_action(self, action_idx: int) -> Optional[int]:
        """
        Convert extended action to backward compatible action (0-20)
        
        Args:
            action_idx: Extended action index (0-30)
            
        Returns:
            Compatible action index (0-20) or None if not arbitrage action
        """
        if action_idx < self.arbitrage_end:
            return action_idx
        return None
    
    def is_arbitrage_action(self, action_idx: int) -> bool:
        """Check if action is an arbitrage action"""
        return action_idx < self.arbitrage_end
    
    def is_flexibility_action(self, action_idx: int) -> bool:
        """Check if action is a flexibility service action"""
        return action_idx >= self.arbitrage_end
    
    def get_action_type(self, action_idx: int) -> ActionType:
        """Get the type of action"""
        decoded = self.decode_action(action_idx)
        return decoded.action_type
    
    def get_all_actions_summary(self) -> Dict[str, List[Tuple[int, str]]]:
        """
        Get summary of all actions grouped by type
        
        Returns:
            Dictionary with action types as keys and list of (index, description) tuples
        """
        summary = {
            "arbitrage": [],
            "fcr": [],
            "afrr": [],
            "mfrr": []
        }
        
        for i in range(self.total_actions):
            action_type = self.get_action_type(i)
            description = self.get_action_description(i)
            
            if action_type == ActionType.ARBITRAGE:
                summary["arbitrage"].append((i, description))
            elif action_type == ActionType.FCR:
                summary["fcr"].append((i, description))
            elif action_type == ActionType.AFRR:
                summary["afrr"].append((i, description))
            elif action_type == ActionType.MFRR:
                summary["mfrr"].append((i, description))
        
        return summary


def create_extended_action_space(battery_power: float = 2.0) -> ExtendedActionSpace:
    """
    Factory function to create extended action space
    
    Args:
        battery_power: Maximum battery power in MW
        
    Returns:
        ExtendedActionSpace instance
    """
    return ExtendedActionSpace(battery_power)


# Example usage and testing
if __name__ == "__main__":
    # Create extended action space
    action_space = create_extended_action_space()
    
    print(f"Total actions: {action_space.get_action_space_size()}")
    print("\nAction space summary:")
    
    summary = action_space.get_all_actions_summary()
    for action_type, actions in summary.items():
        print(f"\n{action_type.upper()} actions:")
        for idx, desc in actions:
            print(f"  {idx:2d}: {desc}")
    
    # Test encoding/decoding
    print("\nTesting encoding/decoding:")
    test_actions = [0, 10, 20, 21, 24, 25, 27, 28, 30]
    
    for action_idx in test_actions:
        decoded = action_space.decode_action(action_idx)
        encoded = action_space.encode_action(decoded)
        description = action_space.get_action_description(action_idx)
        
        print(f"Action {action_idx}: {description}")
        print(f"  Decoded: {decoded}")
        print(f"  Re-encoded: {encoded}")
        print(f"  Match: {action_idx == encoded}")
        print()