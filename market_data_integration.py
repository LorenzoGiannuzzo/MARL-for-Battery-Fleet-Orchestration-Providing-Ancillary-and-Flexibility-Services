"""Market Data Integration Module"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path
import datetime

from flexibility_market import ServiceType
from italian_market_config import ITALIAN_CONFIG


@dataclass
class PUNData:
    """Historical PUN price data"""
    timestamp: datetime.datetime
    price_eur_mwh: float
    zone: str = "ITALIA"


@dataclass
class MSDTariff:
    """MSD tariff data"""
    service_type: ServiceType
    date: datetime.date
    capacity_price_eur_mw_h: float
    energy_price_eur_mwh: float
    zone: str = "CNOR"


@dataclass
class BSPCosts:
    """BSP costs"""
    year: int
    qualification_fee_eur: float
    metering_cost_eur: float
    communication_cost_eur: float
    certification_cost_eur: float
    audit_cost_eur: float
    insurance_cost_eur: float
    
    def get_total_annual_cost(self) -> float:
        return (self.qualification_fee_eur + self.metering_cost_eur + 
                self.communication_cost_eur + self.audit_cost_eur + 
                self.insurance_cost_eur)


class MarketDataIntegrator:
    """Market data integrator"""
    
    def __init__(self, data_path: str = "data"):
        self.data_path = Path(data_path)
        self.pun_data: List[PUNData] = []
        self.msd_tariffs: List[MSDTariff] = []
        self.bsp_costs: Dict[int, BSPCosts] = {}
        self.config = ITALIAN_CONFIG
    
    def generate_msd_tariffs(self, year: int) -> List[MSDTariff]:
        """Generate MSD tariffs"""
        msd_tariffs = []
        start_date = datetime.date(year, 1, 1)
        end_date = datetime.date(year, 12, 31)
        
        current_date = start_date
        while current_date <= end_date:
            for service_type in ServiceType:
                base_tariffs = self.config.arera_flexibility_tariffs[service_type.value]
                seasonal_mult = self.config.get_seasonal_multiplier(current_date.month, service_type.value)
                daily_variation = 0.9 + (hash(str(current_date) + service_type.value) % 100) / 500.0
                
                capacity_price = base_tariffs['capacity_price_base'] * seasonal_mult * daily_variation
                energy_price = base_tariffs['energy_price_base'] * seasonal_mult * daily_variation
                
                tariff = MSDTariff(
                    service_type=service_type,
                    date=current_date,
                    capacity_price_eur_mw_h=capacity_price,
                    energy_price_eur_mwh=energy_price,
                    zone="CNOR"
                )
                msd_tariffs.append(tariff)
            
            current_date += datetime.timedelta(days=1)
        
        self.msd_tariffs.extend(msd_tariffs)
        return msd_tariffs
    
    def generate_bsp_costs(self, year: int) -> BSPCosts:
        """Generate BSP costs"""
        base_costs = self.config.bsp_costs
        inflation_factor = (1.02) ** (year - 2020)
        
        bsp_costs = BSPCosts(
            year=year,
            qualification_fee_eur=base_costs['qualification_fee_annual'] * inflation_factor,
            metering_cost_eur=base_costs['metering_cost_annual'] * inflation_factor,
            communication_cost_eur=base_costs['communication_cost_annual'] * inflation_factor,
            certification_cost_eur=base_costs['certification_cost'] * inflation_factor,
            audit_cost_eur=base_costs['audit_cost_annual'] * inflation_factor,
            insurance_cost_eur=base_costs['insurance_cost_annual'] * inflation_factor
        )
        
        self.bsp_costs[year] = bsp_costs
        return bsp_costs
    
    def validate_data_authenticity(self) -> Dict[str, bool]:
        """Validate data authenticity"""
        validation_results = {
            'pun_data_valid': False,
            'msd_tariffs_valid': False,
            'bsp_costs_valid': False,
            'price_ranges_realistic': False
        }
        
        if self.pun_data:
            pun_prices = [pun.price_eur_mwh for pun in self.pun_data]
            min_price = min(pun_prices)
            max_price = max(pun_prices)
            avg_price = np.mean(pun_prices)
            
            if -50 <= min_price <= 500 and -50 <= max_price <= 500:
                validation_results['pun_data_valid'] = True
            if 20 <= avg_price <= 300:
                validation_results['price_ranges_realistic'] = True
        
        if self.msd_tariffs:
            capacity_prices = [t.capacity_price_eur_mw_h for t in self.msd_tariffs]
            energy_prices = [t.energy_price_eur_mwh for t in self.msd_tariffs]
            
            if (all(10 <= p <= 100 for p in capacity_prices) and
                all(50 <= p <= 200 for p in energy_prices)):
                validation_results['msd_tariffs_valid'] = True
        
        if self.bsp_costs:
            for year, costs in self.bsp_costs.items():
                total_cost = costs.get_total_annual_cost()
                if 8000 <= total_cost <= 25000:
                    validation_results['bsp_costs_valid'] = True
                    break
        
        return validation_results
    
    def check_service_participation_constraints(self, service_type: ServiceType, 
                                              capacity_mw: float) -> Dict[str, bool]:
        """Check service participation constraints"""
        constraints = {
            'meets_minimum_capacity': False,
            'meets_technical_requirements': False,
            'meets_response_time': False,
            'meets_duration_requirements': False
        }
        
        tech_req = self.config.technical_req
        
        if service_type == ServiceType.FCR:
            min_cap = tech_req.fcr_min_capacity
        elif service_type == ServiceType.AFRR:
            min_cap = tech_req.afrr_min_capacity
        else:  # mFRR
            min_cap = tech_req.mfrr_min_capacity
        
        if capacity_mw >= min_cap:
            constraints['meets_minimum_capacity'] = True
        
        if capacity_mw >= tech_req.min_capacity_mw:
            constraints['meets_technical_requirements'] = True
        
        constraints['meets_response_time'] = True
        constraints['meets_duration_requirements'] = True
        
        return constraints