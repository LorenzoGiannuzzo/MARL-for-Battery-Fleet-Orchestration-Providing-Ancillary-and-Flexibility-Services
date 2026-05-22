# Design Document - Annual PPO vs MILP Comparison System

## Overview

The Annual PPO vs MILP Comparison System is designed to conduct comprehensive year-long performance comparisons between Proximal Policy Optimization (PPO) and Mixed Integer Linear Programming (MILP) algorithms for battery energy storage trading in Italian flexibility markets. The system addresses the computational and data management challenges of annual simulations while providing detailed insights into long-term algorithm performance.

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Annual Comparison Controller                  │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Data Manager  │  │ Simulation Core │  │ Analysis Engine │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Market Data     │  │ Battery Model   │  │ Performance     │ │
│  │ Loader          │  │ Manager         │  │ Tracker         │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ PPO Simulator   │  │ MILP Optimizer  │  │ Report Generator│ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Component Interactions

The system follows a modular architecture where each component has specific responsibilities:

1. **Annual Comparison Controller**: Orchestrates the entire annual simulation
2. **Data Manager**: Handles loading, streaming, and caching of market data
3. **Simulation Core**: Manages the execution of PPO and MILP algorithms
4. **Analysis Engine**: Processes results and generates performance metrics
5. **Market Data Loader**: Interfaces with Italian market data sources
6. **Battery Model Manager**: Handles long-term battery degradation modeling
7. **Performance Tracker**: Monitors and logs performance throughout the year

## Components and Interfaces

### 1. Annual Comparison Controller

**Purpose**: Central orchestrator for annual simulations

**Key Methods**:
```python
class AnnualComparisonController:
    def run_annual_comparison(self, config: AnnualConfig) -> AnnualResults
    def setup_simulation_environment(self, year: int, market_data: MarketData)
    def manage_rolling_horizons(self, horizon_config: HorizonConfig)
    def coordinate_algorithm_execution(self, algorithms: List[Algorithm])
    def handle_checkpoints_and_recovery(self, checkpoint_interval: int)
```

**Interfaces**:
- Input: AnnualConfig (simulation parameters, market settings, algorithm configs)
- Output: AnnualResults (comprehensive annual performance data)

### 2. Scalable Market Data Manager

**Purpose**: Efficient handling of large-scale market data

**Key Methods**:
```python
class ScalableMarketDataManager:
    def load_historical_italian_data(self, year: int) -> MarketDataStream
    def generate_synthetic_seasonal_data(self, base_params: SeasonalParams) -> MarketData
    def stream_hourly_data(self, start_date: datetime, end_date: datetime) -> Iterator[HourlyData]
    def cache_frequently_accessed_data(self, cache_config: CacheConfig)
    def validate_data_completeness(self, data_range: DateRange) -> ValidationReport
```

**Data Sources**:
- GME (Gestore Mercati Energetici) historical prices
- Terna ancillary services data
- Synthetic data generation for missing periods
- Weather data for renewable energy correlation

### 3. Long-Term Battery Degradation Model

**Purpose**: Accurate modeling of battery aging over annual periods

**Key Methods**:
```python
class LongTermBatteryModel:
    def initialize_battery_state(self, initial_params: BatteryParams) -> BatteryState
    def update_degradation_daily(self, usage_profile: DailyUsage) -> DegradationUpdate
    def calculate_seasonal_temperature_effects(self, season: Season) -> TemperatureEffects
    def project_end_of_life_timeline(self, current_soh: float) -> EOLProjection
    def optimize_degradation_aware_strategy(self, horizon: int) -> Strategy
```

**Degradation Models**:
- Calendar aging: time-based capacity fade
- Cycle aging: usage-based degradation
- Temperature effects: seasonal variations
- Power fade: internal resistance increase

### 4. Rolling Horizon MILP Optimizer

**Purpose**: Scalable MILP optimization for annual periods

**Key Methods**:
```python
class RollingHorizonMILPOptimizer:
    def setup_rolling_windows(self, window_size: int, overlap: int) -> WindowConfig
    def optimize_window(self, window_data: WindowData) -> WindowResults
    def implement_warm_start(self, previous_solution: Solution) -> WarmStartData
    def manage_solver_resources(self, solver_config: SolverConfig)
    def handle_infeasible_windows(self, infeasible_data: WindowData) -> FallbackSolution
```

**Optimization Strategies**:
- Fixed window size (24h, 48h, 168h)
- Adaptive window sizing based on market volatility
- Parallel window processing
- Solver warm-starting for efficiency

### 5. Annual Performance Analyzer

**Purpose**: Comprehensive analysis of year-long performance

**Key Methods**:
```python
class AnnualPerformanceAnalyzer:
    def calculate_seasonal_metrics(self, annual_data: AnnualData) -> SeasonalMetrics
    def perform_statistical_significance_tests(self, ppo_data: Data, milp_data: Data) -> StatTests
    def generate_cumulative_performance_curves(self, daily_results: List[DailyResult]) -> Curves
    def analyze_market_condition_sensitivity(self, results: Results, conditions: Conditions) -> Sensitivity
    def calculate_financial_metrics(self, cash_flows: CashFlows) -> FinancialMetrics
```

**Analysis Dimensions**:
- Temporal: daily, weekly, monthly, seasonal, annual
- Financial: profit, ROI, payback period, NPV
- Technical: battery utilization, degradation, efficiency
- Market: price sensitivity, volatility response

## Data Models

### Annual Configuration

```python
@dataclass
class AnnualConfig:
    simulation_year: int
    battery_config: BatteryConfig
    market_config: MarketConfig
    algorithm_configs: Dict[str, AlgorithmConfig]
    optimization_config: OptimizationConfig
    analysis_config: AnalysisConfig
    computational_config: ComputationalConfig
```

### Market Data Structure

```python
@dataclass
class HourlyMarketData:
    timestamp: datetime
    energy_price: float
    flexibility_services: List[FlexibilityService]
    renewable_forecast: RenewableForecast
    demand_forecast: DemandForecast
    temperature: float
    market_conditions: MarketConditions
```

### Annual Results Structure

```python
@dataclass
class AnnualResults:
    simulation_metadata: SimulationMetadata
    algorithm_results: Dict[str, AlgorithmAnnualResults]
    comparative_analysis: ComparativeAnalysis
    seasonal_breakdown: SeasonalBreakdown
    financial_summary: FinancialSummary
    battery_degradation_report: DegradationReport
    performance_statistics: PerformanceStatistics
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Annual Data Completeness
*For any* annual simulation, all 8760 hours of the year must have valid market data and simulation results
**Validates: Requirements 1.1, 1.2**

### Property 2: Battery State Consistency
*For any* battery state transition over the annual period, the State of Charge must remain within physical limits (0-100%) and degradation must be monotonically decreasing
**Validates: Requirements 3.2, 3.5**

### Property 3: Rolling Horizon Continuity
*For any* sequence of rolling horizon optimizations, the battery state at the end of one window must equal the initial state of the next window
**Validates: Requirements 5.2, 5.4**

### Property 4: Financial Calculation Accuracy
*For any* annual simulation, the sum of daily profits must equal the total annual profit within numerical precision limits
**Validates: Requirements 8.3, 8.5**

### Property 5: Seasonal Data Partitioning
*For any* annual dataset, partitioning by seasons must account for all 365 days with no overlaps or gaps
**Validates: Requirements 6.1, 6.2**

### Property 6: Degradation Model Monotonicity
*For any* battery usage pattern over the annual period, the State of Health must never increase (degradation is irreversible)
**Validates: Requirements 3.1, 3.3**

### Property 7: Market Data Temporal Ordering
*For any* loaded market data, timestamps must be in strictly ascending order with no duplicates or gaps
**Validates: Requirements 2.1, 2.2**

### Property 8: Algorithm Performance Comparability
*For any* pair of algorithms (PPO, MILP), they must operate on identical market conditions and battery parameters for fair comparison
**Validates: Requirements 1.1, 9.1**

### Property 9: Resource Utilization Bounds
*For any* annual simulation, memory usage must not exceed configured limits and execution time must be within specified bounds
**Validates: Requirements 7.1, 7.4**

### Property 10: Export Data Integrity
*For any* exported simulation data, the data must be complete, consistent, and verifiable against the original simulation results
**Validates: Requirements 10.1, 10.6**

## Error Handling

### Data Loading Errors
- **Missing Historical Data**: Fallback to synthetic data generation
- **Corrupted Market Data**: Data validation and interpolation
- **Network Timeouts**: Retry mechanisms with exponential backoff

### Computational Errors
- **MILP Solver Failures**: Fallback to heuristic solutions
- **Memory Exhaustion**: Automatic data streaming and garbage collection
- **Numerical Instability**: Precision adjustment and regularization

### Long-Running Simulation Errors
- **System Crashes**: Checkpoint-based recovery
- **Hardware Failures**: Distributed computing fallback
- **Time Limit Exceeded**: Graceful degradation with partial results

## Testing Strategy

### Unit Testing
- Individual component functionality
- Data validation and transformation
- Mathematical model accuracy
- Error handling scenarios

### Integration Testing
- End-to-end annual simulation workflows
- Data pipeline integrity
- Algorithm coordination
- Performance monitoring

### Property-Based Testing
- Each correctness property implemented as property-based test
- Minimum 1000 iterations per property test
- Comprehensive input space coverage
- Edge case and boundary condition testing

### Performance Testing
- Scalability testing with different dataset sizes
- Memory usage profiling
- Execution time benchmarking
- Resource utilization monitoring

### Long-Running Tests
- Full annual simulation validation
- Multi-year historical data testing
- Stress testing with extreme market conditions
- Endurance testing for memory leaks

## Implementation Considerations

### Computational Efficiency
- **Data Streaming**: Process data in chunks to manage memory
- **Parallel Processing**: Utilize multiple cores for independent calculations
- **Caching**: Store frequently accessed data and intermediate results
- **Lazy Loading**: Load data only when needed

### Scalability
- **Horizontal Scaling**: Support distributed computing across multiple machines
- **Vertical Scaling**: Optimize for high-memory, multi-core systems
- **Cloud Integration**: Support for cloud-based computation resources
- **Container Deployment**: Docker-based deployment for reproducibility

### Reliability
- **Checkpointing**: Regular saves of simulation state
- **Recovery Mechanisms**: Automatic restart from last checkpoint
- **Data Validation**: Comprehensive input and output validation
- **Monitoring**: Real-time tracking of simulation health

### Maintainability
- **Modular Design**: Clear separation of concerns
- **Configuration Management**: External configuration files
- **Logging**: Comprehensive logging for debugging and monitoring
- **Documentation**: Detailed API documentation and user guides