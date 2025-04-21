import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from smart_home_predictor import SmartHomePredictor

def main():
    predictor = SmartHomePredictor()
    
    # Example: 3-bedroom apartment with ₹1.5L budget
    configs = predictor.generate_multiple_configurations(
        num_rooms=3,
        budget=150000,
        priorities={
            "energy_efficiency": 8,
            "security": 7,
            "climate_control": 6
        }
    )
    config = configs[0] if configs else None
    
    # View optimized component selection
    print("Top 5 Components:")
    for comp in config['optimization_result']['selected_components'][:5]:
        print(f"{comp['Component_Name']} (Score: {comp['Score']:.2f})")
    
    # Generate energy estimates
    energy_report = predictor.estimate_energy_consumption(
        config['optimization_result']['selected_components']
    )
    print(f"\nEstimated Monthly Consumption: {energy_report['monthly_kwh']:.1f}kWh")
    
    # Generate visualizations
    fig1 = predictor.visualize_component_distribution(config)
    fig2 = predictor.visualize_room_allocation(config)
    fig1.savefig('component_distribution.png')
    fig2.savefig('room_allocation.png')

if __name__ == "__main__":
    main()