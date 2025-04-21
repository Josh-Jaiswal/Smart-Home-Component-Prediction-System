import numpy as np
import pandas as pd
import random
from tqdm import tqdm

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

n_samples = 1000

# Define categories and their associated realistic device names and roles
category_data = {
    "Lighting": {
        "names": ["Smart Bulb", "LED Strip", "Ceiling Light", "Floor Lamp", "Wall Sconce", "Pendant Light", "Outdoor Light", "Desk Lamp", 
                 "Track Light", "Recessed Light", "Under Cabinet Light", "Pathway Light", "Garden Light", "Mood Light", "Color Light"],
        "brands": ["Philips Hue", "LIFX", "Sengled", "Wyze", "Nanoleaf", "GE", "TP-Link", "Sylvania", "Govee", "Yeelight", 
                  "Cree", "Feit Electric", "Kasa", "Wiz", "Tuya", "Meross"],
        "roles": ["Lighting control", "Ambiance management", "Energy-efficient lighting", "Automated lighting", 
                 "Circadian rhythm lighting", "Security lighting", "Decorative lighting", "Task lighting"]
    },
    "Security": {
        "names": ["Smart Lock", "Doorbell Camera", "Motion Sensor", "Window Sensor", "Security Camera", "Alarm System", "Glass Break Detector", 
                 "Smoke Detector", "Flood Sensor", "CO Detector", "Smart Safe", "Garage Door Controller", "Perimeter Sensor", "Facial Recognition Camera"],
        "brands": ["Ring", "Nest", "SimpliSafe", "Arlo", "August", "Eufy", "Schlage", "Yale", "Abode", "ADT", 
                  "Logitech", "Blink", "Kwikset", "Wyze", "Kangaroo", "Reolink"],
        "roles": ["Door security", "Perimeter monitoring", "Intrusion detection", "Safety monitoring", 
                 "Access control", "Emergency detection", "Asset protection", "Remote surveillance"]
    },
    "HVAC": {
        "names": ["Smart Thermostat", "Temperature Sensor", "Air Purifier", "Ceiling Fan", "AC Controller", "Heater", "Dehumidifier", 
                 "Air Quality Monitor", "Ventilation Controller", "Humidifier", "HVAC Zoning System", "Radiant Floor Controller", "Heat Pump Controller"],
        "brands": ["Ecobee", "Nest", "Honeywell", "Sensibo", "Dyson", "Tado", "Emerson", "Daikin", "Carrier", "Lennox", 
                  "Mitsubishi", "LG", "Bosch", "Fujitsu", "Rheem", "Cielo"],
        "roles": ["Temperature regulation", "Climate control", "Air quality management", "Humidity control", 
                 "Energy-efficient heating", "Zoned climate control", "Ventilation management", "Allergen reduction"]
    },
    "Energy Management": {
        "names": ["Smart Plug", "Power Strip", "Energy Monitor", "Smart Switch", "Solar Controller", "Battery System", "EV Charger", 
                 "Smart Meter", "Load Controller", "Energy Gateway", "Consumption Monitor", "Grid Tie Inverter", "Power Optimizer", "Energy Display"],
        "brands": ["TP-Link", "Wemo", "Emporia", "Lutron", "Sense", "Tesla", "Schneider", "Eve", "Leviton", "Kasa", 
                  "SolarEdge", "Enphase", "ChargePoint", "Juicebox", "Curb", "Span"],
        "roles": ["Energy monitoring", "Power management", "Consumption tracking", "Load balancing", 
                 "Renewable integration", "Peak shaving", "Demand response", "Energy optimization"]
    }
}

# Define compatibility options with more realistic combinations
compatibility_options = [
    "HomeKit, Google Home, Matter",
    "HomeKit, Alexa",
    "Google Home, Alexa, Matter",
    "Zigbee, Z-Wave",
    "Zigbee, Matter",
    "Z-Wave, Alexa",
    "Wi-Fi, Bluetooth",
    "Wi-Fi, HomeKit",
    "Wi-Fi, Google Home, Alexa",
    "Thread, Matter",
    "Bluetooth, Matter",
    "Wi-Fi, Thread, Matter",
    "Zigbee, Wi-Fi",
    "Z-Wave, Wi-Fi",
    "Proprietary RF"
]

names = []
categories_list = []
prices = []
efficiency = []
reliability = []
compatibility = []
roles = []

# Generate data for each sample with progress bar
for i in tqdm(range(n_samples), desc="Generating data"):
    # Randomly select a category with weighted distribution
    # Make Security and HVAC slightly more common
    category_weights = [0.2, 0.3, 0.3, 0.2]  # Lighting, Security, HVAC, Energy Management
    category = random.choices(list(category_data.keys()), weights=category_weights)[0]
    categories_list.append(category)
    
    # Select a device name and brand based on the category
    device_name = random.choice(category_data[category]["names"])
    brand = random.choice(category_data[category]["brands"])
    names.append(f"{brand} {device_name}")
    
    # Assign a role that matches the category
    role = random.choice(category_data[category]["roles"])
    roles.append(role)
    
    # Set price ranges based on category and device type with more variation
    base_prices = {
        "Lighting": (800, 6000),
        "Security": (2500, 15000),
        "HVAC": (2000, 12000),
        "Energy Management": (1200, 9000)
    }
    
    # Add some price variation based on specific device types
    price_range = base_prices[category]
    
    # Adjust price based on device name (some devices are more expensive)
    if "Camera" in device_name or "System" in device_name or "Controller" in device_name:
        price = np.random.randint(int(price_range[0] * 1.5), int(price_range[1] * 1.2))
    elif "Sensor" in device_name or "Detector" in device_name:
        price = np.random.randint(int(price_range[0] * 0.8), int(price_range[1] * 0.6))
    else:
        price = np.random.randint(price_range[0], price_range[1])
    
    # Add some premium brand pricing
    premium_brands = ["Philips Hue", "Nest", "Ecobee", "Tesla", "Dyson", "Ring", "August"]
    if brand in premium_brands:
        price = int(price * random.uniform(1.1, 1.3))
        
    prices.append(price)
    
    # Generate efficiency and reliability that correlate with price and category
    # Higher-priced items tend to have better efficiency and reliability
    
    # Normalize price to 0-1 range within its category
    category_min, category_max = base_prices[category]
    price_factor = (price - category_min) / (category_max - category_min)
    
    # Add some randomness but maintain correlation with price
    # Different categories have different baseline efficiency
    category_efficiency_base = {
        "Lighting": 6.0,
        "Security": 6.5,
        "HVAC": 5.5,
        "Energy Management": 7.0
    }
    
    eff_base = category_efficiency_base[category] + (price_factor * 3.5)  # Base efficiency varies by category
    eff_random = np.random.normal(0, 0.4)  # Random adjustment with normal distribution
    eff = min(max(eff_base + eff_random, 5), 10)  # Keep within 5-10 range
    efficiency.append(round(eff, 2))
    
    # Reliability follows similar pattern but with different randomness
    # Premium brands tend to have higher reliability
    rel_boost = 0.5 if brand in premium_brands else 0
    rel_base = 5.5 + (price_factor * 4) + rel_boost  # Base reliability from 5.5-10
    rel_random = np.random.normal(0, 0.3)  # Random adjustment with normal distribution
    rel = min(max(rel_base + rel_random, 5), 10)  # Keep within 5-10 range
    reliability.append(round(rel, 2))
    
    # Assign compatibility with weighted probabilities
    # Newer/premium devices more likely to support multiple protocols
    if price_factor > 0.7:
        compat_weights = [0.3, 0.1, 0.2, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0]  # More multi-protocol support
    else:
        compat_weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]  # Even distribution
        
    compat_index = random.choices(range(len(compatibility_options[:10])), weights=compat_weights)[0]
    compatibility.append(compatibility_options[compat_index])

# Create DataFrame
data = pd.DataFrame({
    "Component_Name": names,
    "Category": categories_list,
    "Price_INR": prices,
    "Efficiency": efficiency,
    "Reliability": reliability,
    "Compatibility": compatibility,
    "Role": roles
})

# Data validation - ensure no missing values
print(f"Missing values check: {data.isnull().sum().sum()} missing values")

# Save to CSV
data.to_csv("synthetic_smart_home_components.csv", index=False)

# Print summary statistics
print(f"Generated synthetic data for {n_samples} smart home components")
print("\nCategory distribution:")
print(data["Category"].value_counts())
print("\nPrice statistics by category:")
print(data.groupby("Category")["Price_INR"].describe())
print("\nSample data:")
print(data.head())


def generate_synthetic_data(num_components=100):
    """Generate synthetic smart home component data"""
    categories = ["Lighting", "Security", "HVAC", "Energy Management"]
    
    # Define component types within each category
    component_types = {
        "Lighting": ["Smart Bulb", "LED Strip", "Ceiling Light", "Floor Lamp", "Table Lamp", "Wall Sconce"],
        "Security": ["Camera", "Motion Sensor", "Door Sensor", "Window Sensor", "Smart Lock", "Doorbell"],
        "HVAC": ["Thermostat", "Temperature Sensor", "Air Purifier", "Fan", "Heater", "AC Controller"],
        "Energy Management": ["Smart Plug", "Power Strip", "Energy Monitor", "Smart Switch", "EV Charger"]
    }
    
    # Define brands for variety
    brands = ["Nanoleaf", "Wyze", "Tapo", "Philips", "Sengled", "Govee", "TP-Link", "Meross", "Feit Electric", 
              "Sylvania", "Yeelight", "Lifx", "Leviton", "Kasa", "GE", "Aqara", "Sonoff", "Tuya"]
    
    # Define compatibility protocols
    protocols = ["Zigbee", "Z-Wave", "Wi-Fi", "Bluetooth", "Matter", "Thread"]
    
    data = []
    
    # Ensure we have a good variety by creating at least 3-5 of each component type
    for category, types in component_types.items():
        for comp_type in types:
            # Create 3-5 variations of each component type
            for _ in range(random.randint(3, 5)):
                brand = random.choice(brands)
                
                # Generate a realistic component name
                component_name = f"{brand} {comp_type}"
                
                # Add model number or variant for uniqueness
                if random.random() > 0.5:
                    component_name += f" {random.choice(['Pro', 'Plus', 'Mini', 'Max', 'Ultra'])}"
                else:
                    component_name += f" {random.choice(['A', 'B', 'C', 'X', 'Y', 'Z'])}{random.randint(1, 9)}"
                
                # Generate price based on category and random variation
                base_price = {
                    "Lighting": random.randint(800, 2500),
                    "Security": random.randint(1500, 4000),
                    "HVAC": random.randint(1200, 5000),
                    "Energy Management": random.randint(700, 3000)
                }[category]
                
                # Add some price variation
                price = int(base_price * random.uniform(0.8, 1.2))
                
                # Generate efficiency and reliability scores
                # Higher-priced items tend to have better scores
                price_factor = min(1.0, price / 5000)  # Normalize price to 0-1 scale
                efficiency = round(random.uniform(5, 8) + price_factor * 2, 1)  # 5-10 scale
                reliability = round(random.uniform(5, 8) + price_factor * 2, 1)  # 5-10 scale
                
                # Generate compatibility
                num_protocols = random.randint(1, 3)
                compatibility = ", ".join(random.sample(protocols, num_protocols))
                
                data.append({
                    "Component_Name": component_name,
                    "Category": category,
                    "Price_INR": price,
                    "Efficiency": efficiency,
                    "Reliability": reliability,
                    "Compatibility": compatibility
                })
    
    # If we need more components to reach the requested number
    while len(data) < num_components:
        category = random.choice(categories)
        comp_type = random.choice(component_types[category])
        brand = random.choice(brands)
        
        component_name = f"{brand} {comp_type} {random.choice(['Special', 'Custom', 'Limited'])} Edition"
        
        base_price = {
            "Lighting": random.randint(800, 2500),
            "Security": random.randint(1500, 4000),
            "HVAC": random.randint(1200, 5000),
            "Energy Management": random.randint(700, 3000)
        }[category]
        
        price = int(base_price * random.uniform(0.8, 1.2))
        price_factor = min(1.0, price / 5000)
        efficiency = round(random.uniform(5, 8) + price_factor * 2, 1)
        reliability = round(random.uniform(5, 8) + price_factor * 2, 1)
        
        num_protocols = random.randint(1, 3)
        compatibility = ", ".join(random.sample(protocols, num_protocols))
        
        data.append({
            "Component_Name": component_name,
            "Category": category,
            "Price_INR": price,
            "Efficiency": efficiency,
            "Reliability": reliability,
            "Compatibility": compatibility
        })
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv("synthetic_smart_home_components.csv", index=False)
    print(f"Generated {len(df)} synthetic components")
    return df