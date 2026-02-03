import pandas as pd
import numpy as np
import random

# Set seed for reproducibility for realistic but consistent results
np.random.seed(42)
random.seed(42)

num_samples = 5000
current_year = 2025

# Define Brands and their Models with a "Base New Price" (approx in Won)
brands_info = {
    'Hyundai': {
        'Avante': 22000000, 
        'Sonata': 30000000, 
        'Grandeur': 40000000,
        'Tucson': 28000000,
        'SantaFe': 35000000
    },
    'Kia': {
        'K3': 20000000, 
        'K5': 29000000, 
        'K8': 38000000,
        'Sportage': 27000000,
        'Sorento': 36000000
    },
    'BMW': {
        '3 Series': 60000000, 
        '5 Series': 80000000,
        'X3': 70000000,
        'X5': 100000000
    },
    'Mercedes': {
        'C-Class': 65000000, 
        'E-Class': 85000000,
        'GLC': 75000000
    },
    'Audi': {
        'A4': 55000000, 
        'A6': 78000000,
        'Q5': 72000000
    }
}

data = []

for _ in range(num_samples):
    brand = random.choice(list(brands_info.keys()))
    model = random.choice(list(brands_info[brand].keys()))
    base_price = brands_info[brand][model]

    # Year (2010 - 2024)
    # Give higher probability to newer cars
    year = random.randint(2010, 2024)
    age = current_year - year

    # Mileage: Average 15,000 km/year, but vary it
    # Older cars tend to have more mileage, but not strictly linear
    avg_mileage = age * 15000
    mileage = int(random.normalvariate(avg_mileage, 10000))
    mileage = max(1000, mileage) # Min 1000 km

    # Engine Size: Correlated with model somewhat, but randomized for simplicity here
    # Small cars smaller engine, big cars bigger.
    if base_price < 30000000:
        engine_size = random.choice([1600, 2000])
    elif base_price < 60000000:
        engine_size = random.choice([2000, 2500])
    else:
        engine_size = random.choice([2000, 3000, 3500])

    # Fuel Type
    fuel_type = random.choice(['Gasoline', 'Diesel', 'Hybrid'])
    if fuel_type == 'Hybrid':
        base_price *= 1.1 # Hybrid is usually more expensive

    # Accident History
    accident = random.choices(['No', 'Yes'], weights=[0.85, 0.15])[0]

    # --- Price Calculation Logic ---
    price = base_price

    # 1. Age Depreciation: lose ~7% value per year compounded
    price = price * (0.93 ** age)

    # 2. Mileage Depreciation: lose ~1% per 5,000km
    price = price * (0.99 ** (mileage / 5000))

    # 3. Accident Penalty: lose 20% value
    if accident == 'Yes':
        price *= 0.8

    # 4. Brand Value Retention (Imports might depreciate differently, assume similiar for now or tweak)
    # German cars might have higher repair costs -> faster depreciation in KR used market sometimes, 
    # but let's keep it simple.

    # 5. Random Noise: Market variations (+- 10%)
    price *= random.uniform(0.90, 1.10)

    # Final safeguards
    price = int(price)
    if price < 1000000: # Scrap value floor
        price = random.randint(500000, 1500000)

    data.append([brand, model, year, mileage, engine_size, fuel_type, accident, price])

# Create DataFrame
columns = ['Brand', 'Model', 'Year', 'Mileage', 'Engine_Size', 'Fuel_Type', 'Accident_History', 'Price']
df = pd.DataFrame(data, columns=columns)

# Save to CSV
output_file = 'used_car_prices.csv'
df.to_csv(output_file, index=False)

print(f"Successfully created '{output_file}' with {num_samples} samples.")
print(df.head())
print("\nData Description:")
print(df.describe())
