"""
Generate realistic retail sales data for testing
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


def generate_sales_data(num_rows=50000, output_path="data/sales_data.csv"):
    """
    Generate realistic retail sales data
    
    Args:
        num_rows: Number of sales records to generate
        output_path: Path to save the CSV file
    """
    
    # Seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Define business entities
    regions = ["North", "South", "East", "West", "Central"]
    categories = ["Electronics", "Clothing", "Home & Garden", "Sports", "Toys", "Books"]
    
    products = {
        "Electronics": ["Laptop", "Smartphone", "Tablet", "Headphones", "Smart Watch"],
        "Clothing": ["T-Shirt", "Jeans", "Jacket", "Shoes", "Dress"],
        "Home & Garden": ["Furniture", "Bedding", "Kitchen Appliances", "Garden Tools", "Decor"],
        "Sports": ["Gym Equipment", "Sports Shoes", "Yoga Mat", "Bicycle", "Tennis Racket"],
        "Toys": ["Action Figures", "Board Games", "Dolls", "Building Blocks", "Puzzles"],
        "Books": ["Fiction", "Non-Fiction", "Children's Books", "Comics", "Textbooks"]
    }
    
    channels = ["Online", "In-Store", "Mobile App"]
    payment_methods = ["Credit Card", "Debit Card", "Cash", "Digital Wallet"]
    
    # Date range: 3 years of data
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2023, 12, 31)
    date_range = (end_date - start_date).days
    
    # Generate data
    data = []
    
    print(f"Generating {num_rows:,} sales records...")
    
    for i in range(num_rows):
        # Random date with seasonal trends
        days_offset = random.randint(0, date_range)
        sale_date = start_date + timedelta(days=days_offset)
        month = sale_date.month
        
        # Seasonal factor (higher sales in Nov-Dec)
        seasonal_factor = 1.5 if month in [11, 12] else 1.0
        
        # Select category and product
        category = random.choice(categories)
        product = random.choice(products[category])
        
        # Region
        region = random.choice(regions)
        
        # Quantity (1-10 items per transaction)
        quantity = random.randint(1, 10)
        
        # Base prices by category
        base_prices = {
            "Electronics": (200, 2000),
            "Clothing": (20, 200),
            "Home & Garden": (50, 500),
            "Sports": (30, 300),
            "Toys": (10, 100),
            "Books": (10, 50)
        }
        
        min_price, max_price = base_prices[category]
        unit_price = round(random.uniform(min_price, max_price), 2)
        
        # Calculate revenue
        revenue = round(quantity * unit_price * seasonal_factor, 2)
        
        # Cost (70-85% of revenue)
        cost = round(revenue * random.uniform(0.70, 0.85), 2)
        profit = round(revenue - cost, 2)
        
        # Channel (Online growing over time)
        if sale_date.year == 2021:
            channel = random.choices(channels, weights=[0.3, 0.6, 0.1])[0]
        elif sale_date.year == 2022:
            channel = random.choices(channels, weights=[0.4, 0.5, 0.1])[0]
        else:
            channel = random.choices(channels, weights=[0.5, 0.4, 0.1])[0]
        
        # Customer info
        customer_id = f"CUST{random.randint(10000, 99999)}"
        
        # Transaction
        transaction_id = f"TXN{i+1:08d}"
        
        data.append({
            "transaction_id": transaction_id,
            "date": sale_date.strftime("%Y-%m-%d"),
            "year": sale_date.year,
            "quarter": f"Q{(month-1)//3 + 1}",
            "month": sale_date.strftime("%B"),
            "region": region,
            "category": category,
            "product": product,
            "quantity": quantity,
            "unit_price": unit_price,
            "revenue": revenue,
            "cost": cost,
            "profit": profit,
            "channel": channel,
            "payment_method": random.choice(payment_methods),
            "customer_id": customer_id
        })
        
        if (i + 1) % 10000 == 0:
            print(f"  Generated {i+1:,} records...")
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Successfully generated {num_rows:,} records")
    print(f"📁 Saved to: {output_path}")
    print(f"\nDataset Summary:")
    print(f"  Date Range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Total Revenue: ${df['revenue'].sum():,.2f}")
    print(f"  Total Profit: ${df['profit'].sum():,.2f}")
    print(f"  Regions: {', '.join(df['region'].unique())}")
    print(f"  Categories: {', '.join(df['category'].unique())}")
    print(f"\nSample Records:")
    print(df.head(10).to_string())
    
    return df


if __name__ == "__main__":
    generate_sales_data() 
