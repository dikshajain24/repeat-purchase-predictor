import random, csv
from datetime import datetime, timedelta
random.seed(42)

N_CUSTOMERS = 2000
MAX_ORDERS_PER_CUST = 10
START = datetime(2023, 1, 1)
END   = datetime(2025, 6, 30)

channels = ["online","store","marketplace"]
categories = ["skincare","makeup","fragrance","hair","apparel","footwear","accessories"]

def rand_date(a, b):
    delta = b - a
    return a + timedelta(days=random.randint(0, delta.days))

rows = []
order_id = 1
for cid in range(1, N_CUSTOMERS+1):
    orders = random.randint(1, MAX_ORDERS_PER_CUST)
    dates = sorted([rand_date(START, END) for _ in range(orders)])
    for d in dates:
        qty = random.randint(1, 4)
        price = round(random.uniform(5, 200), 2)
        discount = round(random.choice([0, 0.05, 0.10, 0.15, 0.20]), 2)
        is_return = random.choices([0,1], weights=[0.95, 0.05])[0]
        channel = random.choice(channels)
        category = random.choice(categories)
        rows.append({
            "customer_id": cid,
            "order_id": order_id,
            "order_date": d.strftime("%Y-%m-%d"),
            "qty": qty,
            "price": price,
            "discount": discount,
            "is_return": is_return,
            "channel": channel,
            "category": category
        })
        order_id += 1

with open("data_samples/transactions_sample.csv","w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print("Created data_samples/transactions_sample.csv with", len(rows), "rows")
