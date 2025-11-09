FOODS_3_180_CA_1_validation
│     │ │   │  │ └─ Dataset split (validation set)
│     │ │   │  └─── Store number (1, 2, 3, or 4)
│     │ │   └────── State (CA, TX, or WI)
│     │ └────────── Item number within dept (180th item)
│     └──────────── Department number (FOODS_3 is dept 3 within FOODS)
└────────────────── Category (FOODS, HOBBIES, or HOUSEHOLD)



Hirarchy 
TOTAL WALMART SALES
│
├── FOODS (Category)
│   ├── FOODS_1 (Dept: Prepared Foods)
│   ├── FOODS_2 (Dept: Snacks)
│   └── FOODS_3 (Dept: Perishables)
│       └── FOODS_3_180 (Item: Specific product, like "Organic Bananas")
│           ├── CA_1 (This product in CA Store 1)
│           ├── CA_2 (Same product in CA Store 2)
│           └── ... (10 stores total)
│
├── HOBBIES (Category)
│   └── ...
│
└── HOUSEHOLD (Category)
    └── ...


Revenue per product = Total sales × Average price
Sort products by revenue (descending)
Calculate cumulative revenue %
A = Top products contributing to 80% revenue
B = Next products contributing to 15% revenue  
C = Remaining products (bottom 5% revenue)


CV = Standard Deviation / Mean
```

---

### **The 9-Cell ABC-XYZ Matrix**

Combining both creates **9 product segments**:
```
        │  X (Stable)  │  Y (Moderate)  │  Z (Erratic)
────────┼──────────────┼────────────────┼──────────────
   A    │   🌟 AX      │      AY        │     AZ
(High $ │ Golden Zone  │  Important     │  Challenge
        │ Best models  │  Good models   │  Best ML/DL
────────┼──────────────┼────────────────┼──────────────
   B    │     BX       │   🎯 BY        │     BZ
(Mid $) │  Standard    │  Most common   │  Tricky
        │  ARIMA ok    │  Prophet/LGBM  │  LGBM
────────┼──────────────┼────────────────┼──────────────
   C    │     CX       │      CY        │  ⚠️ CZ
(Low $) │  Easy        │  Acceptable    │  Discontinue?
        │  Simple avg  │  Basic models  │  No forecast


File relationships 
sales_train_validation.csv (d_1, d_2, ...)
         ↓ JOIN on 'd'
calendar.csv (date, events, SNAP)
         
sales_train_validation.csv (item_id, store_id)
         ↓ JOIN on 'item_id' + 'store_id'
sell_prices.csv (sell_price)
         ↓ JOIN on 'wm_yr_wk' from calendar
