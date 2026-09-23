import streamlit as st
import sqlite3
import pandas as pd

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NexDecision | Smart Product Matcher",
    page_icon="⚡",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 2. DATABASE INITIALIZATION & CONNECTION
# -----------------------------------------------------------------------------
DB_NAME = "nexdecision.db"

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Drop old tables to establish clean schema
    cursor.execute("DROP TABLE IF EXISTS laptops;")
    cursor.execute("DROP TABLE IF EXISTS smartphones;")
    cursor.execute("DROP TABLE IF EXISTS cars;")

    # 1. LAPTOPS TABLE
    cursor.execute("""
        CREATE TABLE laptops (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            brand TEXT NOT NULL,
            price REAL NOT NULL,
            cpu_score REAL NOT NULL,
            ram_gb INTEGER NOT NULL,
            storage_gb INTEGER NOT NULL,
            battery_hours REAL NOT NULL,
            weight_kg REAL NOT NULL
        )
    """)

    laptop_seeds = [
        ('MacBook Air M2', 'Apple', 74900, 92.0, 16, 256, 18.0, 1.24),
        ('MacBook Air M3', 'Apple', 94900, 96.0, 16, 512, 18.0, 1.24),
        ('Lenovo IdeaPad Slim 5', 'Lenovo', 62000, 82.0, 16, 512, 8.5, 1.46),
        ('HP Pavilion Plus 14', 'HP', 72000, 85.0, 16, 512, 7.5, 1.40),
        ('ASUS TUF Gaming F15', 'ASUS', 58000, 85.0, 16, 512, 4.5, 2.30),
        ('Acer Nitro V 15', 'Acer', 64000, 88.0, 16, 512, 4.5, 2.10),
        ('Dell Inspiron 14', 'Dell', 52000, 72.0, 8, 512, 8.0, 1.50)
    ]
    cursor.executemany("""
        INSERT INTO laptops (name, brand, price, cpu_score, ram_gb, storage_gb, battery_hours, weight_kg)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, laptop_seeds)

    # 2. SMARTPHONES TABLE
    cursor.execute("""
        CREATE TABLE smartphones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            brand TEXT NOT NULL,
            price REAL NOT NULL,
            camera_mp REAL NOT NULL,
            antutu_score REAL NOT NULL,
            ram_gb INTEGER NOT NULL,
            storage_gb INTEGER NOT NULL,
            battery_mah REAL NOT NULL,
            charging_watts REAL NOT NULL
        )
    """)

    phone_seeds = [
        ('Redmi Note 13 Pro 5G', 'Xiaomi', 21999, 200, 600000, 8, 128, 5100, 67),
        ('Realme GT 6T 5G', 'Realme', 30999, 50, 1500000, 12, 256, 5500, 120),
        ('OnePlus Nord CE 4', 'OnePlus', 24999, 50, 810000, 8, 128, 5500, 100),
        ('iQOO Z9 5G', 'iQOO', 19999, 50, 730000, 8, 128, 5000, 44),
        ('Samsung Galaxy A35 5G', 'Samsung', 27999, 50, 600000, 8, 128, 5000, 25)
    ]
    cursor.executemany("""
        INSERT INTO smartphones (name, brand, price, camera_mp, antutu_score, ram_gb, storage_gb, battery_mah, charging_watts)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, phone_seeds)

    # 3. CARS TABLE
    cursor.execute("""
        CREATE TABLE cars (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            brand TEXT NOT NULL,
            price REAL NOT NULL,
            mileage_kmpl REAL NOT NULL,
            safety_rating REAL NOT NULL,
            power_bhp REAL NOT NULL,
            boot_space_l REAL NOT NULL
        )
    """)

    car_seeds = [
        ('Tata Nexon', 'Tata', 815000, 17.5, 5.0, 118, 382),
        ('Maruti Brezza', 'Maruti', 834000, 20.1, 4.0, 102, 328),
        ('Hyundai Creta', 'Hyundai', 1099000, 17.4, 3.0, 113, 433),
        ('Kia Seltos', 'Kia', 1089000, 17.0, 3.0, 113, 433),
        ('Mahindra XUV300', 'Mahindra', 799000, 18.2, 5.0, 108, 257)
    ]
    cursor.executemany("""
        INSERT INTO cars (name, brand, price, mileage_kmpl, safety_rating, power_bhp, boot_space_l)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, car_seeds)

    conn.commit()
    conn.close()

# Initialize database on app launch
init_db()

# -----------------------------------------------------------------------------
# 3. MAUT (MULTI-ATTRIBUTE UTILITY THEORY) ALGORITHM
# -----------------------------------------------------------------------------
def calculate_maut_score(df, weights, higher_is_better_flags):
    """Normalizes numerical metrics to [0, 1] range and calculates MAUT weighted score."""
    scores = pd.Series(0.0, index=df.index)
    total_weight = sum(weights.values())
    
    if total_weight == 0:
        return scores

    normalized_weights = {k: v / total_weight for k, v in weights.items()}

    for col, weight in normalized_weights.items():
        min_val = df[col].min()
        max_val = df[col].max()
        
        if max_val == min_val:
            norm = pd.Series(1.0, index=df.index)
        else:
            if higher_is_better_flags.get(col, True):
                norm = (df[col] - min_val) / (max_val - min_val)
            else:
                norm = (max_val - df[col]) / (max_val - min_val)
                
        scores += norm * weight

    return scores * 100

# -----------------------------------------------------------------------------
# 4. STREAMLIT UI & SIDEBAR INPUTS
# -----------------------------------------------------------------------------
st.title("⚡ NexDecision Engine")
st.caption("Multi-Attribute Utility Theory (MAUT) Decision Framework")

sidebar = st.sidebar
sidebar.header("1. Category & Budget")
category = sidebar.selectbox("Choose Category", ["Laptops", "Smartphones", "Cars"])

conn = get_db_connection()

if category == "Laptops":
    df = pd.read_sql_query("SELECT * FROM laptops", conn)
    max_price = sidebar.slider("Max Budget (₹)", 40000, 150000, 80000, step=5000)
    
    sidebar.header("2. Preference Weights")
    w_cpu = sidebar.slider("CPU / Speed Weight", 0, 10, 8)
    w_ram = sidebar.slider("RAM Capacity Weight", 0, 10, 7)
    w_battery = sidebar.slider("Battery Life Weight", 0, 10, 6)
    w_weight = sidebar.slider("Portability (Lightness) Weight", 0, 10, 5)

    weights = {'cpu_score': w_cpu, 'ram_gb': w_ram, 'battery_hours': w_battery, 'weight_kg': w_weight}
    flags = {'cpu_score': True, 'ram_gb': True, 'battery_hours': True, 'weight_kg': False}

elif category == "Smartphones":
    df = pd.read_sql_query("SELECT * FROM smartphones", conn)
    max_price = sidebar.slider("Max Budget (₹)", 10000, 100000, 35000, step=2500)
    
    sidebar.header("2. Preference Weights")
    w_camera = sidebar.slider("Camera Resolution Weight", 0, 10, 7)
    w_antutu = sidebar.slider("Gaming / Processor Weight", 0, 10, 8)
    w_ram = sidebar.slider("RAM Weight", 0, 10, 6)
    w_battery = sidebar.slider("Battery Size Weight", 0, 10, 5)

    weights = {'camera_mp': w_camera, 'antutu_score': w_antutu, 'ram_gb': w_ram, 'battery_mah': w_battery}
    flags = {'camera_mp': True, 'antutu_score': True, 'ram_gb': True, 'battery_mah': True}

else: # Cars
    df = pd.read_sql_query("SELECT * FROM cars", conn)
    max_price = sidebar.slider("Max Budget (₹)", 500000, 2000000, 1000000, step=25000)
    
    sidebar.header("2. Preference Weights")
    w_mileage = sidebar.slider("Fuel Efficiency Weight", 0, 10, 8)
    w_safety = sidebar.slider("Safety Rating Weight", 0, 10, 9)
    w_power = sidebar.slider("Engine Power Weight", 0, 10, 6)

    weights = {'mileage_kmpl': w_mileage, 'safety_rating': w_safety, 'power_bhp': w_power}
    flags = {'mileage_kmpl': True, 'safety_rating': True, 'power_bhp': True}

conn.close()

# -----------------------------------------------------------------------------
# 5. FILTER & SCORE DATA
# -----------------------------------------------------------------------------
filtered_df = df[df['price'] <= max_price].copy()

if filtered_df.empty:
    st.warning(f"No {category.lower()} found under ₹{max_price:,}. Try increasing your budget slider.")
else:
    filtered_df['maut_score'] = calculate_maut_score(filtered_df, weights, flags)
    filtered_df = filtered_df.sort_values(by='maut_score', ascending=False).reset_index(drop=True)

    st.subheader(f"Top {len(filtered_df)} Recommendations under ₹{max_price:,}")

    # -------------------------------------------------------------------------
    # 6. RENDER RESULTS (NO EXTERNAL LINKS)
    # -------------------------------------------------------------------------
    for idx, r in filtered_df.iterrows():
        match_score = r['maut_score']
        
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"### #{idx+1}: {r['brand']} {r['name']} — **₹{int(r['price']):,}**")
                
                # Render spec details
                if category == "Laptops":
                    st.write(f"**Specs:** {r['cpu_score']} CPU Score | {r['ram_gb']}GB RAM | {r['storage_gb']}GB SSD | {r['battery_hours']} hrs Battery | {r['weight_kg']} kg")
                    
                    if r['ram_gb'] < 16 and r['cpu_score'] >= 80:
                        st.warning("⚠️ **RAM Bottleneck Warning:** High CPU processing power constrained by 8GB RAM under heavy multitasking.")
                        
                elif category == "Smartphones":
                    st.write(f"**Specs:** {r['camera_mp']} MP Camera | {r['antutu_score']:,} AnTuTu Score | {r['ram_gb']}GB RAM | {r['storage_gb']}GB Storage | {r['battery_mah']} mAh")
                    
                else: # Cars
                    st.write(f"**Specs:** {r['mileage_kmpl']} Kmpl | {r['safety_rating']}★ Safety | {r['power_bhp']} BHP | {r['boot_space_l']}L Boot Space")

            with col2:
                # Displays match score metric card instead of shopping link
                st.metric(label="Decision Match", value=f"{match_score:.1f}%")

        st.divider()
