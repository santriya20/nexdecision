import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import re
import urllib.parse

# -------------------------------------------------------------------
# 1. PAGE CONFIGURATION & DATABASE INITIALIZATION
# -------------------------------------------------------------------
st.set_page_config(page_title="NexDecision | Multi-Category AI Engine", layout="wide", page_icon="⚡")

DB_NAME = "nexdecision_multiproduct.db"

def get_db_connection():
    return sqlite3.connect(DB_NAME)

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Drop old outdated tables so they recreate with working URLs
    cursor.execute("DROP TABLE IF EXISTS laptops;")
    cursor.execute("DROP TABLE IF EXISTS smartphones;")
    cursor.execute("DROP TABLE IF EXISTS cars;")

    # Create Laptops Table
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
            weight_kg REAL NOT NULL,
            buy_url TEXT NOT NULL
        )
    """)

    # Seed Laptops
    laptop_seeds = [
        ('Lenovo IdeaPad Slim 3', 'Lenovo', 45000, 65, 8, 512, 6.0, 1.65, 'https://www.amazon.in/s?k=Lenovo+IdeaPad+Slim+3'),
        ('HP Pavilion 15', 'HP', 62000, 78, 16, 512, 7.5, 1.75, 'https://www.amazon.in/s?k=HP+Pavilion+15'),
        ('ASUS TUF Gaming F15', 'ASUS', 58000, 85, 8, 512, 4.5, 2.30, 'https://www.amazon.in/s?k=ASUS+TUF+Gaming+F15'),
        ('Apple MacBook Air M1', 'Apple', 75000, 90, 8, 256, 15.0, 1.29, 'https://www.amazon.in/s?k=Apple+MacBook+Air+M1'),
        ('Acer Nitro 5', 'Acer', 68000, 88, 16, 512, 4.0, 2.40, 'https://www.amazon.in/s?k=Acer+Nitro+5'),
        ('Dell Inspiron 14', 'Dell', 52000, 72, 8, 512, 8.0, 1.50, 'https://www.amazon.in/s?k=Dell+Inspiron+14')
    ]
    cursor.executemany("""
        INSERT INTO laptops (name, brand, price, cpu_score, ram_gb, storage_gb, battery_hours, weight_kg, buy_url)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, laptop_seeds)

    # Create Smartphones Table
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
            charging_watts REAL NOT NULL,
            buy_url TEXT NOT NULL
        )
    """)

    # Seed Smartphones
    phone_seeds = [
        ('Redmi Note 13 Pro', 'Xiaomi', 22000, 200, 600000, 8, 128, 5000, 67, 'https://www.amazon.in/s?k=Redmi+Note+13+Pro'),
        ('Realme GT Neo 6 SE', 'Realme', 28000, 50, 850000, 12, 256, 5500, 100, 'https://www.amazon.in/s?k=Realme+GT+Neo+6+SE'),
        ('Samsung Galaxy M54', 'Samsung', 25000, 108, 550000, 8, 128, 6000, 25, 'https://www.amazon.in/s?k=Samsung+Galaxy+M54'),
        ('OnePlus Nord 3', 'OnePlus', 32000, 50, 950000, 16, 256, 5000, 80, 'https://www.amazon.in/s?k=OnePlus+Nord+3'),
        ('iQOO Z9 5G', 'iQOO', 20000, 50, 720000, 8, 128, 5000, 44, 'https://www.amazon.in/s?k=iQOO+Z9+5G')
    ]
    cursor.executemany("""
        INSERT INTO smartphones (name, brand, price, camera_mp, antutu_score, ram_gb, storage_gb, battery_mah, charging_watts, buy_url)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, phone_seeds)

    # Create Cars Table
    cursor.execute("""
        CREATE TABLE cars (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            brand TEXT NOT NULL,
            price REAL NOT NULL,
            mileage_kmpl REAL NOT NULL,
            safety_rating REAL NOT NULL,
            power_bhp REAL NOT NULL,
            boot_space_l REAL NOT NULL,
            buy_url TEXT NOT NULL
        )
    """)

    # Seed Cars
    car_seeds = [
        ('Tata Nexon', 'Tata', 850000, 17.5, 5.0, 118, 382, 'https://www.cardekho.com/carmodels/Tata/Tata_Nexon'),
        ('Maruti Brezza', 'Maruti', 830000, 20.1, 4.0, 102, 328, 'https://www.cardekho.com/carmodels/Maruti/Maruti_Brezza'),
        ('Hyundai Creta', 'Hyundai', 1100000, 17.4, 3.0, 113, 433, 'https://www.cardekho.com/carmodels/Hyundai/Hyundai_Creta'),
        ('Kia Seltos', 'Kia', 1090000, 17.0, 3.0, 113, 433, 'https://www.cardekho.com/carmodels/Kia/Kia_Seltos'),
        ('Mahindra XUV300', 'Mahindra', 790000, 18.2, 5.0, 108, 257, 'https://www.cardekho.com/carmodels/Mahindra/Mahindra_XUV300')
    ]
    cursor.executemany("""
        INSERT INTO cars (name, brand, price, mileage_kmpl, safety_rating, power_bhp, boot_space_l, buy_url)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, car_seeds)

    conn.commit()
    conn.close()

init_db()

# Force clear cache on load to ensure old datasets are purged
st.cache_data.clear()

@st.cache_data
def load_data(table_name):
    conn = get_db_connection()
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

# -------------------------------------------------------------------
# 2. CORE MATHEMATICAL & NLP FUNCTIONS
# -------------------------------------------------------------------
def min_max_scale(series, is_benefit=True):
    min_val, max_val = float(series.min()), float(series.max())
    if max_val == min_val:
        return series.apply(lambda x: 1.0)
    if is_benefit:
        return (series - min_val) / (max_val - min_val)
    return (max_val - series) / (max_val - min_val)

def parse_user_intent_with_ai(prompt):
    prompt_lower = prompt.lower()
    
    # 1. Automatic Category Routing
    detected_cat = "Laptops"
    if any(k in prompt_lower for k in ['phone', 'mobile', 'smartphone', 'camera', 'antutu', 'charging', 'snapdragon']):
        detected_cat = "Smartphones"
    elif any(k in prompt_lower for k in ['car', 'mileage', 'suv', 'boot space', 'vehicle', 'bhp', 'ncap']):
        detected_cat = "Cars"

    # 2. Extract Budget using Regular Expressions
    budget = 100000 if detected_cat != "Cars" else 1500000
    budget_match = re.search(r"(\d+)\s*(k|thousand|lakh|l)?", prompt_lower)
    if budget_match:
        num = int(budget_match.group(1))
        unit = budget_match.group(2)
        if unit in ('k', 'thousand'):
            budget = num * 1000
        elif unit in ('l', 'lakh'):
            budget = num * 100000
        elif num < 200 and detected_cat != "Cars":
            budget = num * 1000

    # 3. Dynamic Weight Assignment
    w1, w2, w3, w4 = 25, 25, 25, 25
    if any(w in prompt_lower for w in ['fast', 'game', 'gaming', 'coding', 'speed', 'performance', 'power']):
        w1 += 35
    if any(w in prompt_lower for w in ['battery', 'charge', 'travel', 'college', 'mileage', 'efficient']):
        w2 += 30
    if any(w in prompt_lower for w in ['light', 'portable', 'thin', 'safety', 'ncap', 'camera']):
        w3 += 25
    if any(w in prompt_lower for w in ['cheap', 'budget', 'affordable', 'value', 'price']):
        w4 += 30

    return {
        "category": detected_cat,
        "budget": float(budget),
        "w1": w1, "w2": w2, "w3": w3, "w4": w4
    }

# -------------------------------------------------------------------
# 3. SESSION STATE MANAGEMENT & HEADER
# -------------------------------------------------------------------
if "selected_category" not in st.session_state:
    st.session_state.selected_category = "Laptops"
if "budget" not in st.session_state:
    st.session_state.budget = 80000.0
if "w1" not in st.session_state:
    st.session_state.w1 = 30
if "w2" not in st.session_state:
    st.session_state.w2 = 25
if "w3" not in st.session_state:
    st.session_state.w3 = 20
if "w4" not in st.session_state:
    st.session_state.w4 = 25

st.title("⚡ NexDecision: Multi-Category AI Decision Engine")
st.write("Eliminate choice fatigue with unbiased mathematical decision utility (MAUT) & machine learning valuation.")

# -------------------------------------------------------------------
# 4. NATURAL LANGUAGE AI SEARCH
# -------------------------------------------------------------------
st.subheader("💬 AI Natural Language Search")
user_prompt = st.text_input("Describe what you need in plain English (e.g., 'Gaming laptop under 65k with high speed'):")
if st.button("✨ Apply AI Search", type="primary") and user_prompt:
    extracted = parse_user_intent_with_ai(user_prompt)
    st.session_state.selected_category = extracted["category"]
    st.session_state.budget = extracted["budget"]
    st.session_state.w1 = extracted["w1"]
    st.session_state.w2 = extracted["w2"]
    st.session_state.w3 = extracted["w3"]
    st.session_state.w4 = extracted["w4"]
    st.rerun()

st.divider()

# -------------------------------------------------------------------
# 5. SIDEBAR CONTROLS
# -------------------------------------------------------------------
category = st.sidebar.selectbox(
    "📁 Select Category",
    ["Laptops", "Smartphones", "Cars"],
    index=["Laptops", "Smartphones", "Cars"].index(st.session_state.selected_category)
)
st.session_state.selected_category = category

max_budget_limit = 150000.0 if category != "Cars" else 2500000.0
budget = st.sidebar.slider(
    "💰 Max Budget (₹)",
    min_value=10000.0,
    max_value=max_budget_limit,
    value=min(float(st.session_state.budget), max_budget_limit),
    step=1000.0 if category != "Cars" else 10000.0
)

st.sidebar.subheader("⚖️ MAUT Metric Weights")
if category == "Laptops":
    w1 = st.sidebar.slider("CPU Performance", 0, 100, st.session_state.w1)
    w2 = st.sidebar.slider("Battery Life", 0, 100, st.session_state.w2)
    w3 = st.sidebar.slider("Portability (Light Weight)", 0, 100, st.session_state.w3)
    w4 = st.sidebar.slider("Price Economy", 0, 100, st.session_state.w4)
elif category == "Smartphones":
    w1 = st.sidebar.slider("AnTuTu Performance", 0, 100, st.session_state.w1)
    w2 = st.sidebar.slider("Battery & Charging", 0, 100, st.session_state.w2)
    w3 = st.sidebar.slider("Camera Resolution", 0, 100, st.session_state.w3)
    w4 = st.sidebar.slider("Price Economy", 0, 100, st.session_state.w4)
else: # Cars
    w1 = st.sidebar.slider("Power (BHP)", 0, 100, st.session_state.w1)
    w2 = st.sidebar.slider("Fuel Efficiency (Mileage)", 0, 100, st.session_state.w2)
    w3 = st.sidebar.slider("Safety Rating", 0, 100, st.session_state.w3)
    w4 = st.sidebar.slider("Price Economy", 0, 100, st.session_state.w4)

# -------------------------------------------------------------------
# 6. DATA PROCESSING & MAUT SCORING
# -------------------------------------------------------------------
table_map = {"Laptops": "laptops", "Smartphones": "smartphones", "Cars": "cars"}
df = load_data(table_map[category])

# Hard Budget Filter
eligible = df[df['price'] <= budget].copy()

if eligible.empty:
    st.error(f"No {category} found within the budget of ₹{budget:,.0f}. Try increasing your budget.")
else:
    # Calculate Normalized Scores
    if category == "Laptops":
        eligible['n1'] = min_max_scale(eligible['cpu_score'], True)
        eligible['n2'] = min_max_scale(eligible['battery_hours'], True)
        eligible['n3'] = min_max_scale(eligible['weight_kg'], False)
        eligible['n4'] = min_max_scale(eligible['price'], False)
        feature_cols = ['cpu_score', 'ram_gb', 'storage_gb', 'battery_hours', 'weight_kg']
    elif category == "Smartphones":
        eligible['n1'] = min_max_scale(eligible['antutu_score'], True)
        eligible['n2'] = min_max_scale(eligible['battery_mah'] * eligible['charging_watts'], True)
        eligible['n3'] = min_max_scale(eligible['camera_mp'], True)
        eligible['n4'] = min_max_scale(eligible['price'], False)
        feature_cols = ['camera_mp', 'antutu_score', 'ram_gb', 'storage_gb', 'battery_mah', 'charging_watts']
    else: # Cars
        eligible['n1'] = min_max_scale(eligible['power_bhp'], True)
        eligible['n2'] = min_max_scale(eligible['mileage_kmpl'], True)
        eligible['n3'] = min_max_scale(eligible['safety_rating'], True)
        eligible['n4'] = min_max_scale(eligible['price'], False)
        feature_cols = ['mileage_kmpl', 'safety_rating', 'power_bhp', 'boot_space_l']

    total_w = w1 + w2 + w3 + w4
    if total_w == 0:
        total_w = 1

    eligible['score'] = (eligible['n1']*w1 + eligible['n2']*w2 + eligible['n3']*w3 + eligible['n4']*w4) / total_w
    ranked = eligible.sort_values(by='score', ascending=False).reset_index(drop=True)

    # ML Price Prediction Model
    model = RandomForestRegressor(n_estimators=50, random_state=42).fit(df[feature_cols], df['price'])
    ranked['predicted_price'] = model.predict(ranked[feature_cols])
    ranked['deal_gap'] = ranked['predicted_price'] - ranked['price']

    # -------------------------------------------------------------------
    # 7. UI RECOMMENDATION CARDS
    # -------------------------------------------------------------------
    st.subheader(f"🏆 Top Ranked Recommendations for {category}")

    for i, r in ranked.iterrows():
        match_pct = round(r['score'] * 100, 1)
        deal_badge = " 💎 ML Value Deal" if r['deal_gap'] > 2000 else ""
        
        with st.expander(f"#{i+1}: {r['name']} — ₹{int(r['price']):,} (Match: {match_pct}%){deal_badge}"):
            col1, col2 = st.columns([2, 1])

            with col1:
                if category == "Laptops":
                    st.write(f"**Specs:** {r['cpu_score']} CPU Score | {int(r['ram_gb'])}GB RAM | {int(r['storage_gb'])}GB SSD | {r['battery_hours']} hrs Battery | {r['weight_kg']} kg")
                    if r['ram_gb'] < 16 and r['cpu_score'] >= 80:
                        st.warning("⚠️ **RAM Bottleneck:** High CPU power constrained by 8GB RAM.")
                elif category == "Smartphones":
                    st.write(f"**Specs:** {r['antutu_score']:,} AnTuTu | {int(r['ram_gb'])}GB RAM | {int(r['storage_gb'])}GB | {r['camera_mp']} MP | {int(r['battery_mah'])} mAh | {int(r['charging_watts'])}W")
                    if r['charging_watts'] <= 25 and r['battery_mah'] >= 5000:
                        st.warning("⚠️ **Slow Charging Mismatch:** Large battery paired with basic <=25W charging.")
                else: # Cars
                    st.write(f"**Specs:** {r['power_bhp']} BHP | {r['mileage_kmpl']} kmpl | {r['safety_rating']} ★ NCAP | {int(r['boot_space_l'])}L Boot Space")
                    if r['safety_rating'] < 4.0:
                        st.warning("⚠️ **Safety Notice:** Vehicle safety rating is below 4.0 Stars.")

                if r['deal_gap'] > 2000:
                    st.success(f"💎 **Value Deal:** Priced ₹{int(r['deal_gap']):,} below algorithmic valuation based on hardware specs.")

            with col2:
                if category != "Cars":
                    # Clean search query using exact product name + RAM spec
                    search_term = f"{r['name']} {int(r['ram_gb'])}GB"
                    encoded_query = urllib.parse.quote(search_term)
                    
                    buy_link = f"https://www.amazon.in/s?k={encoded_query}"
                    btn_label = f"🛒 Find {r['name']} on Amazon"
                else:
                    buy_link = r['buy_url']
                    btn_label = f"🚗 View {r['name']} on CarDekho"

                st.link_button(btn_label, buy_link)

    # -------------------------------------------------------------------
    # 8. VISUAL COMPARISON & DATA EXPORT
    # -------------------------------------------------------------------
    if len(ranked) >= 2:
        st.divider()
        st.subheader("📊 Side-by-Side Top Match Breakdown")
        r1, r2 = ranked.iloc[0], ranked.iloc[1]

        metric_names = ["Performance", "Endurance / Utility", "Portability / Secondary", "Price Economy"]
        scores_1 = [r1['n1']*100, r1['n2']*100, r1['n3']*100, r1['n4']*100]
        scores_2 = [r2['n1']*100, r2['n2']*100, r2['n3']*100, r2['n4']*100]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=metric_names, x=scores_1,
            name=f"🥇 #1 {r1['name'][:18]}",
            orientation='h', marker=dict(color='#2ECC71')
        ))
        fig.add_trace(go.Bar(
            y=metric_names, x=scores_2,
            name=f"🥈 #2 {r2['name'][:18]}",
            orientation='h', marker=dict(color='#3498DB')
        ))
        fig.update_layout(
            barmode='group',
            title="Normalized Spec Comparison Score (0-100%)",
            xaxis=dict(title="Score (%)", range=[0, 100]),
            height=350,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    # CSV Download Button
    st.divider()
    csv_data = ranked.to_csv(index=False)
    st.download_button(
        label="📥 Download Recommendation Report (CSV)",
        data=csv_data,
        file_name=f"nexdecision_{category.lower()}_report.csv",
        mime="text/csv"
    )
