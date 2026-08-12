import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import json
import re
import urllib.parse

# Optional Gemini integration
try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

st.set_page_config(page_title="NexDecision | Multi-Category AI Engine", layout="wide", page_icon="⚡")
st.title("⚡ NexDecision: Multi-Category Decision Engine")

DB_NAME = "nexdecision_multiproduct.db"

def get_db_connection():
    return sqlite3.connect(DB_NAME)

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Laptops Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS laptops (
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
    cursor.execute("SELECT COUNT(*) FROM laptops")
    if cursor.fetchone()[0] == 0:
        cursor.executemany("""
            INSERT INTO laptops (name, brand, price, cpu_score, ram_gb, storage_gb, battery_hours, weight_kg)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            ("Lenovo IdeaPad Slim 3", "Lenovo", 46990, 68, 8, 512, 7.0, 1.62),
            ("HP Pavilion 14 (i5 13th)", "HP", 57990, 82, 16, 512, 8.5, 1.41),
            ("Acer Swift Go 14 (OLED)", "Acer", 61990, 88, 16, 512, 9.5, 1.25),
            ("ASUS TUF Gaming F15", "ASUS", 58990, 90, 16, 512, 4.0, 2.30),
            ("Apple MacBook Air M1", "Apple", 64990, 86, 8, 256, 15.0, 1.29),
            ("Dell Inspiron 3520", "Dell", 49990, 72, 16, 512, 6.0, 1.65),
            ("Lenovo Legion 5", "Lenovo", 74990, 94, 16, 512, 5.0, 2.40),
            ("ASUS Zenbook 14 OLED", "ASUS", 79990, 91, 16, 1024, 12.0, 1.20)
        ])

    # 2. Smartphones Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS smartphones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            brand TEXT NOT NULL,
            price REAL NOT NULL,
            camera_mp REAL NOT NULL,
            antutu_score REAL NOT NULL,
            battery_mah REAL NOT NULL,
            charging_watts REAL NOT NULL
        )
    """)
    cursor.execute("SELECT COUNT(*) FROM smartphones")
    if cursor.fetchone()[0] == 0:
        cursor.executemany("""
            INSERT INTO smartphones (name, brand, price, camera_mp, antutu_score, battery_mah, charging_watts)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [
            ("OnePlus 12R", "OnePlus", 39999, 50, 1300000, 5500, 100),
            ("Samsung Galaxy S23 FE", "Samsung", 41999, 50, 1150000, 4500, 25),
            ("iQOO Neo 9 Pro", "iQOO", 36999, 50, 1400000, 5160, 120),
            ("Pixel 7a", "Google", 37999, 64, 800000, 4385, 18),
            ("Nothing Phone (2)", "Nothing", 36999, 50, 1050000, 4700, 45),
            ("Realme GT 6T", "Realme", 30999, 50, 1250000, 5500, 120)
        ])

    # 3. Cars Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cars (
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
    cursor.execute("SELECT COUNT(*) FROM cars")
    if cursor.fetchone()[0] == 0:
        cursor.executemany("""
            INSERT INTO cars (name, brand, price, mileage_kmpl, safety_rating, power_bhp, boot_space_l)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [
            ("Tata Nexon", "Tata", 815000, 17.4, 5.0, 118, 382),
            ("Hyundai Venue", "Hyundai", 794000, 17.5, 3.0, 82, 350),
            ("Maruti Brezza", "Maruti", 834000, 19.8, 4.0, 101, 328),
            ("Kia Sonet", "Kia", 799000, 18.2, 3.0, 118, 392),
            ("Mahindra XUV 3XO", "Mahindra", 779000, 18.0, 5.0, 110, 364),
            ("Skoda Kushaq", "Skoda", 1099000, 15.8, 5.0, 114, 385)
        ])

    conn.commit()
    conn.close()

def load_data(table_name):
    init_db()
    conn = get_db_connection()
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

def min_max_scale(series, is_benefit=True):
    min_val, max_val = float(series.min()), float(series.max())
    if max_val == min_val:
        return series.apply(lambda x: 1.0)
    if is_benefit:
        return (series - min_val) / (max_val - min_val)
    return (max_val - series) / (max_val - min_val)

def parse_user_intent_with_ai(prompt, api_key=None):
    prompt_lower = prompt.lower()
    
    # 1. Automatic Category Detection
    detected_cat = "Laptops"
    if any(k in prompt_lower for k in ['phone', 'mobile', 'smartphone', 'camera', 'antutu', 'charging', 'snapdragon']):
        detected_cat = "Smartphones"
    elif any(k in prompt_lower for k in ['car', 'mileage', 'suv', 'boot space', 'vehicle', 'bhp', 'ncap']):
        detected_cat = "Cars"

    # 2. Extract Budget
    budget_match = re.search(r"(\d+)\s*(k|thousand|lakh|l)?", prompt_lower)
    if detected_cat == "Smartphones":
        budget = 40000
    elif detected_cat == "Cars":
        budget = 900000
    else:
        budget = 65000

    if budget_match:
        num = int(budget_match.group(1))
        unit = budget_match.group(2)
        if unit in ('k', 'thousand'):
            budget = num * 1000
        elif unit in ('l', 'lakh'):
            budget = num * 100000
        elif num < 200:
            budget = num * 1000

    # 3. Dynamic MAUT Weights
    w1, w2, w3, w4 = 25, 25, 25, 25
    if any(w in prompt_lower for w in ['fast', 'game', 'gaming', 'coding', 'edit', 'speed', 'performance']):
        w1 += 35
    if any(w in prompt_lower for w in ['battery', 'charge', 'charging', 'safety', 'travel', 'college']):
        w2 += 30
    if any(w in prompt_lower for w in ['light', 'portable', 'camera', 'photo', 'power']):
        w3 += 25
    if any(w in prompt_lower for w in ['cheap', 'budget', 'value', 'save', 'money', 'student']):
        w4 += 30

    return {
        "category": detected_cat,
        "budget": budget,
        "w1": w1,
        "w2": w2,
        "w3": w3,
        "w4": w4
    }

# --- SESSION STATE INITIALIZATION ---
if "selected_category" not in st.session_state:
    st.session_state.selected_category = "Laptops"
if "budget" not in st.session_state:
    st.session_state.budget = 70000
if "w1" not in st.session_state:
    st.session_state.w1 = 35
if "w2" not in st.session_state:
    st.session_state.w2 = 30
if "w3" not in st.session_state:
    st.session_state.w3 = 20
if "w4" not in st.session_state:
    st.session_state.w4 = 15

# --- AI SEARCH BOX ---
st.subheader("💬 AI Natural Language Search")
col_ai_input, col_ai_btn = st.columns([4, 1])
with col_ai_input:
    user_prompt = st.text_input(
        "Describe what you need in plain English:",
        placeholder="e.g., Best smartphone for college under 40k with long battery"
    )
with col_ai_btn:
    st.write("")
    st.write("")
    apply_ai = st.button("✨ Apply AI Search", type="primary")

if apply_ai and user_prompt:
    extracted = parse_user_intent_with_ai(user_prompt)
    st.session_state.selected_category = extracted["category"]
    st.session_state.budget = extracted["budget"]
    st.session_state.w1 = extracted["w1"]
    st.session_state.w2 = extracted["w2"]
    st.session_state.w3 = extracted["w3"]
    st.session_state.w4 = extracted["w4"]
    st.success(f"🤖 Switched to **{extracted['category']}** | Budget: ₹{extracted['budget']:,}")
    st.rerun()

st.divider()

# --- SIDEBAR CATEGORY SELECTOR ---
st.sidebar.title("🏷️ Select Category")
category_options = ["Laptops", "Smartphones", "Cars"]
current_idx = category_options.index(st.session_state.selected_category)
category = st.sidebar.selectbox("Product Domain", category_options, index=current_idx)

if category != st.session_state.selected_category:
    st.session_state.selected_category = category
    if category == "Laptops":
        st.session_state.budget = 70000
    elif category == "Smartphones":
        st.session_state.budget = 40000
    else:
        st.session_state.budget = 900000
    st.rerun()

# --- CATEGORY SETUP & SLIDERS ---
if category == "Laptops":
    table_name = "laptops"
    df = load_data(table_name)
    
    st.sidebar.header("🔍 Manual Adjustments")
    budget = st.sidebar.slider("Max Budget (₹)", 30000, 100000, min(max(st.session_state.budget, 30000), 100000), 1000)
    min_ram = st.sidebar.select_slider("Minimum RAM (GB)", options=[0, 8, 16, 32], value=8)
    
    st.sidebar.subheader("🎯 Priority Weights")
    w1 = st.sidebar.slider("CPU Processing Power", 0, 100, st.session_state.w1)
    w2 = st.sidebar.slider("Battery Endurance", 0, 100, st.session_state.w2)
    w3 = st.sidebar.slider("Lightweight Portability", 0, 100, st.session_state.w3)
    w4 = st.sidebar.slider("Value for Money", 0, 100, st.session_state.w4)

    weights = {"CPU Performance": w1, "Battery Life": w2, "Portability": w3, "Value for Money": w4}
    total_w = sum(weights.values()) or 1

    eligible = df[(df['price'] <= budget) & (df['ram_gb'] >= min_ram)].copy()
    feature_cols = ['cpu_score', 'ram_gb', 'battery_hours', 'weight_kg']

    if not eligible.empty:
        eligible['n1'] = min_max_scale(eligible['cpu_score'], True)
        eligible['n2'] = min_max_scale(eligible['battery_hours'], True)
        eligible['n3'] = min_max_scale(eligible['weight_kg'], False)
        eligible['n4'] = min_max_scale(eligible['price'], False)
        eligible['score'] = (eligible['n1']*w1 + eligible['n2']*w2 + eligible['n3']*w3 + eligible['n4']*w4) / total_w
        metric_labels = ['CPU Power', 'Battery Life', 'Portability', 'Value for Money']

elif category == "Smartphones":
    table_name = "smartphones"
    df = load_data(table_name)

    st.sidebar.header("🔍 Manual Adjustments")
    budget = st.sidebar.slider("Max Budget (₹)", 20000, 60000, min(max(st.session_state.budget, 20000), 60000), 1000)
    min_bat = st.sidebar.slider("Minimum Battery (mAh)", 4000, 6000, 4500, 100)

    st.sidebar.subheader("🎯 Priority Weights")
    w1 = st.sidebar.slider("Processor Speed (Antutu)", 0, 100, st.session_state.w1)
    w2 = st.sidebar.slider("Fast Charging Speed", 0, 100, st.session_state.w2)
    w3 = st.sidebar.slider("Camera Resolution", 0, 100, st.session_state.w3)
    w4 = st.sidebar.slider("Value for Money", 0, 100, st.session_state.w4)

    weights = {"Antutu Performance": w1, "Fast Charging": w2, "Camera": w3, "Value for Money": w4}
    total_w = sum(weights.values()) or 1

    eligible = df[(df['price'] <= budget) & (df['battery_mah'] >= min_bat)].copy()
    feature_cols = ['antutu_score', 'battery_mah', 'charging_watts', 'camera_mp']

    if not eligible.empty:
        eligible['n1'] = min_max_scale(eligible['antutu_score'], True)
        eligible['n2'] = min_max_scale(eligible['charging_watts'], True)
        eligible['n3'] = min_max_scale(eligible['camera_mp'], True)
        eligible['n4'] = min_max_scale(eligible['price'], False)
        eligible['score'] = (eligible['n1']*w1 + eligible['n2']*w2 + eligible['n3']*w3 + eligible['n4']*w4) / total_w
        metric_labels = ['Processor Speed', 'Fast Charging', 'Camera Quality', 'Value for Money']

else:  # Cars
    table_name = "cars"
    df = load_data(table_name)

    st.sidebar.header("🔍 Manual Adjustments")
    budget = st.sidebar.slider("Max Budget (₹)", 700000, 1500000, min(max(st.session_state.budget, 700000), 1500000), 25000)
    min_safety = st.sidebar.select_slider("Minimum Safety Stars (NCAP)", options=[1, 2, 3, 4, 5], value=3)

    st.sidebar.subheader("🎯 Priority Weights")
    w1 = st.sidebar.slider("Fuel Efficiency / Mileage", 0, 100, st.session_state.w1)
    w2 = st.sidebar.slider("Safety Rating", 0, 100, st.session_state.w2)
    w3 = st.sidebar.slider("Engine Power (BHP)", 0, 100, st.session_state.w3)
    w4 = st.sidebar.slider("Price Economy", 0, 100, st.session_state.w4)

    weights = {"Mileage": w1, "Safety": w2, "Engine Power": w3, "Value for Money": w4}
    total_w = sum(weights.values()) or 1

    eligible = df[(df['price'] <= budget) & (df['safety_rating'] >= min_safety)].copy()
    feature_cols = ['mileage_kmpl', 'safety_rating', 'power_bhp', 'boot_space_l']

    if not eligible.empty:
        eligible['n1'] = min_max_scale(eligible['mileage_kmpl'], True)
        eligible['n2'] = min_max_scale(eligible['safety_rating'], True)
        eligible['n3'] = min_max_scale(eligible['power_bhp'], True)
        eligible['n4'] = min_max_scale(eligible['price'], False)
        eligible['score'] = (eligible['n1']*w1 + eligible['n2']*w2 + eligible['n3']*w3 + eligible['n4']*w4) / total_w
        metric_labels = ['Fuel Mileage', 'Safety Stars', 'Engine Power', 'Value for Money']

# --- RECOMMENDATION DISPLAY ---
if eligible.empty:
    st.warning(f"No {category} match your current filter criteria. Try expanding your budget.")
else:
    # Machine Learning Pricing Model
    model = RandomForestRegressor(n_estimators=50, random_state=42).fit(df[feature_cols], df['price'])
    eligible['predicted_price'] = model.predict(eligible[feature_cols])
    eligible['deal_gap'] = eligible['predicted_price'] - eligible['price']

    ranked = eligible.sort_values(by="score", ascending=False).reset_index(drop=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader(f"🏆 Top Recommended {category}")
        for i, r in ranked.iterrows():
            with st.expander(f"#{i+1}: {r['name']} — ₹{int(r['price']):,} (Match: {round(r['score']*100, 1)}%)", expanded=(i == 0)):
                if category == "Laptops":
                    st.write(f"⚙️ **Specs:** {r['cpu_score']}/100 CPU | {r['ram_gb']}GB RAM | {r['storage_gb']}GB SSD | 🔋 {r['battery_hours']}h | ⚖️ {r['weight_kg']}kg")
                    if r['ram_gb'] < 16 and r['cpu_score'] >= 80:
                        st.warning("⚠️ **RAM Bottleneck:** Fast processor constrained by 8GB RAM.")
                elif category == "Smartphones":
                    st.write(f"📱 **Specs:** {int(r['antutu_score']):,} Antutu | {r['camera_mp']}MP Camera | {int(r['battery_mah'])}mAh | ⚡ {r['charging_watts']}W Charging")
                    if r['charging_watts'] <= 25 and r['battery_mah'] >= 4500:
                        st.warning("⚠️ **Slow Charging:** Large battery paired with basic <=25W charging speed.")
                elif category == "Cars":
                    st.write(f"🚗 **Specs:** {r['mileage_kmpl']} kmpl | ⭐ {r['safety_rating']} Stars | {r['power_bhp']} BHP | 🧳 {r['boot_space_l']}L Boot Space")
                    if r['safety_rating'] < 4.0:
                        st.warning("⚠️ **Safety Notice:** Vehicle carries a 3-star safety rating.")

                if r['deal_gap'] > 2000:
                    st.success(f"💎 **ML Value Deal:** Listed ₹{int(r['deal_gap']):,} below algorithmic estimate.")

                # Dynamic Store/Purchase Link Generation
                encoded_name = urllib.parse.quote_plus(r['name'])
                if category in ["Laptops", "Smartphones"]:
                    product_url = f"https://www.amazon.in/s?k={encoded_name}"
                    btn_label = f"🛒 View {r['name']} on Amazon"
                else:
                    product_url = f"https://www.cardekho.com/new-cars+{encoded_name.replace('+', '-')}"
                    btn_label = f"🚗 View {r['name']} on CarDekho"

                st.link_button(btn_label, product_url)

        csv_data = ranked.to_csv(index=False)
        st.download_button(f"📥 Export Ranked {category} (CSV)", data=csv_data, file_name=f"{category.lower()}_ranked.csv", mime="text/csv")

    with col2:
        st.subheader("📊 Side-by-Side Breakdown (Top 2)")
        if len(ranked) >= 2:
            r1, r2 = ranked.iloc[0], ranked.iloc[1]
            scores_1 = [round(r1['n1'] * 100, 1), round(r1['n2'] * 100, 1), round(r1['n3'] * 100, 1), round(r1['n4'] * 100, 1)]
            scores_2 = [round(r2['n1'] * 100, 1), round(r2['n2'] * 100, 1), round(r2['n3'] * 100, 1), round(r2['n4'] * 100, 1)]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=metric_labels,
                x=scores_1,
                name=f"🥇 #1 {r1['name'][:16]}",
                orientation='h',
                marker=dict(color='#2ECC71'),
                text=[f"{s}%" for s in scores_1],
                textposition='auto'
            ))
            fig.add_trace(go.Bar(
                y=metric_labels,
                x=scores_2,
                name=f"🥈 #2 {r2['name'][:16]}",
                orientation='h',
                marker=dict(color='#3498DB'),
                text=[f"{s}%" for s in scores_2],
                textposition='auto'
            ))

            fig.update_layout(
                barmode='group',
                xaxis=dict(title="Score (0 - 100%)", range=[0, 115]),
                yaxis=dict(autorange="reversed"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=380,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig, width='stretch')
