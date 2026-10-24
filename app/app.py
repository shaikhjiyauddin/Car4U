import streamlit as st
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from recommendation_system import CarRecommendationSystem
from price_prediction import PricePredictor
import database as db
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Car4U - Intelligent Car Recommendation System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    h1, h2, h3, .main-header, .hero-headline { font-family: 'Space Grotesk', sans-serif; }

    .main-header {
        font-size: 4rem;
        font-weight: 800;
        background: -webkit-linear-gradient(#1f77b4, #00d4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: fadeInDown 1s ease-out;
    }
    .sub-header {
        font-size: 1.6rem;
        color: #a0a0a0;
        text-align: center;
        margin-bottom: 3rem;
        font-style: italic;
    }
    .hero-wrap {
        background:
            radial-gradient(circle at 15% 20%, rgba(0,212,255,0.16) 0%, transparent 45%),
            radial-gradient(circle at 85% 80%, rgba(46,204,113,0.12) 0%, transparent 45%),
            linear-gradient(135deg, rgba(31,119,180,0.10) 0%, rgba(0,0,0,0) 100%);
        padding: 3.5rem 3rem 3rem 3rem;
        border-radius: 28px;
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 2.5rem;
    }
    .hero-kicker {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 16px;
        border-radius: 50px;
        background: rgba(46, 204, 113, 0.12);
        border: 1px solid rgba(46, 204, 113, 0.4);
        color: #2ecc71;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 1.2rem;
    }
    .hero-kicker .dot {
        width: 7px; height: 7px; border-radius: 50%;
        background: #2ecc71; box-shadow: 0 0 8px #2ecc71;
        animation: pulse 1.6s infinite;
    }
    .hero-headline {
        font-size: 3.1rem;
        font-weight: 700;
        line-height: 1.1;
        margin: 0 0 1rem 0;
        max-width: 750px;
    }
    .hero-headline .accent {
        background: -webkit-linear-gradient(#1f77b4, #00d4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-sub {
        font-size: 1.15rem;
        color: #9aa0a6;
        max-width: 620px;
        margin-bottom: 2rem;
        line-height: 1.6;
    }
    .ticker-strip {
        display: flex;
        gap: 14px;
        flex-wrap: wrap;
    }
    .ticker-pill {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 14px;
        padding: 12px 20px;
        min-width: 150px;
    }
    .ticker-pill .n { font-family: 'Space Grotesk', sans-serif; font-size: 1.5rem; font-weight: 700; color: #fff; }
    .ticker-pill .l { font-size: 0.78rem; color: #9aa0a6; text-transform: uppercase; letter-spacing: 0.5px; }
    .feature-card {
        padding: 1.6rem;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.1);
        background: rgba(255,255,255,0.03);
        height: 100%;
        transition: all 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-4px);
        background: rgba(255,255,255,0.06);
        border-color: rgba(0,212,255,0.35);
    }
    .feature-icon {
        width: 44px; height: 44px;
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.4rem;
        background: linear-gradient(135deg, #1f77b4, #00d4ff);
        margin-bottom: 14px;
    }
    .car-card {
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 1.5rem 0;
        background: rgba(255, 255, 255, 0.05); /* Glassmorphism */
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        color: inherit;
        animation: fadeInUp 0.8s ease-out;
    }
    .car-card:hover {
        transform: scale(1.02);
        background: rgba(255, 255, 255, 0.1);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.25);
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .metric-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 50px;
        background: #1f77b4;
        color: white;
        font-size: 0.9rem;
        font-weight: bold;
        margin-right: 10px;
    }
    .status-live {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 5px 14px;
        border-radius: 50px;
        background: rgba(46, 204, 113, 0.15);
        border: 1px solid rgba(46, 204, 113, 0.5);
        color: #2ecc71;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.3px;
    }
    .status-live .dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #2ecc71;
        box-shadow: 0 0 8px #2ecc71;
        animation: pulse 1.6s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.3; }
        100% { opacity: 1; }
    }
    .status-reference {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 5px 14px;
        border-radius: 50px;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.18);
        color: #aaa;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize the recommendation system
@st.cache_resource
def load_system():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'car_model.pkl')
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'used_cars_v2.csv')
    if os.path.exists(model_path):
        try:
            return CarRecommendationSystem.load_model(model_path)
        except Exception as e:
            return CarRecommendationSystem(data_path)
    else:
        return CarRecommendationSystem(data_path)

@st.cache_resource
def load_predictor():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'price_predictor_model.pkl')
    if os.path.exists(model_path):
        try:
            return PricePredictor.load_model(model_path)
        except Exception as e:
            return None
    return None

@st.cache_data
def load_used_cars_data():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'used_cars_v2.csv')
    return pd.read_csv(data_path)

system = load_system()
predictor = load_predictor()
used_cars_df = load_used_cars_data()
db.init_db()

# ---- Marketplace auth (session state) ----
if 'user' not in st.session_state:
    st.session_state.user = None  # dict with id, name, email once logged in

# Main header
st.markdown('<h1 class="main-header">🚗 Car4U</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Intelligent Car Recommendation System</p>', unsafe_allow_html=True)

# Sidebar: Marketplace account (login/signup)
st.sidebar.title("Marketplace Account")
if st.session_state.user:
    st.sidebar.success(f"Signed in as **{st.session_state.user['name']}**")
    if st.sidebar.button("Log out"):
        st.session_state.user = None
        st.rerun()
else:
    auth_tab = st.sidebar.radio("Account", ["Login", "Sign Up"], horizontal=True, label_visibility="collapsed")
    if auth_tab == "Login":
        with st.sidebar.form("login_form"):
            login_email = st.text_input("Email")
            login_password = st.text_input("Password", type="password")
            if st.form_submit_button("Log In"):
                user = db.verify_login(login_email, login_password)
                if user:
                    st.session_state.user = user
                    st.rerun()
                else:
                    st.sidebar.error("Incorrect email or password.")
    else:
        with st.sidebar.form("signup_form"):
            signup_name = st.text_input("Your Name")
            signup_email = st.text_input("Email")
            signup_password = st.text_input("Password (min 6 characters)", type="password")
            if st.form_submit_button("Create Account"):
                ok, msg = db.create_user(signup_name, signup_email, signup_password)
                if ok:
                    st.sidebar.success(msg + " Please log in.")
                else:
                    st.sidebar.error(msg)

st.sidebar.markdown("---")

# Sidebar for navigation
st.sidebar.title("Navigation")
nav_options = ["Home", "Recommendations", "Price Predictor", "Compare",
               "Car Database", "Analytics", "Marketplace", "Sell Your Car", "My Messages"]
if 'nav_override' in st.session_state:
    st.session_state['nav_page'] = st.session_state.pop('nav_override')
page = st.sidebar.radio("Select Page", nav_options, key="nav_page")

if page == "Home":
    pipeline_stats = db.get_pipeline_stats()

    # Hero Section — leads with real, live activity instead of a generic banner
    hero_html = f"""
    <div class="hero-wrap">
        <div class="hero-kicker"><span class="dot"></span>LIVE MARKETPLACE — REAL SELLERS, REAL CHATS</div>
        <div class="hero-headline">Buy and sell used cars,<br>backed by <span class="accent">AI that actually learns</span></div>
        <p class="hero-sub">
            Car4U blends {len(system.df):,} historical market records with real listings from real people —
            get a price prediction grounded in data, then chat directly with the owner. No phone numbers, no guesswork.
        </p>
        <div class="ticker-strip">
            <div class="ticker-pill"><div class="n">{pipeline_stats['total_listings']}</div><div class="l">Live Listings</div></div>
            <div class="ticker-pill"><div class="n">{pipeline_stats['total_users']}</div><div class="l">Registered Sellers</div></div>
            <div class="ticker-pill"><div class="n">{pipeline_stats['total_messages']}</div><div class="l">Buyer–Seller Chats</div></div>
            <div class="ticker-pill"><div class="n">{len(system.df):,}</div><div class="l">Cars in AI Training Set</div></div>
        </div>
    </div>
    """
    hero_html = " ".join(line.strip() for line in hero_html.split("\n") if line.strip())
    st.markdown(hero_html, unsafe_allow_html=True)

    st.markdown("### How Car4U works")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            '<div class="feature-card"><div class="feature-icon">🧠</div>'
            '<h4 style="margin:0 0 8px 0;">Smart Prediction</h4>'
            '<p style="color:#9aa0a6; margin:0;">A regression model trained on real transactions estimates fair prices — and grades its own accuracy after every retrain.</p>'
            '</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(
            '<div class="feature-card"><div class="feature-icon">💬</div>'
            '<h4 style="margin:0 0 8px 0;">Chat, Not Cold Calls</h4>'
            '<p style="color:#9aa0a6; margin:0;">Every live listing connects you straight to the owner in-app. No phone number ever gets published.</p>'
            '</div>', unsafe_allow_html=True)

    with col3:
        st.markdown(
            '<div class="feature-card"><div class="feature-icon">📊</div>'
            '<h4 style="margin:0 0 8px 0;">Transparent Data</h4>'
            '<p style="color:#9aa0a6; margin:0;">Every result is labeled — a real live listing, or historical reference data — so you always know what you\'re looking at.</p>'
            '</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### The dataset behind the AI")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Cars Analyzed", f"{len(system.df):,}")
    with col2:
        st.metric("Brands Covered", len(system.get_all_brands()))
    with col3:
        st.metric("Cities Covered", len(system.get_all_locations()))
    with col4:
        st.metric("Price Range", f"₹ {system.df['price'].min():.2f}L – {system.df['price'].max():.2f}L")

elif page == "Recommendations":
    st.header("Get Personalized Used Car Recommendations")
    st.markdown("### Set Your Preferences")
    
    col1, col2 = st.columns(2)
    with col1:
        budget = st.slider("Budget (₹ Lakhs)", min_value=1, max_value=100, value=15, step=1)
        fuel_type = st.selectbox("Fuel Type", ["All"] + system.get_all_fuel_types())
        rec_transmission = st.selectbox("Transmission", ["All"] + system.get_all_transmissions(), key="rec_trans")
        condition_grade = st.selectbox("Preferred Condition", ["All", "A", "B", "C", "D", "E"])
    
    with col2:
        rec_location = st.selectbox("Location", ["All"] + system.get_all_locations(), key="rec_loc")
        rec_owner_type = st.selectbox("Owner Type", ["All"] + system.get_all_owner_types(), key="rec_owner")
        min_seats = st.slider("Minimum Seats", min_value=2, max_value=10, value=5)
        min_safety = st.slider("Minimum Safety Rating", min_value=1, max_value=5, value=3)
        max_kilometers = st.slider("Max Kilometers Driven", min_value=0, max_value=200000, value=100000, step=5000)
    
    if st.button("Get Recommendations", type="primary"):
        preferences = {
            'budget': budget,
            'fuel_type': fuel_type if fuel_type != "All" else None,
            'transmission': rec_transmission if rec_transmission != "All" else None,
            'condition_grade': condition_grade if condition_grade != "All" else None,
            'location': rec_location if rec_location != "All" else None,
            'owner_type': rec_owner_type if rec_owner_type != "All" else None,
            'min_seats': min_seats,
            'min_safety': min_safety,
            'max_kilometers': max_kilometers
        }
        with st.spinner("Finding the best cars for you..."):
            recommendations = system.recommend_by_preferences(preferences, n_recommendations=5)
        
        if len(recommendations) > 0:
            st.success(f"Found {len(recommendations)} cars matching your preferences!")
            for idx, row in recommendations.iterrows():
                is_live = pd.notna(row.get('listing_ref'))
                live_listing = db.get_listing(int(row['listing_ref'])) if is_live else None
                # A listing_ref can exist but point at a deleted listing — guard against that.
                is_live = is_live and live_listing is not None

                with st.container():
                    safety_stars = "⭐" * int(row.get('safety_rating', 0) if pd.notna(row.get('safety_rating', 0)) else 0)
                    status_badge = (
                        '<span class="status-live"><span class="dot"></span>LIVE LISTING</span>'
                        if is_live else
                        '<span class="status-reference">📊 MARKET REFERENCE DATA</span>'
                    )
                    photo_html = (
                        f'<img src="data:image/jpeg;base64,{live_listing["photo"]}" '
                        f'style="width:100%; max-height:280px; object-fit:cover; border-radius:14px; margin-bottom:16px;">'
                        if is_live and live_listing.get('photo') else ''
                    )
                    card_html = f"""
                    <div class="car-card">
                        <div style="margin-bottom: 12px;">{status_badge}</div>
                        {photo_html}
                        <div style="display: flex; justify-content: space-between; align-items: start;">
                            <h2 style="margin:0; color:#1f77b4;">{row['brand']} {row['model']} <span style="font-size:1.2rem; color:#888;">({row['year']})</span></h2>
                            <div class="metric-badge">₹ {row['price']:.2f} Lakhs</div>
                        </div>
                        <div style="margin: 15px 0;">
                            <span class="metric-badge" style="background:#eee; color:#666;">{row['fuel_type']}</span>
                            <span class="metric-badge" style="background:#eee; color:#666;">{row['transmission']}</span>
                            <span class="metric-badge" style="background:#eee; color:#666;">📍 {row['location']}</span>
                            <span class="metric-badge" style="background:#eee; color:#666;">👤 {row['owner_type']} Owner</span>
                        </div>
                        <hr style="border:0; border-top:1px solid rgba(255,255,255,0.1); margin: 20px 0;">
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                            <div>
                                <p style="margin:5px 0;">⚡ <strong>Power:</strong> {row.get('power_raw', 'N/A')}</p>
                                <p style="margin:5px 0;">🔧 <strong>Engine:</strong> {row.get('engine_raw', 'N/A')}</p>
                                <p style="margin:5px 0;">⛽ <strong>Mileage:</strong> {row.get('mileage_raw', 'N/A')}</p>
                            </div>
                            <div>
                                <p style="margin:5px 0;">🛣️ <strong>Km Driven:</strong> {row['kilometers_driven']:,.0f}</p>
                                <p style="margin:5px 0;">🛡️ <strong>Safety:</strong> {safety_stars}</p>
                                <p style="margin:5px 0;">✨ <strong>Condition:</strong> Grade {row.get('condition_grade', 'N/A')}</p>
                            </div>
                        </div>
                    </div>
                    """
                    card_html = " ".join(line.strip() for line in card_html.split("\n") if line.strip())
                    st.markdown(card_html, unsafe_allow_html=True)

                    action_col1, action_col2 = st.columns([1, 1])
                    with action_col1:
                        if is_live:
                            if st.session_state.user is None:
                                st.caption(f"💬 This is a real listing from {live_listing['owner_name']}. Log in from the sidebar to chat with them.")
                            elif st.session_state.user['id'] == live_listing['owner_id']:
                                st.caption("💬 This is your own Marketplace listing.")
                            else:
                                if st.button(f"💬 Chat with {live_listing['owner_name']}", key=f"rec_chat_{idx}"):
                                    st.session_state['open_chat'] = (live_listing['id'], live_listing['owner_id'])
                                    st.session_state['nav_override'] = "Marketplace"
                                    st.rerun()
                        else:
                            st.caption("This is historical market data for reference — not an active listing, so there's no seller to chat with.")
                    with action_col2:
                        with st.expander(f"🔍 Find cars similar to this {row['brand']} {row['model']}"):
                            similar = system.get_similar_cars(idx, n_similar=5)
                            if len(similar) == 0:
                                st.caption("No similar cars found.")
                            else:
                                for _, srow in similar.iterrows():
                                    sim_pct = srow['similarity'] * 100
                                    s_is_live = pd.notna(srow.get('listing_ref'))
                                    s_tag = "🟢 Live" if s_is_live else "📊 Reference"
                                    st.caption(f"{s_tag} — **{srow['brand']} {srow['model']}** ({srow['year']}) "
                                               f"— ₹{srow['price']:.2f}L — {sim_pct:.0f}% similar")
        else:
            st.warning("No cars match your preferences. Try adjusting your criteria.")

elif page == "Price Predictor":
    st.header("Used Car Price Predictor")
    st.markdown("Use our advanced Linear Regression model trained on real market data to estimate the price of your used car.")
    
    if predictor is None:
        st.error("Price predictor model is not trained yet. Please run `python src/train_predictor.py` first.")
    else:
        st.markdown("### Enter Car Specifications")
        col1, col2 = st.columns(2)
        with col1:
            brands = sorted(list(predictor.brand_to_models.keys()))
            brand = st.selectbox("Brand", brands)
            
            models = sorted(predictor.brand_to_models.get(brand, []))
            model_name = st.selectbox("Model", models)
            
            year = st.number_input("Year", min_value=1990, max_value=2024, value=2015)
            km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=50000, step=1000)
            fuel_type = st.selectbox("Fuel Type", ["Diesel", "Petrol", "CNG", "LPG", "Electric"])
            transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
            owner = st.selectbox("Owner Type", ["First", "Second", "Third", "Fourth & Above"])
            
        with col2:
            location = st.selectbox("Location", ["Mumbai", "Pune", "Chennai", "Coimbatore", "Hyderabad", "Jaipur", "Kochi", "Kolkata", "Delhi", "Bangalore", "Ahmedabad"])
            mileage = st.number_input("Mileage (kmpl)", min_value=0.0, max_value=50.0, value=18.0)
            engine = st.number_input("Engine Size (CC)", min_value=500, max_value=6000, value=1200)
            power = st.number_input("Power (bhp)", min_value=30, max_value=600, value=80)
            seats = st.number_input("Seats", min_value=2, max_value=10, value=5)
            safety_rating = st.slider("Safety Rating", min_value=1.0, max_value=5.0, value=3.0, step=1.0)
            condition = st.selectbox("Condition Grade", ["A", "B", "C", "D", "E"], index=2)
            
        if st.button("Estimate Price", type="primary"):
            input_dict = {
                'Brand': brand, 'Model': model_name, 'Year': year, 'Kilometers_Driven': km_driven,
                'Fuel_Type': fuel_type, 'Transmission': transmission, 'Owner_Type': owner,
                'Location': location, 'Mileage': mileage,
                'Engine': engine, 'Power': power, 'Seats': seats, 
                'Safety_Rating': safety_rating, 'Condition_Grade': condition
            }
            with st.spinner("Calculating..."):
                price = predictor.predict(input_dict)
            st.success(f"Estimated Price: **₹ {price:,.2f} Lakhs**")

elif page == "Compare":
    st.header("Compare Used Cars")
    st.markdown("Select up to 3 cars to compare their specifications side-by-side.")
    
    car_list = [f"{row['brand']} {row['model']} ({row['year']})" for idx, row in system.df.iterrows()]
    selected_cars = st.multiselect("Select Cars", car_list, max_selections=3)
    
    if selected_cars:
        cols = st.columns(len(selected_cars))
        comparison_data = []
        for car_name in selected_cars:
            brand_model_year = car_name.rsplit(" (", 1)
            brand_model = brand_model_year[0]
            year = int(brand_model_year[1].replace(")", ""))
            
            # Match against the DataFrame directly to handle multi-word brands
            match = system.df[
                (system.df['brand'] + ' ' + system.df['model'] == brand_model) &
                (system.df['year'] == year)
            ]
            if len(match) == 0:
                st.warning(f"Could not find car: {car_name}")
                continue
            car_info = match.iloc[0]
            comparison_data.append(car_info)
            
        if comparison_data:
            specs = ['brand', 'model', 'year', 'price', 'fuel_type', 'transmission',
                     'power_raw', 'engine_raw', 'mileage_raw', 'kilometers_driven',
                     'seats', 'safety_rating', 'condition_grade', 'location', 'owner_type']
            comp_df = pd.DataFrame([c[specs] for c in comparison_data]).T
            comp_df.index = ['Brand', 'Model', 'Year', 'Price (₹ Lakhs)', 'Fuel Type', 'Transmission',
                            'Power', 'Engine', 'Mileage', 'Km Driven',
                            'Seats', 'Safety Rating', 'Condition', 'Location', 'Owner Type']
            comp_df.columns = selected_cars
            st.table(comp_df)
            
            st.markdown("### Key Highlights")
            for i, car in enumerate(comparison_data):
                with cols[i]:
                    st.info(f"**{car['brand']} {car['model']}**")
                    st.write(f"💰 Price: ₹ {car['price']:.2f} Lakhs")
                    st.write(f"⚡ Power: {car['power_raw']}")
                    st.write(f"🛡️ Safety: {car['safety_rating']}/5")
    else:
        st.info("Select cars from the dropdown to start comparing.")

elif page == "Car Database":
    st.header("Used Car Database")
    st.markdown("Browse our real market dataset of used cars.")
    
    # Preprocess a bit for display
    display_df = used_cars_df.copy()
    display_df = display_df.dropna(subset=['Price']) # display only priced cars
    
    # Search bar
    search_query = st.text_input("🔍 Search by Name", placeholder="e.g. Maruti, Honda...")
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        locations = sorted(display_df['Location'].dropna().unique().tolist())
        loc_filter = st.multiselect("Location", locations)
    with col2:
        fuels = sorted(display_df['Fuel_Type'].dropna().unique().tolist())
        fuel_filter = st.multiselect("Fuel Type", fuels)
    with col3:
        trans = sorted(display_df['Transmission'].dropna().unique().tolist())
        trans_filter = st.multiselect("Transmission", trans)
    with col4:
        owners = sorted(display_df['Owner_Type'].dropna().unique().tolist())
        owner_filter = st.multiselect("Owner Type", owners)
    
    # Apply filters
    if search_query:
        display_df = display_df[display_df['Name'].str.contains(search_query, case=False, na=False)]
    if loc_filter:
        display_df = display_df[display_df['Location'].isin(loc_filter)]
    if fuel_filter:
        display_df = display_df[display_df['Fuel_Type'].isin(fuel_filter)]
    if trans_filter:
        display_df = display_df[display_df['Transmission'].isin(trans_filter)]
    if owner_filter:
        display_df = display_df[display_df['Owner_Type'].isin(owner_filter)]
    
    st.write(f"Showing {len(display_df)} cars")
    
    # Display cars limited to first 100 for performance
    for idx, row in display_df.head(100).iterrows():
        with st.expander(f"{row['Name']} ({row['Year']}) - ₹ {row['Price']} Lakhs"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Location:** {row['Location']}")
                st.write(f"**Fuel Type:** {row['Fuel_Type']}")
                st.write(f"**Transmission:** {row['Transmission']}")
                st.write(f"**Owner Type:** {row['Owner_Type']}")
                st.write(f"**Safety Rating:** {row.get('Safety_Rating', 'N/A')} ⭐")
                st.write(f"**Condition:** Grade {row.get('Condition_Grade', 'N/A')}")
            with col2:
                st.write(f"**Mileage:** {row['Mileage']}")
                st.write(f"**Engine:** {row['Engine']}")
                st.write(f"**Power:** {row['Power']}")
                st.write(f"**Kilometers Driven:** {row['Kilometers_Driven']:,}")

elif page == "Analytics":
    st.header("Used Car Market Analytics")

    # ---- Live Data Pipeline dashboard ----
    st.subheader("🔴 Live Data Pipeline")
    st.caption("Real, live activity from the Marketplace — this is separate from the historical dataset charted below.")

    pipeline_stats = db.get_pipeline_stats()
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.metric("Registered Users", pipeline_stats['total_users'])
    with p2:
        st.metric("Live Listings", pipeline_stats['total_listings'])
    with p3:
        st.metric("Messages Exchanged", pipeline_stats['total_messages'])
    with p4:
        st.metric("Merged into Training Data", pipeline_stats['exported_to_training'])

    timeline = db.get_listings_timeline()
    if timeline:
        timeline_df = pd.DataFrame(timeline)
        timeline_df['created_at'] = pd.to_datetime(timeline_df['created_at'])
        timeline_df = timeline_df.sort_values('created_at')
        timeline_df['cumulative_listings'] = range(1, len(timeline_df) + 1)

        fig_growth = px.line(timeline_df, x='created_at', y='cumulative_listings',
                              title='Live Listings Growth Over Time', markers=True)
        fig_growth.update_layout(xaxis_title='Date Added', yaxis_title='Total Live Listings (cumulative)')
        st.plotly_chart(fig_growth, use_container_width=True)

        st.markdown("**Most recent live listings:**")
        recent = timeline_df.sort_values('created_at', ascending=False).head(5)
        for _, row in recent.iterrows():
            exported_badge = "✅ merged into training data" if pd.notna(row['exported_at']) else "⏳ pending export"
            st.caption(f"🚗 {row['brand']} {row['model']} — added {row['created_at'].strftime('%d %b %Y, %I:%M %p')} — {exported_badge}")

        if pipeline_stats['pending_export'] > 0:
            st.warning(f"⚠️ {pipeline_stats['pending_export']} live listing(s) haven't been merged into the training dataset yet. "
                       f"Run `python src/export_listings.py` to include them in the next model retrain.")
        else:
            st.success("✅ Every live listing has already been merged into the training dataset.")
    else:
        st.info("No live listings yet. Once people start listing cars via **Sell Your Car**, this dashboard will show real growth over time.")

    st.markdown("---")
    st.subheader("📊 Historical Market Dataset")
    st.caption("These charts are built from the historical reference dataset, which trains the AI models' baseline "
               "intelligence — the autonomous pipeline above blends live Marketplace data into it over time.")

    # Clean data a bit for analytics
    analytics_df = used_cars_df.copy()
    analytics_df = analytics_df.dropna(subset=['Price'])
    
    # Extract numerical engine and power for charts
    def extract_num(val):
        if pd.isna(val): return np.nan
        if isinstance(val, str):
            val_clean = val.strip().lower()
            if 'null' in val_clean or 'nan' in val_clean or not val_clean:
                return np.nan
            for suffix in ['CC', 'bhp', 'kmpl', 'km/kg']:
                if val.endswith(suffix):
                    try:
                        return float(val.replace(suffix, '').strip())
                    except ValueError:
                        return np.nan
            try:
                return float(val)
            except ValueError:
                return np.nan
        try:
            return float(val)
        except (ValueError, TypeError):
            return np.nan

    analytics_df['Engine_Num'] = analytics_df['Engine'].apply(extract_num)
    analytics_df['Power_Num'] = analytics_df['Power'].apply(extract_num)
    analytics_df['Mileage_Num'] = analytics_df['Mileage'].apply(extract_num)
    
    # Extract Brand for brand-wise analysis
    if 'Car_Brand' in analytics_df.columns:
        analytics_df['Brand'] = analytics_df['Car_Brand']
    elif 'Name' in analytics_df.columns:
        analytics_df['Brand'] = analytics_df['Name'].str.split(' ').str[0]
    else:
        analytics_df['Brand'] = 'Unknown'
    
    # Price distribution
    st.subheader("Price Distribution (Lakhs)")
    fig_price = px.histogram(analytics_df, x='Price', nbins=40, title='Used Car Price Distribution')
    st.plotly_chart(fig_price, use_container_width=True)
    
    # Location distribution
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Cars by Location")
        loc_counts = analytics_df['Location'].value_counts()
        fig_loc = px.bar(x=loc_counts.index, y=loc_counts.values, title='Listings per Location')
        st.plotly_chart(fig_loc, use_container_width=True)
    
    with col2:
        st.subheader("Fuel Type Distribution")
        fig_fuel = px.pie(analytics_df, names='Fuel_Type', title='Cars by Fuel Type')
        st.plotly_chart(fig_fuel, use_container_width=True)
    
    # Price by Brand
    st.subheader("Average Price by Top Brands")
    top_brands = analytics_df['Brand'].value_counts().nlargest(15).index
    brand_df = analytics_df[analytics_df['Brand'].isin(top_brands)]
    fig_price_brand = px.box(brand_df, x='Brand', y='Price', title='Price Range by Top Brands')
    st.plotly_chart(fig_price_brand, use_container_width=True)
    
    # Power vs Price
    st.subheader("Power vs Price")
    fig_hp_price = px.scatter(analytics_df, x='Power_Num', y='Price', color='Fuel_Type', 
                              title='Power vs Price (Lakhs)', hover_data=['Name'])
    st.plotly_chart(fig_hp_price, use_container_width=True)
    
    # Correlation Heatmap
    st.subheader("Feature Correlation Heatmap")
    corr_df = analytics_df[['Price', 'Year', 'Kilometers_Driven', 'Mileage_Num', 'Engine_Num', 'Power_Num', 'Seats']].corr()
    fig_corr = px.imshow(corr_df, text_auto='.2f', aspect="auto", 
                         title="Correlation Between Car Specs",
                         color_continuous_scale='RdBu_r')
    st.plotly_chart(fig_corr, use_container_width=True)

elif page == "Marketplace":
    st.header("🏷️ Marketplace — Cars Listed by Real Sellers")
    st.markdown("Browse cars listed by other Car4U users, and chat directly with the owner — no phone number needed.")

    listings = db.get_all_listings()
    if not listings:
        st.info("No listings yet. Be the first — head to **Sell Your Car** in the sidebar!")
    else:
        for listing in listings:
            with st.container():
                photo_html = (
                    f'<img src="data:image/jpeg;base64,{listing["photo"]}" '
                    f'style="width:100%; max-height:280px; object-fit:cover; border-radius:14px; margin-bottom:16px;">'
                    if listing.get('photo') else ''
                )
                card_html = f"""
                <div class="car-card">
                    {photo_html}
                    <div style="display: flex; justify-content: space-between; align-items: start;">
                        <h2 style="margin:0; color:#1f77b4;">{listing['brand']} {listing['model']} <span style="font-size:1.2rem; color:#888;">({listing['year']})</span></h2>
                        <div class="metric-badge">₹ {listing['price']:.2f} Lakhs</div>
                    </div>
                    <div style="margin: 15px 0;">
                        <span class="metric-badge" style="background:#eee; color:#666;">👤 Listed by {listing['owner_name']}</span>
                        <span class="metric-badge" style="background:#eee; color:#666;">📍 {listing.get('location') or 'N/A'}</span>
                        <span class="metric-badge" style="background:#eee; color:#666;">{listing.get('fuel_type') or 'N/A'}</span>
                        <span class="metric-badge" style="background:#eee; color:#666;">{listing.get('transmission') or 'N/A'}</span>
                    </div>
                    <p>🛣️ <strong>Km Driven:</strong> {listing.get('kilometers_driven') or 'N/A'}</p>
                    <p>{listing.get('description') or ''}</p>
                </div>
                """
                card_html = " ".join(line.strip() for line in card_html.split("\n") if line.strip())
                st.markdown(card_html, unsafe_allow_html=True)

                if st.session_state.user is None:
                    st.caption("Log in from the sidebar to chat with the owner.")
                elif st.session_state.user['id'] == listing['owner_id']:
                    st.caption("This is your own listing.")
                else:
                    if st.button(f"💬 Chat with {listing['owner_name']}", key=f"chat_btn_{listing['id']}"):
                        st.session_state['open_chat'] = (listing['id'], listing['owner_id'])
                        st.rerun()

    # Inline chat window, opened via the button above
    if st.session_state.get('open_chat') and st.session_state.user:
        chat_listing_id, chat_other_id = st.session_state['open_chat']
        chat_listing = db.get_listing(chat_listing_id)
        other_user = db.get_user_by_id(chat_other_id)
        if chat_listing and other_user:
            st.markdown("---")
            st.subheader(f"💬 Chat with {other_user['name']} about {chat_listing['brand']} {chat_listing['model']}")

            convo = db.get_conversation(chat_listing_id, st.session_state.user['id'], chat_other_id)
            chat_box = st.container(height=300)
            with chat_box:
                if not convo:
                    st.caption("No messages yet — say hello!")
                for msg in convo:
                    align = "right" if msg['sender_id'] == st.session_state.user['id'] else "left"
                    bubble_color = '#1f77b4' if align == 'right' else '#333'
                    msg_html = (
                        f'<div style="text-align:{align}; margin: 6px 0;">'
                        f'<span style="background:{bubble_color}; color:white; padding: 8px 14px; '
                        f'border-radius: 14px; display:inline-block; max-width:70%;">'
                        f'<strong>{msg["sender_name"]}:</strong> {msg["body"]}</span></div>'
                    )
                    st.markdown(msg_html, unsafe_allow_html=True)

            with st.form("send_message_form", clear_on_submit=True):
                new_message = st.text_input("Type a message", label_visibility="collapsed",
                                             placeholder="Type your message...")
                sent = st.form_submit_button("Send")
                if sent and new_message.strip():
                    db.send_message(chat_listing_id, st.session_state.user['id'], chat_other_id, new_message)
                    st.rerun()

            if st.button("Close chat"):
                del st.session_state['open_chat']
                st.rerun()

elif page == "Sell Your Car":
    st.header("🚘 Sell Your Car")
    if st.session_state.user is None:
        st.warning("Please log in or sign up from the sidebar first — your name will appear as the seller so buyers know who they're chatting with.")
    else:
        st.markdown(f"Listing as **{st.session_state.user['name']}**.")
        with st.form("sell_car_form"):
            col1, col2 = st.columns(2)
            with col1:
                s_brand = st.text_input("Brand", placeholder="e.g. Maruti")
                s_model = st.text_input("Model", placeholder="e.g. Swift VXI")
                s_year = st.number_input("Year", min_value=1990, max_value=2026, value=2018)
                s_price = st.number_input("Asking Price (₹ Lakhs)", min_value=0.1, value=5.0, step=0.1)
            with col2:
                s_fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
                s_transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
                s_km = st.number_input("Kilometers Driven", min_value=0, value=30000, step=1000)
                s_location = st.text_input("Location (City)", placeholder="e.g. Ahmedabad")
            s_description = st.text_area("Description", placeholder="Tell buyers about the car's condition, service history, etc.")
            s_photo = st.file_uploader("Photo (optional)", type=["jpg", "jpeg", "png"])

            if st.form_submit_button("List My Car", type="primary"):
                if not s_brand.strip() or not s_model.strip():
                    st.error("Brand and Model are required.")
                else:
                    photo_b64 = db.encode_photo(s_photo.getvalue()) if s_photo is not None else None
                    listing_id = db.add_listing(
                        st.session_state.user['id'], s_brand, s_model, s_year, s_price,
                        s_fuel, s_transmission, s_km, s_location, s_description, photo=photo_b64
                    )
                    st.success("Your car is now listed on the Marketplace! Buyers can chat with you directly.")

        st.markdown("---")
        st.subheader("Your Current Listings")
        my_listings = db.get_listings_by_owner(st.session_state.user['id'])
        if not my_listings:
            st.caption("You haven't listed any cars yet.")
        for listing in my_listings:
            lcol1, lcol2 = st.columns([1, 5])
            with lcol1:
                if listing.get('photo'):
                    st.image(f"data:image/jpeg;base64,{listing['photo']}", width=80)
            with lcol2:
                st.write(f"**{listing['brand']} {listing['model']} ({listing['year']})** — ₹ {listing['price']:.2f} Lakhs")

elif page == "My Messages":
    st.header("💬 My Messages")
    if st.session_state.user is None:
        st.warning("Please log in from the sidebar to see your messages.")
    else:
        conversations = db.get_user_conversations(st.session_state.user['id'])
        if not conversations:
            st.info("No conversations yet. Visit the Marketplace and start chatting with a seller!")
        else:
            for convo in conversations:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"**{convo['listing_title']}** — conversation with **{convo['other_user_name']}**")
                with col2:
                    if st.button("Open", key=f"open_{convo['listing_id']}_{convo['other_user_id']}"):
                        st.session_state['open_chat'] = (convo['listing_id'], convo['other_user_id'])
                        st.session_state['nav_override'] = "Marketplace"
                        st.rerun()
                st.markdown("---")
        st.caption("Tip: opening a conversation takes you to the Marketplace page where the chat window appears.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Car4U Advanced Edition - Intelligent Car Recommendation & Price Prediction System</p>
    <p>Optimized for Performance | Data Science & Machine Learning</p>
</div>
""", unsafe_allow_html=True)
