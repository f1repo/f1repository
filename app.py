import os
import datetime
import requests
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import fastf1
from sklearn.linear_model import LinearRegression

# Optional AI import
try:
    from groq import Groq
except Exception:
    Groq = None

st.set_page_config(
    page_title="F1Repository",
    page_icon="https://i.postimg.cc/xTjBrHHZ/F1Repository-Gold.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

import tempfile
cache_dir = tempfile.mkdtemp()
fastf1.Cache.enable_cache(cache_dir)

GROQ_KEY = "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"

TEAM_COLORS = {
    "mercedes": "#00D7B6",
    "red bull": "#4781D7",
    "ferrari": "#ED1131",
    "mclaren": "#F47600",
    "aston martin": "#229971",
    "williams": "#1868DB",
    "alpine": "#00A1E8",
    "kick sauber": "#01C00E",
    "sauber": "#01C00E",
    "alfa romeo": "#01C00E",
    "alphatauri": "#6C98FF",
    "racing bulls": "#6C98FF",
    "rb": "#6C98FF",
    "haas": "#9C9FA2",
}

ALL_YEARS = list(range(1950, 2027))

RIP_DRIVERS = [
    "Ayrton Senna", "Gilles Villeneuve", "Jim Clark", "Jochen Rindt",
    "Bruce McLaren", "Francois Cevert", "Ronnie Peterson", "Wolfgang von Trips",
    "Luigi Musso", "Peter Collins", "Alberto Ascari", "John Surtees",
    "James Hunt", "Niki Lauda", "Michele Alboreto", "Elio de Angelis",
    "Roland Ratzenberger", "Jules Bianchi", "Stefan Bellof", "Tom Pryce",
    "Carlos Pace", "Roger Williamson"
]

HEAD_TO_HEAD_DRIVERS = [
    "Lewis Hamilton", "Michael Schumacher", "Max Verstappen", "Ayrton Senna",
    "Sebastian Vettel", "Fernando Alonso", "Charles Leclerc", "Lando Norris",
    "Carlos Sainz", "George Russell", "Oscar Piastri", "Alain Prost",
    "Niki Lauda", "Kimi Raikkonen", "Mika Hakkinen", "Juan Manuel Fangio",
    "Jim Clark", "Jackie Stewart", "Nigel Mansell", "Gilles Villeneuve",
    "Jenson Button", "Daniel Ricciardo", "Valtteri Bottas", "Jules Bianchi"
]


def get_team_color(team_name):
    if not team_name:
        return "#86868b"
    team_name = str(team_name).lower()
    for key, value in TEAM_COLORS.items():
        if key in team_name:
            return value
    return "#86868b"


@st.cache_data
def fetch_json(url):
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


@st.cache_data
def get_races_for_year(year):
    d = fetch_json(f"https://api.jolpi.ca/ergast/f1/{year}.json")
    if not d:
        return []
    try:
        return [r["raceName"] for r in d["MRData"]["RaceTable"]["Races"]]
    except Exception:
        return []


@st.cache_data
def get_results(year, rnd):
    d = fetch_json(f"https://api.jolpi.ca/ergast/f1/{year}/{rnd}/results.json")
    if not d:
        return None, None, None
    try:
        races = d["MRData"]["RaceTable"]["Races"]
        if not races:
            return None, None, None
        race = races[0]
        rows = []
        for r in race["Results"]:
            rows.append({
                "Position": int(r["position"]),
                "Driver": f"{r['Driver']['givenName']} {r['Driver']['familyName']}",
                "Abbreviation": r["Driver"].get("code", r["Driver"]["driverId"][:3].upper()),
                "Team": r["Constructor"]["name"],
                "Grid": int(r["grid"]),
                "Points": float(r["points"]),
                "Status": r["status"],
            })
        return pd.DataFrame(rows), race["raceName"], race["Circuit"]["circuitName"]
    except Exception:
        return None, None, None


@st.cache_data
def get_standings(year, rnd):
    d = fetch_json(f"https://api.jolpi.ca/ergast/f1/{year}/{rnd}/driverStandings.json")
    if not d:
        return None
    try:
        sl = d["MRData"]["StandingsTable"]["StandingsLists"]
        if not sl:
            return None
        rows = []
        for s in sl[0]["DriverStandings"]:
            rows.append({
                "Position": int(s["position"]),
                "Driver": f"{s['Driver']['givenName']} {s['Driver']['familyName']}",
                "Team": s["Constructors"][0]["name"],
                "Points": float(s["points"]),
                "Wins": int(s["wins"]),
            })
        return pd.DataFrame(rows)
    except Exception:
        return None


@st.cache_data
def get_all_drivers():
    all_drivers = []
    offset = 0
    while True:
        d = fetch_json(f"https://api.jolpi.ca/ergast/f1/drivers.json?limit=100&offset={offset}")
        if not d:
            break
        try:
            drivers = d["MRData"]["DriverTable"]["Drivers"]
            if not drivers:
                break
            for dr in drivers:
                all_drivers.append({
                    "Name": f"{dr['givenName']} {dr['familyName']}",
                    "Nationality": dr.get("nationality", "Unknown"),
                    "DOB": dr.get("dateOfBirth", "Unknown"),
                    "Driver ID": dr["driverId"],
                })
            total = int(d["MRData"]["total"])
            offset += 100
            if offset >= total:
                break
        except Exception:
            break
    return all_drivers


@st.cache_data
def get_driver_stats(driver_id):
    all_races = []
    offset = 0
    while True:
        d = fetch_json(f"https://api.jolpi.ca/ergast/f1/drivers/{driver_id}/results.json?limit=100&offset={offset}")
        if not d:
            break
        try:
            races = d["MRData"]["RaceTable"]["Races"]
            if not races:
                break
            all_races.extend(races)
            total = int(d["MRData"]["total"])
            offset += 100
            if offset >= total:
                break
        except Exception:
            break

    try:
        wins = podiums = poles = fastest_laps = 0
        total_points = 0.0

        for race in all_races:
            if "Results" in race and race["Results"]:
                res = race["Results"][0]
                pos = int(res.get("position", 99))
                pts = float(res.get("points", 0))
                grid = int(res.get("grid", 99))

                total_points += pts
                if pos == 1:
                    wins += 1
                if pos <= 3:
                    podiums += 1
                if grid == 1:
                    poles += 1
                if res.get("FastestLap", {}).get("rank") == "1":
                    fastest_laps += 1

        return {
            "Wins": wins,
            "Podiums": podiums,
            "Poles": poles,
            "Races": len(all_races),
            "Points": round(total_points, 1),
            "Fastest Laps": fastest_laps,
        }
    except Exception:
        return None


css = """
<style>
@media (max-width: 768px) {
    .hero-title {
        font-size: 80px !important;
        letter-spacing: -4px !important;
    }
    .product-title {
        font-size: 32px !important;
    }
    .product-description {
        font-size: 14px !important;
        padding: 0 20px !important;
    }
    .stats-bar {
        flex-direction: column !important;
        gap: 30px !important;
        padding: 40px 20px !important;
    }
    .stat-value {
        font-size: 40px !important;
    }
    .hero {
        padding: 100px 20px 80px !important;
    }
    .product-section {
        padding: 80px 20px !important;
    }
    .page-title {
        font-size: 32px !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 9px !important;
        padding: 12px 8px !important;
        letter-spacing: 1px !important;
    }
}
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@200;300;400;500;600;700&family=Cormorant+Garamond:wght@300;400;500;600;700&display=swap');

:root{
    --bg:#030303;
    --card:rgba(255,255,255,0.012);
    --card-hover:rgba(255,255,255,0.025);
    --border:rgba(255,255,255,0.04);
    --gold:#B8A04A;
    --platinum:#D4D4D4;
    --text:#E8E8E8;
    --muted:#4a4a4a;
    --dimmed:#707070;
}

.main{background-color:var(--bg);}
#MainMenu,footer,header{visibility:hidden;}
section[data-testid="stSidebar"]{display:none;}

html,body,[class*="css"]{
    font-family:'Inter',sans-serif;
    -webkit-font-smoothing:antialiased;
}

@keyframes shimmer{
    0%{background-position:-600% center;}
    100%{background-position:600% center;}
}

.stTabs [data-baseweb="tab-list"]{
    display:flex;
    justify-content:center;
    background:rgba(3,3,3,0.98);
    backdrop-filter:blur(30px);
    border-bottom:1px solid rgba(255,255,255,0.025);
    padding:0;
}
.stTabs [data-baseweb="tab"]{
    color:var(--muted);
    font-size:10px;
    font-weight:300;
    letter-spacing:2.5px;
    text-transform:uppercase;
    padding:22px 20px;
    background:transparent;
    border:none;
    border-bottom:1px solid transparent;
}
.stTabs [data-baseweb="tab"]:hover{
    color:var(--dimmed);
}
.stTabs [aria-selected="true"]{
    color:var(--gold)!important;
    border-bottom:1px solid rgba(184,160,74,0.4)!important;
    font-weight:400;
}

.hero{
    text-align:center;
    padding:220px 20px 180px;
    max-width:1000px;
    margin:0 auto;
    position:relative;
}
.hero::before{
    content:'';
    position:absolute;
    top:50%;left:50%;
    transform:translate(-50%,-50%);
    width:900px;height:900px;
    background:radial-gradient(circle,rgba(184,160,74,0.015) 0%,transparent 60%);
    border-radius:50%;
    pointer-events:none;
}
.hero-tag{
    font-size:9px;
    color:var(--gold);
    font-weight:300;
    letter-spacing:12px;
    text-transform:uppercase;
    margin-bottom:50px;
    opacity:0.6;
}
.hero-title{
    font-family:'Cormorant Garamond',serif;
    font-size:160px;
    font-weight:300;
    letter-spacing:-8px;
    line-height:0.8;
    margin-bottom:25px;
    background:linear-gradient(135deg,var(--platinum) 0%,var(--gold) 35%,var(--platinum) 65%,var(--gold) 100%);
    background-size:400% auto;
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    animation:shimmer 16s linear infinite;
}
.hero-underline{
    width:40px;
    height:1px;
    background:var(--gold);
    margin:45px auto;
    opacity:0.25;
}
.hero-subtitle{
    font-size:11px;
    font-weight:200;
    color:var(--muted);
    letter-spacing:8px;
    text-transform:uppercase;
}

.stats-bar{
    display:flex;
    justify-content:center;
    gap:100px;
    padding:100px 40px;
    max-width:1100px;
    margin:0 auto;
    border-top:1px solid var(--border);
    border-bottom:1px solid var(--border);
}
.stat-item{text-align:center;}
.stat-value{
    font-family:'Cormorant Garamond',serif;
    font-size:60px;
    font-weight:300;
    color:var(--platinum);
    letter-spacing:-3px;
}
.stat-label{
    font-size:8px;
    color:var(--muted);
    text-transform:uppercase;
    letter-spacing:5px;
    margin-top:16px;
    font-weight:300;
}

.product-section{
    text-align:center;
    padding:160px 20px;
    max-width:650px;
    margin:0 auto;
}
.product-label{
    font-size:9px;
    color:var(--gold);
    font-weight:300;
    letter-spacing:8px;
    text-transform:uppercase;
    margin-bottom:28px;
    opacity:0.6;
}
.product-title{
    font-family:'Cormorant Garamond',serif;
    font-size:52px;
    font-weight:400;
    color:var(--text);
    letter-spacing:-1px;
    line-height:1.12;
    margin-bottom:30px;
}
.product-description{
    font-size:14px;
    color:var(--muted);
    max-width:440px;
    margin:0 auto;
    line-height:2;
    font-weight:200;
}

.page-header{
    text-align:center;
    padding:120px 20px 60px;
    max-width:800px;
    margin:0 auto;
}
.page-title{
    font-family:'Cormorant Garamond',serif;
    font-size:48px;
    font-weight:400;
    color:var(--text);
    letter-spacing:-1px;
    margin-bottom:20px;
}
.page-subtitle{
    font-size:14px;
    color:var(--muted);
    font-weight:200;
    line-height:1.8;
}

.section-title-center{
    color:var(--gold);
    font-size:9px;
    font-weight:300;
    letter-spacing:6px;
    text-transform:uppercase;
    margin-bottom:35px;
    text-align:center;
    opacity:0.6;
}

.feature-card,.info-card,.record-card,.step-card{
    background:var(--card);
    border:1px solid var(--border);
    border-radius:12px;
}

.feature-card{
    padding:48px 30px;
    text-align:center;
    height:100%;
}
.feature-card:hover,.info-card:hover,.record-card:hover,.step-card:hover{
    background:var(--card-hover);
}
.feature-name{
    font-family:'Cormorant Garamond',serif;
    font-size:22px;
    font-weight:400;
    color:var(--text);
    margin-bottom:16px;
}
.feature-desc,.info-card-desc,.record-holder,.step-desc{
    font-size:12px;
    color:var(--muted);
    line-height:1.8;
    font-weight:200;
}

.info-card,.record-card{
    padding:28px;
    margin-bottom:8px;
}
.info-card-title,.record-title,.step-number{
    color:var(--gold);
    font-size:8px;
    font-weight:300;
    letter-spacing:5px;
    text-transform:uppercase;
    margin-bottom:12px;
    opacity:0.6;
}
.info-card-value{
    font-family:'Cormorant Garamond',serif;
    color:var(--text);
    font-size:36px;
    font-weight:400;
}
.record-value{
    font-family:'Cormorant Garamond',serif;
    color:var(--text);
    font-size:44px;
    font-weight:300;
}

.step-card{
    padding:48px 24px;
    text-align:center;
    height:100%;
}
.step-icon{font-size:36px;margin-bottom:20px;}
.step-title{
    font-family:'Cormorant Garamond',serif;
    font-size:20px;
    font-weight:400;
    color:var(--text);
    margin-bottom:12px;
}

div[data-testid="stButton"]{
    display:flex;
    justify-content:center;
}
.stButton>button{
    background:transparent;
    color:var(--gold);
    border:1px solid rgba(184,160,74,0.25);
    border-radius:12px;
    padding:16px 56px;
    font-size:9px;
    font-weight:300;
    letter-spacing:6px;
    text-transform:uppercase;
}
.stButton>button:hover{
    background:rgba(184,160,74,0.08);
    border-color:rgba(184,160,74,0.4);
}

.stMetric{
    background:var(--card);
    padding:20px;
    border-radius:12px;
    border:1px solid var(--border);
}

.apple-divider,.apple-divider-sm{
    border:none;
    height:1px;
    background:var(--border);
}
.apple-divider{margin:0;}
.apple-divider-sm{margin:60px 0;}

.clean-footer{
    text-align:center;
    padding:120px 20px;
    border-top:1px solid var(--border);
    margin-top:180px;
}

.stSelectbox label,.stSlider label{
    color:var(--muted)!important;
    font-size:9px!important;
    font-weight:200!important;
    letter-spacing:4px!important;
    text-transform:uppercase!important;
}
.stDataFrame{
    border-radius:12px;
    overflow:hidden;
    border:1px solid var(--border);
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

tabs = st.tabs(["Home","◈ Races","◉ AI Lab","∅ Drivers","◆ Teams","△ Standings","✦ Records","⚔ Head to Head","◎ Roman","⏱ Race Weekend","◇ Circuits","▲ Season","◆ Constructors","◷ Lap Times","◎ Points Calc","◈ Race Rewind","◉ Form Guide","◈ About"])
with tabs[0]:
    st.markdown("""
    <div class='hero'>
        <div class='hero-tag'>F1Repository</div>
        <div class='hero-title'>F1Repository</div>
        <div class='hero-underline'></div>
        <div class='hero-subtitle'>Formula 1 intelligence. Refined.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='stats-bar'>
        <div class='stat-item'><div class='stat-value'>75+</div><div class='stat-label'>Years</div></div>
        <div class='stat-item'><div class='stat-value'>1100+</div><div class='stat-label'>Races</div></div>
        <div class='stat-item'><div class='stat-value'>860+</div><div class='stat-label'>Drivers</div></div>
        <div class='stat-item'><div class='stat-value'>24</div><div class='stat-label'>Circuits</div></div>
    </div>
    """, unsafe_allow_html=True)

    sections = [
        ("Race Analysis", "Every lap. Every point.<br>Every detail.", "Explore complete race results, positions, points and visual breakdowns from across Formula 1 history."),
        ("AI Predictor", "Predict the<br>unpredictable.", "Train a machine learning model on real race data and forecast finishing positions from any grid slot."),
        ("Driver Intelligence", "Know your driver.<br>Inside out.", "Understand performance, position changes, points scored and race outcomes in a clean premium interface."),
        ("Constructor Analytics", "Team intelligence.<br>Redefined.", "Compare teams, scorelines, and constructor performance through clean visual charts and standings.")
    ]

    for label, title, desc in sections:
        st.markdown("<hr class='apple-divider'>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='product-section'>
            <div class='product-label'>{label}</div>
            <div class='product-title'>{title}</div>
            <div class='product-description'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='apple-divider'>", unsafe_allow_html=True)

    features = [
        ("Historical Results", "Browse race results from 1950 to 2026."),
        ("Modern Telemetry", "Use FastF1-powered race analysis for modern sessions."),
        ("Championship Tables", "Check standings at any point in a season."),
        ("Machine Learning", "Predict finishing positions using real race data."),
        ("Team Colors", "Visuals styled with constructor-inspired colors."),
        ("Premium UI", "A luxury dark interface built for F1 fans."),
    ]

    st.markdown("""
    <div class='page-header'>
        <div class='product-label'>Platform</div>
        <div class='page-title'>Everything essential.</div>
    </div>
    """, unsafe_allow_html=True)

    for i in range(0, len(features), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            if i + j < len(features):
                name, desc = features[i + j]
                col.markdown(f"""
                <div class='feature-card'>
                    <div class='feature-name'>{name}</div>
                    <div class='feature-desc'>{desc}</div>
                </div>
                """, unsafe_allow_html=True)
        st.write("")


with tabs[1]:
    st.markdown("""
    <div class='page-header'>
        <div class='product-label'>Race Analysis</div>
        <div class='page-title'>Explore any Grand Prix.</div>
        <div class='page-subtitle'>Historical results and modern race analysis in one place.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='apple-divider-sm'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        race_year = st.selectbox("Year", ALL_YEARS, index=len(ALL_YEARS)-2, key="race_year")
    race_list = get_races_for_year(race_year)
    with col2:
        race_name = st.selectbox("Grand Prix", race_list, key="race_name") if race_list else None

    if st.button("Analyze Race", key="analyze_race") and race_name:
        rnd = race_list.index(race_name) + 1

        if race_year >= 2018:
            try:
                session = fastf1.get_session(race_year, rnd, 'R')
                session.load()

                st.markdown(f"""
                <div class='page-header'>
                    <div class='product-label'>Race Result</div>
                    <div class='page-title'>{race_year} {race_name}</div>
                </div>
                """, unsafe_allow_html=True)

                winner = session.results.iloc[0]
                second = session.results.iloc[1]
                third = session.results.iloc[2]

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Winner", winner["Abbreviation"])
                c2.metric("Second", second["Abbreviation"])
                c3.metric("Third", third["Abbreviation"])
                c4.metric("Laps", session.total_laps)

                left, right = st.columns(2)

                with left:
                    st.dataframe(
                        session.results[["Abbreviation", "TeamName", "Position", "Points"]],
                        use_container_width=True,
                        height=450
                    )

                with right:
                    top10 = session.results.head(10)
                    colors = [get_team_color(row["TeamName"]) for _, row in top10.iterrows()]
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.bar(top10["Abbreviation"], top10["Points"], color=colors, width=0.6)
                    ax.set_facecolor("#030303")
                    fig.patch.set_facecolor("#030303")
                    ax.tick_params(colors="#707070")
                    ax.set_ylabel("Points", color="#707070")
                    for s in ax.spines.values():
                        s.set_color("#1d1d1f")
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.grid(axis="y", color="#1d1d1f", linewidth=0.5)
                    plt.tight_layout()
                    st.pyplot(fig)

            except Exception:
                st.error("Could not load this race session.")
        else:
            df, full_name, circuit = get_results(race_year, rnd)
            if df is not None:
                st.markdown(f"""
                <div class='page-header'>
                    <div class='product-label'>Historical Result</div>
                    <div class='page-title'>{race_year} {full_name}</div>
                    <div class='page-subtitle'>{circuit}</div>
                </div>
                """, unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)
                c1.metric("Winner", df.iloc[0]["Driver"])
                c2.metric("Second", df.iloc[1]["Driver"])
                c3.metric("Third", df.iloc[2]["Driver"])

                st.dataframe(df, use_container_width=True, height=500)
            else:
                st.error("No historical data available for this race.")


with tabs[2]:
    st.markdown("""
    <div class='page-header'>
        <div class='product-label'>AI Predictor</div>
        <div class='page-title'>Predict the unpredictable.</div>
        <div class='page-subtitle'>Train a simple machine learning model using real Formula 1 race data.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='apple-divider-sm'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        ai_year = st.selectbox("Year", list(range(2018, 2027)), index=6, key="ai_year")
    ai_races = get_races_for_year(ai_year)
    with col2:
        ai_race = st.selectbox("Race", ai_races, key="ai_race") if ai_races else None

    if st.button("Train AI", key="train_ai") and ai_race:
        rnd = ai_races.index(ai_race) + 1
        try:
            session = fastf1.get_session(ai_year, rnd, 'R')
            session.load()
            data = session.results[["GridPosition", "Position"]].dropna()
            model = LinearRegression()
            model.fit(data[["GridPosition"]], data["Position"])
            st.session_state["model"] = model
            st.session_state["ai_data"] = data
            st.session_state["ai_label"] = f"{ai_year} {ai_race}"
            st.success(f"AI trained on {ai_year} {ai_race}")
        except Exception:
            st.error("Could not train the model for this race.")

    if "model" in st.session_state:
        st.markdown("<hr class='apple-divider-sm'>", unsafe_allow_html=True)
        st.markdown(f"<div class='section-title-center'>{st.session_state['ai_label']}</div>", unsafe_allow_html=True)

        grid = st.slider("Grid Position", 1, 20, 1, key="grid_predict")
        pred = round(st.session_state["model"].predict([[grid]])[0])
        pred = max(1, min(20, pred))

        c1, c2 = st.columns(2)
        c1.metric("Grid", f"P{grid}")
        c2.metric("Predicted Finish", f"P{pred}")


with tabs[3]:
    st.markdown("""
    <div class='page-header'>
        <div class='product-label'>Drivers</div>
        <div class='page-title'>Know your driver.</div>
        <div class='page-subtitle'>Load a race and inspect an individual driver's result.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='apple-divider-sm'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        d_year = st.selectbox("Year", ALL_YEARS, index=len(ALL_YEARS)-2, key="d_year")
    d_races = get_races_for_year(d_year)
    with col2:
        d_race = st.selectbox("Race", d_races, key="d_race") if d_races else None

    if st.button("Load Drivers", key="load_drivers") and d_race:
        rnd = d_races.index(d_race) + 1
        if d_year >= 2018:
            try:
                session = fastf1.get_session(d_year, rnd, 'R')
                session.load()
                st.session_state["driver_session"] = session
                st.session_state["driver_list"] = session.results["Abbreviation"].tolist()
                st.success("Select a driver below.")
            except Exception:
                st.error("Could not load driver list.")
        else:
            df, _, _ = get_results(d_year, rnd)
            if df is not None:
                st.session_state["driver_df"] = df
                st.session_state["driver_list_hist"] = df["Driver"].tolist()
                st.success("Select a driver below.")

    if "driver_session" in st.session_state:
        driver = st.selectbox("Driver", st.session_state["driver_list"], key="driver_select_live")
        if st.button("Show Driver Stats", key="show_driver_live"):
            result = st.session_state["driver_session"].results[st.session_state["driver_session"].results["Abbreviation"] == driver]
            if not result.empty:
                row = result.iloc[0]
                tc = get_team_color(row["TeamName"])
                st.markdown(f"<div class='page-header'><div class='page-title' style='color:{tc};'>{driver}</div><div class='page-subtitle'>{row['TeamName']}</div></div>", unsafe_allow_html=True)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Grid", f"P{int(row['GridPosition'])}")
                c2.metric("Finish", f"P{int(row['Position'])}")
                c3.metric("Points", int(row["Points"]))
                c4.metric("Change", int(row["GridPosition"]) - int(row["Position"]))

    if "driver_df" in st.session_state:
        driver = st.selectbox("Driver", st.session_state["driver_list_hist"], key="driver_select_hist")
        if st.button("Show Driver Stats", key="show_driver_hist"):
            result = st.session_state["driver_df"][st.session_state["driver_df"]["Driver"] == driver]
            if not result.empty:
                row = result.iloc[0]
                st.markdown(f"<div class='page-header'><div class='page-title'>{row['Driver']}</div><div class='page-subtitle'>{row['Team']}</div></div>", unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                c1.metric("Grid", f"P{int(row['Grid'])}")
                c2.metric("Finish", f"P{int(row['Position'])}")
                c3.metric("Points", int(row["Points"]))


with tabs[4]:
    st.markdown("""
    <div class='page-header'>
        <div class='product-label'>Teams</div>
        <div class='page-title'>Team intelligence.</div>
        <div class='page-subtitle'>Load a race and inspect constructor performance.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='apple-divider-sm'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        t_year = st.selectbox("Year", ALL_YEARS, index=len(ALL_YEARS)-2, key="t_year")
    t_races = get_races_for_year(t_year)
    with col2:
        t_race = st.selectbox("Race", t_races, key="t_race") if t_races else None

    if st.button("Load Teams", key="load_teams") and t_race:
        rnd = t_races.index(t_race) + 1
        if t_year >= 2018:
            try:
                session = fastf1.get_session(t_year, rnd, 'R')
                session.load()
                st.session_state["team_session"] = session
                st.session_state["team_list"] = session.results["TeamName"].unique().tolist()
                st.success("Select a team below.")
            except Exception:
                st.error("Could not load teams.")

    if "team_session" in st.session_state:
        team = st.selectbox("Team", st.session_state["team_list"], key="team_select_live")
        if st.button("Show Team Stats", key="show_team_live"):
            tr = st.session_state["team_session"].results[st.session_state["team_session"].results["TeamName"] == team]
            if not tr.empty:
                tc = get_team_color(team)
                st.markdown(f"<div class='page-header'><div class='page-title' style='color:{tc};'>{team}</div></div>", unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                c1.metric("Drivers", len(tr))
                c2.metric("Points", int(tr["Points"].sum()))
                c3.metric("Best Finish", f"P{int(tr['Position'].min())}")
                st.dataframe(tr[["Abbreviation", "GridPosition", "Position", "Points"]], use_container_width=True)


with tabs[5]:
    st.markdown("""
    <div class='page-header'>
        <div class='product-label'>Championships</div>
        <div class='page-title'>Season standings.</div>
        <div class='page-subtitle'>Driver standings from any season and race checkpoint.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='apple-divider-sm'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        s_year = st.selectbox("Year", ALL_YEARS, index=len(ALL_YEARS)-2, key="s_year")
    s_races = get_races_for_year(s_year)
    with col2:
        s_race = st.selectbox("After Race", s_races, key="s_race") if s_races else None

    if st.button("Load Standings", key="load_standings") and s_race:
        rnd = s_races.index(s_race) + 1
        standings = get_standings(s_year, rnd)
        if standings is not None and not standings.empty:
            c1, c2, c3 = st.columns(3)
            c1.metric("Leader", standings.iloc[0]["Driver"], f"{int(standings.iloc[0]['Points'])} pts")
            c2.metric("Second", standings.iloc[1]["Driver"], f"{int(standings.iloc[1]['Points'])} pts")
            c3.metric("Third", standings.iloc[2]["Driver"], f"{int(standings.iloc[2]['Points'])} pts")

            left, right = st.columns(2)
            with left:
                st.dataframe(standings, use_container_width=True, height=450)
            with right:
                top20 = standings.head(20)
                colors = [get_team_color(row["Team"]) for _, row in top20.iterrows()]
                fig, ax = plt.subplots(figsize=(8, 7))
                ax.barh(top20["Driver"][::-1], top20["Points"][::-1], color=colors[::-1], height=0.6)
                ax.set_facecolor("#030303")
                fig.patch.set_facecolor("#030303")
                ax.tick_params(colors="#707070")
                for s in ax.spines.values():
                    s.set_color("#1d1d1f")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)


with tabs[6]:
    st.markdown("""
    <div class='page-header'>
        <div class='product-label'>Hall of Fame</div>
        <div class='page-title'>Records that define greatness.</div>
    </div>
    """, unsafe_allow_html=True)

    records = {
        "Driver Records": [
            ("Most Championships", "7", "Hamilton & Schumacher"),
            ("Most Race Wins", "103", "Lewis Hamilton"),
            ("Most Podiums", "197", "Lewis Hamilton"),
            ("Most Consecutive Wins", "10", "Verstappen (2023)"),
            ("Most Pole Positions", "104", "Lewis Hamilton"),
            ("Most Entries", "400+", "Fernando Alonso"),
        ],
        "Constructor Records": [
            ("Most Titles", "16", "Ferrari"),
            ("Most Wins", "243", "Ferrari"),
            ("Most Dominant", "21/22", "Red Bull (2023)"),
        ],
        "Season Records": [
            ("Most Wins In Season", "19", "Verstappen (2023)"),
            ("Most Points In Season", "575", "Verstappen (2023)"),
            ("Longest Race", "3h 49m", "2011 Canadian GP"),
        ]
    }

    for section, items in records.items():
        st.markdown("<hr class='apple-divider-sm'>", unsafe_allow_html=True)
        st.markdown(f"<div class='section-title-center'>{section}</div>", unsafe_allow_html=True)
        cols = st.columns(3)
        for i, (title, value, holder) in enumerate(items):
            cols[i % 3].markdown(
                f"""
                <div class='record-card'>
                    <div class='record-title'>{title}</div>
                    <div class='record-value'>{value}</div>
                    <div class='record-holder'>{holder}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
with tabs[7]:
    st.markdown("<div class='page-header'><div class='product-label'>Head to Head</div><div class='page-title'>The ultimate battle.</div></div>", unsafe_allow_html=True)
    st.markdown("<hr class='apple-divider-sm'>", unsafe_allow_html=True)

    compare_mode = st.radio("Mode", ["Career Battle", "Race Battle", "Season Battle"], horizontal=True, key="h2h_mode")
    st.markdown("<hr class='apple-divider-sm'>", unsafe_allow_html=True)

    h2h_c1, h2h_c2 = st.columns(2)
    with h2h_c1:
        d1_name = st.selectbox("Driver 1", HEAD_TO_HEAD_DRIVERS, index=0, key="h2h_d1")
    with h2h_c2:
        d2_name = st.selectbox("Driver 2", HEAD_TO_HEAD_DRIVERS, index=1, key="h2h_d2")

    if compare_mode == "Career Battle":
        if st.button("Compare Careers", key="h2h_career"):
            if d1_name == d2_name:
                st.error("Pick two different drivers.")
            else:
                ad = get_all_drivers()
                d1_id = next((d["Driver ID"] for d in ad if d["Name"] == d1_name), None)
                d2_id = next((d["Driver ID"] for d in ad if d["Name"] == d2_name), None)
                if not d1_id or not d2_id:
                    st.error("Could not find one or both drivers.")
                else:
                    with st.spinner("Loading career data..."):
                        d1_stats = get_driver_stats(d1_id)
                        d2_stats = get_driver_stats(d2_id)
                    if not d1_stats or not d2_stats:
                        st.error("Could not load career stats.")
                    else:
                        st.markdown(f"<div class='page-header'><div class='product-label'>Career Battle</div><div class='page-title'>{d1_name} vs {d2_name}</div></div>", unsafe_allow_html=True)
                        metrics = ["Races", "Wins", "Podiums", "Poles", "Points", "Fastest Laps"]
                        for m in metrics:
                            v1 = d1_stats[m]
                            v2 = d2_stats[m]
                            w1 = "#B8A04A" if v1 >= v2 else "#4a4a4a"
                            w2 = "#B8A04A" if v2 >= v1 else "#4a4a4a"
                            c1, mid, c2 = st.columns([2, 1, 2])
                            with c1:
                                st.markdown(f"<div class='info-card' style='text-align:right;'><div class='info-card-value' style='color:{w1};'>{v1}</div></div>", unsafe_allow_html=True)
                            with mid:
                                st.markdown(f"<div class='info-card' style='text-align:center;'><div class='info-card-title'>{m}</div></div>", unsafe_allow_html=True)
                            with c2:
                                st.markdown(f"<div class='info-card' style='text-align:left;'><div class='info-card-value' style='color:{w2};'>{v2}</div></div>", unsafe_allow_html=True)

                        d1s = sum(1 for m in metrics if d1_stats[m] > d2_stats[m])
                        d2s = sum(1 for m in metrics if d2_stats[m] > d1_stats[m])
                        verdict = d1_name if d1s > d2s else (d2_name if d2s > d1s else "TIE")
                        vs = f"{max(d1s, d2s)}-{min(d1s, d2s)}"
                        st.markdown(f"<div style='text-align:center;padding:40px;'><div class='product-label'>The Verdict</div><div class='page-title' style='color:#B8A04A;'>{verdict}</div><div class='hero-subtitle'>Wins {vs}</div></div>", unsafe_allow_html=True)

    if compare_mode == "Race Battle":
        rb_c1, rb_c2 = st.columns(2)
        with rb_c1:
            rb_year = st.selectbox("Year", ALL_YEARS, index=len(ALL_YEARS)-2, key="rb_year")
        with rb_c2:
            rb_races = get_races_for_year(rb_year)
            rb_race = st.selectbox("Race", rb_races, key="rb_race") if rb_races else None
        if st.button("Compare Race", key="h2h_race") and rb_race:
            rnd = rb_races.index(rb_race) + 1
            if rb_year >= 2018:
                try:
                    session = fastf1.get_session(rb_year, rnd, 'R')
                    session.load()
                    a1 = d1_name.split()[-1][:3].upper()
                    a2 = d2_name.split()[-1][:3].upper()
                    aa = session.results["Abbreviation"].tolist()
                    f1 = next((a for a in aa if a == a1), None)
                    f2 = next((a for a in aa if a == a2), None)
                    if not f1 or not f2:
                        st.error(f"Not found. Available: {', '.join(aa)}")
                    else:
                        r1 = session.results[session.results["Abbreviation"] == f1].iloc[0]
                        r2 = session.results[session.results["Abbreviation"] == f2].iloc[0]
                        st.markdown(f"<div class='page-header'><div class='page-title'>{f1} vs {f2}</div></div>", unsafe_allow_html=True)
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown(f"<div class='record-card' style='border-left:4px solid {get_team_color(r1['TeamName'])};'><div class='record-value'>{f1}</div><div class='record-holder'>Grid: P{int(r1['GridPosition'])} | Finish: P{int(r1['Position'])}</div></div>", unsafe_allow_html=True)
                        with c2:
                            st.markdown(f"<div class='record-card' style='border-left:4px solid {get_team_color(r2['TeamName'])};'><div class='record-value'>{f2}</div><div class='record-holder'>Grid: P{int(r2['GridPosition'])} | Finish: P{int(r2['Position'])}</div></div>", unsafe_allow_html=True)
                except Exception:
                    st.error("Error loading race data.")
            else:
                df, fn, ci = get_results(rb_year, rnd)
                if df is not None:
                    l1 = d1_name.split()[-1].lower()
                    l2 = d2_name.split()[-1].lower()
                    r1 = df[df["Driver"].str.lower().str.contains(l1)]
                    r2 = df[df["Driver"].str.lower().str.contains(l2)]
                    if not r1.empty and not r2.empty:
                        r1 = r1.iloc[0]
                        r2 = r2.iloc[0]
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown(f"<div class='record-card'><div class='record-value'>{r1['Driver'].split()[-1]}</div><div class='record-holder'>P{r1['Position']}</div></div>", unsafe_allow_html=True)
                        with c2:
                            st.markdown(f"<div class='record-card'><div class='record-value'>{r2['Driver'].split()[-1]}</div><div class='record-holder'>P{r2['Position']}</div></div>", unsafe_allow_html=True)

    if compare_mode == "Season Battle":
        sb_year = st.selectbox("Season", ALL_YEARS, index=len(ALL_YEARS)-2, key="sb_year")
        if st.button("Compare Season", key="h2h_season"):
            races = get_races_for_year(sb_year)
            if races:
                standings = get_standings(sb_year, len(races))
                if standings is not None:
                    r1 = standings[standings["Driver"].str.contains(d1_name.split()[-1], case=False)]
                    r2 = standings[standings["Driver"].str.contains(d2_name.split()[-1], case=False)]
                    if not r1.empty and not r2.empty:
                        r1 = r1.iloc[0]
                        r2 = r2.iloc[0]
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown(f"<div class='record-card'><div class='record-value'>{r1['Driver'].split()[-1]}</div><div class='record-holder'>P{r1['Position']} | {int(r1['Points'])} pts</div></div>", unsafe_allow_html=True)
                        with c2:
                            st.markdown(f"<div class='record-card'><div class='record-value'>{r2['Driver'].split()[-1]}</div><div class='record-holder'>P{r2['Position']} | {int(r2['Points'])} pts</div></div>", unsafe_allow_html=True)


with tabs[8]:
    st.markdown("<div class='page-header'><div class='product-label'>Meet Roman</div><div class='page-title'>Your F1 companion.</div></div>", unsafe_allow_html=True)
    st.markdown("<hr class='apple-divider-sm'>", unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for msg in st.session_state["chat_history"]:
        border = "#B8A04A" if msg["role"] == "user" else "#00D7B6"
        label = "YOU" if msg["role"] == "user" else "ROMAN"
        st.markdown(f"<div class='info-card' style='border-left:3px solid {border};'><div class='info-card-title'>{label}</div><div class='info-card-desc' style='color:#E8E8E8;font-size:14px;'>{msg['content']}</div></div>", unsafe_allow_html=True)

    user_input = st.text_input("Ask anything about F1...", key="roman_input", placeholder="Who won the 2021 championship?")

    if st.button("Ask Roman", key="roman_btn") and user_input:
        if not GROQ_KEY or GROQ_KEY == "paste_your_groq_key_here":
            st.error("Add your Groq API key to the code.")
        elif Groq is None:
            st.error("Groq library not installed. Run: pip install groq")
        else:
            with st.spinner("Roman is thinking..."):
                try:
                    client = Groq(api_key=GROQ_KEY)
                    result = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "You are Roman, an expert F1 assistant. You know everything about F1 from 1950 to 2025. Be concise and factual."},
                            {"role": "user", "content": user_input}
                        ],
                        model="llama-3.3-70b-versatile",
                        temperature=0.7,
                        max_tokens=1024,
                    )
                    answer = result.choices[0].message.content
                    st.session_state["chat_history"].append({"role": "user", "content": user_input})
                    st.session_state["chat_history"].append({"role": "assistant", "content": answer})
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    if st.button("Clear Chat", key="roman_clear"):
        st.session_state["chat_history"] = []
        st.rerun()


with tabs[9]:
    st.markdown("<div class='page-header'><div class='product-label'>Race Weekend</div><div class='page-title'>Race Countdown</div></div>", unsafe_allow_html=True)
    st.markdown("<hr class='apple-divider-sm'>", unsafe_allow_html=True)

    CAL = [
        {"name":"Australian GP","loc":"Melbourne","circuit":"Albert Park","round":1,"date":"2026-03-15 05:00","cancelled":False,"reason":""},
        {"name":"Chinese GP","loc":"Shanghai","circuit":"Shanghai Circuit","round":2,"date":"2026-03-22 07:00","cancelled":False,"reason":""},
        {"name":"Japanese GP","loc":"Suzuka","circuit":"Suzuka Circuit","round":3,"date":"2026-04-05 06:00","cancelled":False,"reason":""},
        {"name":"Bahrain GP","loc":"Sakhir","circuit":"Bahrain Circuit","round":4,"date":"2026-04-12 15:00","cancelled":True,"reason":"Cancelled - West Asia conflicts"},
        {"name":"Saudi Arabian GP","loc":"Jeddah","circuit":"Jeddah Circuit","round":5,"date":"2026-04-19 17:00","cancelled":True,"reason":"Cancelled - West Asia conflicts"},
        {"name":"Miami GP","loc":"Miami","circuit":"Miami Autodrome","round":6,"date":"2026-05-03 19:00","cancelled":False,"reason":""},
        {"name":"Emilia Romagna GP","loc":"Imola","circuit":"Imola Circuit","round":7,"date":"2026-05-17 13:00","cancelled":False,"reason":""},
        {"name":"Monaco GP","loc":"Monte Carlo","circuit":"Circuit de Monaco","round":8,"date":"2026-05-24 13:00","cancelled":False,"reason":""},
        {"name":"Spanish GP","loc":"Barcelona","circuit":"Catalunya Circuit","round":9,"date":"2026-05-31 13:00","cancelled":False,"reason":""},
        {"name":"Canadian GP","loc":"Montreal","circuit":"Gilles Villeneuve","round":10,"date":"2026-06-14 18:00","cancelled":False,"reason":""},
        {"name":"Austrian GP","loc":"Spielberg","circuit":"Red Bull Ring","round":11,"date":"2026-06-28 13:00","cancelled":False,"reason":""},
        {"name":"British GP","loc":"Silverstone","circuit":"Silverstone Circuit","round":12,"date":"2026-07-05 14:00","cancelled":False,"reason":""},
        {"name":"Belgian GP","loc":"Spa","circuit":"Spa-Francorchamps","round":13,"date":"2026-07-26 13:00","cancelled":False,"reason":""},
        {"name":"Hungarian GP","loc":"Budapest","circuit":"Hungaroring","round":14,"date":"2026-08-02 13:00","cancelled":False,"reason":""},
        {"name":"Dutch GP","loc":"Zandvoort","circuit":"Zandvoort Circuit","round":15,"date":"2026-08-30 13:00","cancelled":False,"reason":""},
        {"name":"Italian GP","loc":"Monza","circuit":"Monza Circuit","round":16,"date":"2026-09-06 13:00","cancelled":False,"reason":""},
        {"name":"Azerbaijan GP","loc":"Baku","circuit":"Baku City Circuit","round":17,"date":"2026-09-20 11:00","cancelled":False,"reason":""},
        {"name":"Singapore GP","loc":"Singapore","circuit":"Marina Bay","round":18,"date":"2026-10-04 12:00","cancelled":False,"reason":""},
        {"name":"US GP","loc":"Austin","circuit":"COTA","round":19,"date":"2026-10-18 19:00","cancelled":False,"reason":""},
        {"name":"Mexico GP","loc":"Mexico City","circuit":"Hermanos Rodriguez","round":20,"date":"2026-10-25 20:00","cancelled":False,"reason":""},
        {"name":"Sao Paulo GP","loc":"Sao Paulo","circuit":"Interlagos","round":21,"date":"2026-11-08 17:00","cancelled":False,"reason":""},
        {"name":"Las Vegas GP","loc":"Las Vegas","circuit":"Strip Circuit","round":22,"date":"2026-11-21 06:00","cancelled":False,"reason":""},
        {"name":"Qatar GP","loc":"Lusail","circuit":"Lusail Circuit","round":23,"date":"2026-11-29 17:00","cancelled":False,"reason":""},
        {"name":"Abu Dhabi GP","loc":"Abu Dhabi","circuit":"Yas Marina","round":24,"date":"2026-12-06 13:00","cancelled":False,"reason":""},
    ]

    now = datetime.datetime.utcnow()
    next_race = None
    for race in CAL:
        rt = datetime.datetime.strptime(race["date"], "%Y-%m-%d %H:%M")
        if rt > now and not race.get("cancelled", False):
            next_race = race
            break

    if next_race:
        rt = datetime.datetime.strptime(next_race["date"], "%Y-%m-%d %H:%M")
        diff = rt - now
        st.markdown(f"<div style='text-align:center;padding:40px;'><div class='product-label'>Next Race</div><div class='page-title'>{next_race['name']}</div><div class='page-subtitle'>{next_race['circuit']} | {next_race['loc']}</div></div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Round", f"R{next_race['round']}")
        c2.metric("Days", diff.days)
        c3.metric("Hours", diff.seconds // 3600)
        c4.metric("Minutes", (diff.seconds % 3600) // 60)

        st.markdown("<hr class='apple-divider-sm'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title-center'>Full 2026 Calendar</div>", unsafe_allow_html=True)

        for race in CAL:
            rdt = datetime.datetime.strptime(race["date"], "%Y-%m-%d %H:%M")
            ic = race.get("cancelled", False)
            inx = next_race and race["name"] == next_race["name"] and not ic
            ip = rdt < now

            if ic:
                color = "#ED1131"
                status = "CANCELLED"
            elif inx:
                color = "#B8A04A"
                status = "NEXT RACE"
            elif ip:
                color = "#4a4a4a"
                status = "COMPLETED"
            else:
                color = "#E8E8E8"
                status = "UPCOMING"

            ct = ""
            if ic:
                ct = "<div style='color:#ED1131;font-size:10px;margin-top:6px;'>" + race["reason"] + "</div>"

            st.markdown(
                "<div class='info-card' style='border-left:3px solid " + color + ";'>"
                "<div style='display:flex;justify-content:space-between;align-items:center;'>"
                "<div>"
                "<div class='info-card-title'>Round " + str(race["round"]) + " | " + status + "</div>"
                "<div class='info-card-value' style='font-size:22px;color:" + color + ";'>" + race["name"] + "</div>"
                "<div class='info-card-desc'>" + race["circuit"] + " | " + race["loc"] + "</div>"
                + ct +
                "</div>"
                "<div style='text-align:right;color:" + color + ";font-weight:600;'>" + rdt.strftime("%b %d") + "</div>"
                "</div>"
                "</div>",
                unsafe_allow_html=True
            )
    else:
        st.info("2026 Season Complete!")


with tabs[10]:
    st.markdown("<div class='page-header'><div class='product-label'>Circuit Guide</div><div class='page-title'>Every track. Every detail.</div></div>", unsafe_allow_html=True)
    st.markdown("<hr class='apple-divider-sm'>", unsafe_allow_html=True)

    CIRCUITS = [
        {"name":"Albert Park","loc":"Melbourne","length":"5.278 km","laps":58,"turns":14,"record":"1:20.235 - Leclerc (2024)","desc":"Fast circuit around a lake."},
        {"name":"Shanghai Circuit","loc":"Shanghai","length":"5.451 km","laps":56,"turns":16,"record":"1:32.238 - Schumacher (2004)","desc":"Unique snail-shaped turns."},
        {"name":"Suzuka","loc":"Japan","length":"5.807 km","laps":53,"turns":18,"record":"1:30.983 - Hamilton (2019)","desc":"Figure-eight with legendary 130R."},
        {"name":"Circuit de Monaco","loc":"Monte Carlo","length":"3.337 km","laps":78,"turns":19,"record":"1:12.909 - Hamilton (2021)","desc":"The crown jewel of F1."},
        {"name":"Silverstone","loc":"UK","length":"5.891 km","laps":52,"turns":18,"record":"1:27.097 - Verstappen (2020)","desc":"Home of F1 since 1950."},
        {"name":"Spa-Francorchamps","loc":"Belgium","length":"7.004 km","laps":44,"turns":19,"record":"1:46.286 - Bottas (2018)","desc":"The greatest circuit. Eau Rouge."},
        {"name":"Monza","loc":"Italy","length":"5.793 km","laps":53,"turns":11,"record":"1:21.046 - Barrichello (2004)","desc":"Temple of Speed."},
        {"name":"Yas Marina","loc":"Abu Dhabi","length":"5.281 km","laps":58,"turns":16,"record":"1:26.103 - Verstappen (2021)","desc":"Season finale venue."},
        {"name":"COTA","loc":"Austin","length":"5.513 km","laps":56,"turns":20,"record":"1:36.169 - Leclerc (2019)","desc":"Iconic uphill Turn 1."},
        {"name":"Jeddah","loc":"Saudi Arabia","length":"6.174 km","laps":50,"turns":27,"record":"1:30.734 - Hamilton (2021)","desc":"Fastest street circuit."},
        {"name":"Miami","loc":"USA","length":"5.412 km","laps":57,"turns":19,"record":"1:29.708 - Verstappen (2023)","desc":"Hard Rock Stadium circuit."},
        {"name":"Red Bull Ring","loc":"Austria","length":"4.318 km","laps":71,"turns":10,"record":"1:05.619 - Sainz (2020)","desc":"Short and fast in mountains."},
    ]

    for c in CIRCUITS:
        st.markdown(
            "<div class='info-card'>"
            "<div class='info-card-value' style='font-size:24px;'>" + c["name"] + "</div>"
            "<div class='info-card-desc'>" + c["loc"] + "</div>"
            "<div class='info-card-desc'>Length: " + c["length"] + " | Laps: " + str(c["laps"]) + " | Turns: " + str(c["turns"]) + "</div>"
            "<div class='info-card-desc'>Record: " + c["record"] + "</div>"
            "<div class='info-card-desc' style='margin-top:8px;'>" + c["desc"] + "</div>"
            "</div>",
            unsafe_allow_html=True
        )


with tabs[11]:
    st.markdown("<div class='page-header'><div class='product-label'>Season Progression</div><div class='page-title'>Championship battle.</div></div>", unsafe_allow_html=True)
    st.markdown("<hr class='apple-divider-sm'>", unsafe_allow_html=True)

    sp_year = st.selectbox("Season", ALL_YEARS, index=len(ALL_YEARS)-2, key="sp_year")
    if st.button("Load Season", key="sp_btn"):
        with st.spinner("Loading season data..."):
            races = get_races_for_year(sp_year)
            if races:
                dp = {}
                for idx in range(min(len(races), 10)):
                    standings = get_standings(sp_year, idx + 1)
                    if standings is not None:
                        for _, row in standings.iterrows():
                            driver = row["Driver"].split()[-1]
                            if driver not in dp:
                                dp[driver] = []
                            dp[driver].append(row["Points"])

                if dp:
                    fig, ax = plt.subplots(figsize=(14, 7))
                    cl = ["#B8A04A","#D4D4D4","#A8A8A8","#00D7B6","#4781D7","#ED1131","#F47600","#229971"]
                    top = sorted(dp.items(), key=lambda x: x[1][-1] if x[1] else 0, reverse=True)[:8]
                    for i, (driver, points) in enumerate(top):
                        ax.plot(range(1, len(points)+1), points, marker='o', linewidth=2.5, markersize=5, label=driver, color=cl[i % len(cl)])
                    ax.set_facecolor("#030303")
                    fig.patch.set_facecolor("#030303")
                    ax.tick_params(colors="#707070")
                    ax.legend(facecolor="#0a0a0a", edgecolor="#1d1d1f", labelcolor="#D4D4D4", fontsize=9)
                    for s in ax.spines.values():
                        s.set_color("#1d1d1f")
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.grid(color="#1d1d1f", linewidth=0.5)
                    plt.tight_layout()
                    st.pyplot(fig)


with tabs[12]:
    st.markdown("<div class='page-header'><div class='product-label'>Constructor Cup</div><div class='page-title'>Team championship.</div></div>", unsafe_allow_html=True)
    st.markdown("<hr class='apple-divider-sm'>", unsafe_allow_html=True)

    cc_year = st.selectbox("Season", ALL_YEARS, index=len(ALL_YEARS)-2, key="cc_year")
    if st.button("Load Constructors", key="cc_btn"):
        with st.spinner("Loading..."):
            d = fetch_json(f"https://api.jolpi.ca/ergast/f1/{cc_year}/constructorStandings.json")
            if d:
                try:
                    sl = d["MRData"]["StandingsTable"]["StandingsLists"]
                    if sl:
                        rows = []
                        for s in sl[0]["ConstructorStandings"]:
                            rows.append({"Position": int(s["position"]), "Team": s["Constructor"]["name"], "Points": float(s["points"]), "Wins": int(s["wins"])})
                        df = pd.DataFrame(rows)
                        current_year = datetime.datetime.now().year
                        if cc_year < current_year:
                            label1 = "Champion"
                        else:
                            label1 = "Current Leader"
                        if len(df) >= 3:
                            c1, c2, c3 = st.columns(3)
                            c1.metric(label1, df.iloc[0]["Team"], f"{int(df.iloc[0]['Points'])} pts")
                            c2.metric("Second", df.iloc[1]["Team"], f"{int(df.iloc[1]['Points'])} pts")
                            c3.metric("Third", df.iloc[2]["Team"], f"{int(df.iloc[2]['Points'])} pts")
                        left, right = st.columns(2)
                        with left:
                            st.dataframe(df, use_container_width=True)
                        with right:
                            colors = [get_team_color(row["Team"]) for _, row in df.iterrows()]
                            fig, ax = plt.subplots(figsize=(8, 6))
                            ax.barh(df["Team"][::-1], df["Points"][::-1], color=colors[::-1], height=0.6)
                            ax.set_facecolor("#030303")
                            fig.patch.set_facecolor("#030303")
                            ax.tick_params(colors="#707070")
                            for s in ax.spines.values():
                                s.set_color("#1d1d1f")
                            ax.spines["top"].set_visible(False)
                            ax.spines["right"].set_visible(False)
                            plt.tight_layout()
                            st.pyplot(fig)
                except Exception:
                    st.error("Could not load constructor standings.")


with tabs[13]:
    st.markdown("<div class='page-header'><div class='product-label'>Lap Times</div><div class='page-title'>Lap time analysis.</div></div>", unsafe_allow_html=True)
    st.markdown("<hr class='apple-divider-sm'>", unsafe_allow_html=True)

    lt_c1, lt_c2 = st.columns(2)
    with lt_c1:
        lt_year = st.selectbox("Year", list(range(2018, 2027)), index=6, key="lt_year")
    with lt_c2:
        lt_races = get_races_for_year(lt_year)
        lt_race = st.selectbox("Race", lt_races, key="lt_race") if lt_races else None

    if st.button("Load Laps", key="lt_btn") and lt_race:
        lt_rnd = lt_races.index(lt_race) + 1
        with st.spinner("Downloading lap data..."):
            try:
                session = fastf1.get_session(lt_year, lt_rnd, 'R')
                session.load()
                laps = session.laps
                drivers = session.results["Abbreviation"].tolist()[:8]

                fig, ax = plt.subplots(figsize=(14, 7))
                for driver in drivers:
                    dl = laps.pick_drivers(driver).pick_quicklaps()
                    if not dl.empty:
                        lt_vals = dl["LapTime"].dt.total_seconds()
                        ln = dl["LapNumber"]
                        tc = get_team_color(session.results[session.results["Abbreviation"] == driver]["TeamName"].values[0] if len(session.results[session.results["Abbreviation"] == driver]) > 0 else "")
                        ax.plot(ln, lt_vals, linewidth=1.5, label=driver, color=tc, alpha=0.8)

                ax.set_facecolor("#030303")
                fig.patch.set_facecolor("#030303")
                ax.tick_params(colors="#707070")
                ax.set_xlabel("Lap", color="#707070")
                ax.set_ylabel("Lap Time (s)", color="#707070")
                ax.legend(facecolor="#0a0a0a", edgecolor="#1d1d1f", labelcolor="#D4D4D4", fontsize=9)
                for s in ax.spines.values():
                    s.set_color("#1d1d1f")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.grid(color="#1d1d1f", linewidth=0.5)
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error: {str(e)}")


with tabs[14]:
    st.markdown("<div class='page-header'><div class='product-label'>Points Calculator</div><div class='page-title'>What if?</div><div class='page-subtitle'>Simulate a race result and see championship impact.</div></div>", unsafe_allow_html=True)
    st.markdown("<hr class='apple-divider-sm'>", unsafe_allow_html=True)

    pts_system = {1:25, 2:18, 3:15, 4:12, 5:10, 6:8, 7:6, 8:4, 9:2, 10:1}

    pc_c1, pc_c2 = st.columns(2)
    with pc_c1:
        pc_year = st.selectbox("Season", ALL_YEARS, index=len(ALL_YEARS)-2, key="pc_year")
    pc_races = get_races_for_year(pc_year)
    with pc_c2:
        pc_race = st.selectbox("After Race", pc_races, key="pc_race") if pc_races else None

    if st.button("Load Standings", key="pc_btn") and pc_race:
        rnd = pc_races.index(pc_race) + 1
        standings = get_standings(pc_year, rnd)
        if standings is not None:
            st.session_state["pc_standings"] = standings
            st.dataframe(standings.head(10), use_container_width=True)

    if "pc_standings" in st.session_state:
        st.markdown("<hr class='apple-divider-sm'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title-center'>Simulate Next Race</div>", unsafe_allow_html=True)

        standings = st.session_state["pc_standings"]
        top5 = standings.head(5)
        sim_results = {}
        sim_cols = st.columns(5)
        for i, (_, row) in enumerate(top5.iterrows()):
            with sim_cols[i]:
                sim_results[row["Driver"]] = st.selectbox(row["Driver"].split()[-1], list(range(1, 21)), index=i, key=f"sim_{i}")

        if st.button("Calculate", key="pc_calc"):
            results = []
            for _, row in standings.iterrows():
                gained = pts_system.get(sim_results.get(row["Driver"], 99), 0)
                results.append({"Driver": row["Driver"], "Team": row["Team"], "Current": row["Points"], "Gained": gained, "New Total": row["Points"] + gained})

            rdf = pd.DataFrame(results).sort_values("New Total", ascending=False)
            for _, r in rdf.head(5).iterrows():
                clr = "#00D7B6" if r["Gained"] > 0 else "#4a4a4a"
                st.markdown(
                    "<div class='info-card'>"
                    "<div style='display:flex;justify-content:space-between;align-items:center;'>"
                    "<div>"
                    "<div class='info-card-title'>" + r["Team"] + "</div>"
                    "<div class='info-card-value' style='font-size:22px;'>" + r["Driver"] + "</div>"
                    "</div>"
                    "<div style='text-align:right;'>"
                    "<div style='color:" + clr + ";font-size:13px;'>+" + str(int(r["Gained"])) + " pts</div>"
                    "<div class='info-card-value' style='font-size:26px;'>" + str(int(r["New Total"])) + "</div>"
                    "</div>"
                    "</div>"
                    "</div>",
                    unsafe_allow_html=True
                )


with tabs[15]:
    st.markdown("<div class='page-header'><div class='product-label'>Race Rewind</div><div class='page-title'>Relive any race.</div><div class='page-subtitle'>Lap by lap position changes through data.</div></div>", unsafe_allow_html=True)
    st.markdown("<hr class='apple-divider-sm'>", unsafe_allow_html=True)

    rw_c1, rw_c2 = st.columns(2)
    with rw_c1:
        rw_year = st.selectbox("Year", list(range(2018, 2027)), index=6, key="rw_year")
    with rw_c2:
        rw_races = get_races_for_year(rw_year)
        rw_race = st.selectbox("Race", rw_races, key="rw_race") if rw_races else None

    if st.button("Rewind Race", key="rw_btn") and rw_race:
        rw_rnd = rw_races.index(rw_race) + 1
        with st.spinner("Loading race data..."):
            try:
                session = fastf1.get_session(rw_year, rw_rnd, 'R')
                session.load()
                laps = session.laps
                drivers = session.results["Abbreviation"].tolist()[:10]

                fig, ax = plt.subplots(figsize=(14, 8))
                for driver in drivers:
                    dl = laps.pick_drivers(driver)
                    if not dl.empty:
                        tc = get_team_color(session.results[session.results["Abbreviation"] == driver]["TeamName"].values[0] if len(session.results[session.results["Abbreviation"] == driver]) > 0 else "")
                        ax.plot(dl["LapNumber"], dl["Position"], linewidth=2, label=driver, color=tc)

                ax.set_facecolor("#030303")
                fig.patch.set_facecolor("#030303")
                ax.tick_params(colors="#707070")
                ax.set_xlabel("Lap", color="#707070")
                ax.set_ylabel("Position", color="#707070")
                ax.invert_yaxis()
                ax.legend(facecolor="#0a0a0a", edgecolor="#1d1d1f", labelcolor="#D4D4D4", fontsize=9, ncol=5, loc="upper center")
                for s in ax.spines.values():
                    s.set_color("#1d1d1f")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.grid(color="#1d1d1f", linewidth=0.5)
                plt.tight_layout()
                st.pyplot(fig)

                st.markdown("<hr class='apple-divider-sm'>", unsafe_allow_html=True)
                st.dataframe(session.results[["Abbreviation", "TeamName", "GridPosition", "Position", "Points"]].head(10), use_container_width=True)
            except Exception as e:
                st.error(f"Error: {str(e)}")


with tabs[16]:
    st.markdown("<div class='page-header'><div class='product-label'>Form Guide</div><div class='page-title'>Who is hot. Who is not.</div><div class='page-subtitle'>Driver performance trends from the last 5 races.</div></div>", unsafe_allow_html=True)
    st.markdown("<hr class='apple-divider-sm'>", unsafe_allow_html=True)

    fg_year = st.selectbox("Season", ALL_YEARS, index=len(ALL_YEARS)-2, key="fg_year")
    if st.button("Analyze Form", key="fg_btn"):
        with st.spinner("Analyzing..."):
            races = get_races_for_year(fg_year)
            if races:
                last5 = races[-5:] if len(races) >= 5 else races
                driver_form = {}
                for rname in last5:
                    rnd = races.index(rname) + 1
                    df, _, _ = get_results(fg_year, rnd)
                    if df is not None:
                        for _, row in df.iterrows():
                            driver = row["Driver"]
                            if driver not in driver_form:
                                driver_form[driver] = {"positions": [], "points": 0, "team": row["Team"]}
                            driver_form[driver]["positions"].append(row["Position"])
                            driver_form[driver]["points"] += row["Points"]

                if driver_form:
                    form_data = []
                    for driver, data in driver_form.items():
                        avg = sum(data["positions"]) / len(data["positions"])
                        trend = data["positions"][0] - data["positions"][-1] if len(data["positions"]) > 1 else 0
                        form_data.append({"driver": driver, "team": data["team"], "avg": avg, "points": data["points"], "trend": trend, "positions": data["positions"]})

                    form_data.sort(key=lambda x: x["avg"])

                    for fd in form_data[:15]:
                        tc = get_team_color(fd["team"])

                        if fd["trend"] > 0:
                            trend_text = "IMPROVING"
                            trend_color = "#00D7B6"
                        elif fd["trend"] < 0:
                            trend_text = "DECLINING"
                            trend_color = "#ED1131"
                        else:
                            trend_text = "STABLE"
                            trend_color = "#B8A04A"

                        if fd["avg"] <= 3:
                            form_label = "ON FIRE"
                            form_color = "#00D7B6"
                        elif fd["avg"] <= 6:
                            form_label = "STRONG"
                            form_color = "#B8A04A"
                        elif fd["avg"] <= 10:
                            form_label = "SOLID"
                            form_color = "#707070"
                        else:
                            form_label = "STRUGGLING"
                            form_color = "#ED1131"

                        positions_str = " - ".join(["P" + str(int(p)) for p in fd["positions"]])
                        avg_str = str(round(fd["avg"], 1))
                        pts_str = str(int(fd["points"]))
with tabs[17]:
    st.markdown("<div class='hero'><div class='hero-tag'>ABOUT</div><div class='hero-title'>F1R</div><div class='hero-underline'></div><div class='hero-subtitle'>The Formula 1 Repository</div></div>", unsafe_allow_html=True)

    st.markdown("<hr class='apple-divider'>", unsafe_allow_html=True)
    st.markdown("<div class='product-section'><div class='product-label'>Our Mission</div><div class='product-title'>Making F1 data<br>accessible to everyone.</div><div class='product-description'>F1Repository was built to give every F1 fan access to professional-level data and intelligence. From 1950 to today, every race, every driver, every stat in one place.</div></div>", unsafe_allow_html=True)

    st.markdown("<hr class='apple-divider'>", unsafe_allow_html=True)
    st.markdown("<div class='product-section'><div class='product-label'>The Platform</div><div class='product-title'>17 features.<br>One platform.</div><div class='product-description'>Race analysis, AI predictions, driver comparisons, championship standings, circuit guides, lap times, form guides, and an AI companion named Roman. Everything a Formula 1 fan needs.</div></div>", unsafe_allow_html=True)

    st.markdown("<hr class='apple-divider'>", unsafe_allow_html=True)
    st.markdown("<div class='product-section'><div class='product-label'>Meet Roman</div><div class='product-title'>Your F1<br>companion.</div><div class='product-description'>Roman is our AI-powered F1 assistant. Named after a very good dog, Roman knows everything about F1 from 1950 to today. Ask him anything.</div></div>", unsafe_allow_html=True)

    st.markdown("<hr class='apple-divider'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title-center'>By The Numbers</div>", unsafe_allow_html=True)
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.markdown("<div class='info-card' style='text-align:center;'><div class='info-card-value'>75+</div><div class='info-card-title'>Years</div></div>", unsafe_allow_html=True)
    c2.markdown("<div class='info-card' style='text-align:center;'><div class='info-card-value'>1100+</div><div class='info-card-title'>Races</div></div>", unsafe_allow_html=True)
    c3.markdown("<div class='info-card' style='text-align:center;'><div class='info-card-value'>860+</div><div class='info-card-title'>Drivers</div></div>", unsafe_allow_html=True)
    c4.markdown("<div class='info-card' style='text-align:center;'><div class='info-card-value'>24</div><div class='info-card-title'>Circuits</div></div>", unsafe_allow_html=True)
    c5.markdown("<div class='info-card' style='text-align:center;'><div class='info-card-value'>17</div><div class='info-card-title'>Features</div></div>", unsafe_allow_html=True)

    st.markdown("<hr class='apple-divider'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title-center'>Technology</div>", unsafe_allow_html=True)
    t1,t2,t3 = st.columns(3)
    t1.markdown("<div class='feature-card'><div class='feature-name'>Python</div><div class='feature-desc'>Core language.</div></div>", unsafe_allow_html=True)
    t2.markdown("<div class='feature-card'><div class='feature-name'>FastF1</div><div class='feature-desc'>Race telemetry.</div></div>", unsafe_allow_html=True)
    t3.markdown("<div class='feature-card'><div class='feature-name'>Jolpica API</div><div class='feature-desc'>Historical data.</div></div>", unsafe_allow_html=True)
    st.write("")
    t4,t5,t6 = st.columns(3)
    t4.markdown("<div class='feature-card'><div class='feature-name'>Scikit-Learn</div><div class='feature-desc'>ML predictions.</div></div>", unsafe_allow_html=True)
    t5.markdown("<div class='feature-card'><div class='feature-name'>Groq AI</div><div class='feature-desc'>Powers Roman.</div></div>", unsafe_allow_html=True)
    t6.markdown("<div class='feature-card'><div class='feature-name'>Streamlit</div><div class='feature-desc'>Web framework.</div></div>", unsafe_allow_html=True)

    st.markdown("<hr class='apple-divider'>", unsafe_allow_html=True)
    st.markdown("<div class='product-section'><div class='product-label'>Connect</div><div class='product-title'>Follow us.</div><div class='product-description'>Follow F1Repository on Instagram for daily F1 stats and exclusive content.</div></div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center;padding:40px;'><div class='info-card' style='max-width:400px;margin:0 auto;'><div class='info-card-title'>Instagram</div><div class='info-card-value' style='font-size:24px;'>@f1repository</div></div></div>", unsafe_allow_html=True)

    st.markdown("<hr class='apple-divider'>", unsafe_allow_html=True)
    st.markdown("<div class='product-section'><div class='product-label'>Legal</div><div class='product-title'>Disclaimer</div><div class='product-description'>F1Repository is an independent platform not affiliated with Formula 1, FIA, or any F1 team. All data from publicly available APIs. A fan-made project for the F1 community.</div></div>", unsafe_allow_html=True)
