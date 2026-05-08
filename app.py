import os
import datetime
import requests
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import fastf1
from sklearn.linear_model import LinearRegression

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

GROQ_KEY = os.getenv("GROQ_API_KEY", "paste_your_groq_key_here")

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

RIP_DRIVERS = ["Ayrton Senna", "Gilles Villeneuve", "Jim Clark", "Jochen Rindt", "Bruce McLaren", "Francois Cevert", "Ronnie Peterson", "Wolfgang von Trips", "Luigi Musso", "Peter Collins", "Alberto Ascari", "John Surtees", "James Hunt", "Niki Lauda", "Michele Alboreto", "Elio de Angelis", "Roland Ratzenberger", "Jules Bianchi", "Stefan Bellof", "Tom Pryce", "Carlos Pace", "Roger Williamson"]

HEAD_TO_HEAD_DRIVERS = ["Lewis Hamilton", "Michael Schumacher", "Max Verstappen", "Ayrton Senna", "Sebastian Vettel", "Fernando Alonso", "Charles Leclerc", "Lando Norris", "Carlos Sainz", "George Russell", "Oscar Piastri", "Alain Prost", "Niki Lauda", "Kimi Raikkonen", "Mika Hakkinen", "Juan Manuel Fangio", "Jim Clark", "Jackie Stewart", "Nigel Mansell", "Gilles Villeneuve", "Jenson Button", "Daniel Ricciardo", "Valtteri Bottas", "Jules Bianchi"]


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


css = """
<style>
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

.stTabs [data-baseweb="tab-list"]{
    display:flex;
    justify-content:flex-start;
    overflow-x:auto;
    overflow-y:hidden;
    background:rgba(3,3,3,0.98);
    backdrop-filter:blur(30px);
    border-bottom:1px solid rgba(255,255,255,0.025);
    padding:0;
    scrollbar-width:thin;
    scrollbar-color:#B8A04A #030303;
    -webkit-overflow-scrolling:touch;
}

.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar{height:4px;}
.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-track{background:#030303;}
.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-thumb{background:#B8A04A;border-radius:2px;}

.stTabs [data-baseweb="tab"]{
    color:var(--muted);
    font-size:11px;
    font-weight:300;
    letter-spacing:2px;
    text-transform:uppercase;
    padding:18px 16px;
    background:transparent;
    border:none;
    border-bottom:1px solid transparent;
    white-space:nowrap;
    flex-shrink:0;
}

.stTabs [data-baseweb="tab"]:hover{color:var(--platinum);}
.stTabs [aria-selected="true"]{
    color:var(--gold)!important;
    border-bottom:1px solid rgba(184,160,74,0.4)!important;
    font-weight:400;
}

.hero{
    text-align:center;
    padding:120px 20px 100px;
    max-width:1000px;
    margin:0 auto;
}

.hero-tag{
    font-size:9px;
    color:var(--gold);
    font-weight:300;
    letter-spacing:8px;
    text-transform:uppercase;
    margin-bottom:30px;
    opacity:0.6;
}

.hero-title{
    font-family:'Cormorant Garamond',serif;
    font-size:120px;
    font-weight:300;
    letter-spacing:-6px;
    line-height:0.85;
    margin-bottom:20px;
    color:#D4D4D4;
}

.hero-underline{
    width:40px;
    height:1px;
    background:var(--gold);
    margin:30px auto;
    opacity:0.3;
}

.hero-subtitle{
    font-size:11px;
    font-weight:200;
    color:var(--muted);
    letter-spacing:6px;
    text-transform:uppercase;
}

.stats-bar{
    display:flex;
    justify-content:center;
    flex-wrap:wrap;
    gap:60px;
    padding:60px 20px;
    max-width:1100px;
    margin:0 auto;
    border-top:1px solid var(--border);
    border-bottom:1px solid var(--border);
}

.stat-item{text-align:center;}
.stat-value{
    font-family:'Cormorant Garamond',serif;
    font-size:48px;
    font-weight:300;
    color:var(--platinum);
    letter-spacing:-2px;
}

.stat-label{
    font-size:8px;
    color:var(--muted);
    text-transform:uppercase;
    letter-spacing:4px;
    margin-top:12px;
    font-weight:300;
}

.product-section{
    text-align:center;
    padding:100px 20px;
    max-width:650px;
    margin:0 auto;
}

.product-label{
    font-size:9px;
    color:var(--gold);
    font-weight:300;
    letter-spacing:6px;
    text-transform:uppercase;
    margin-bottom:24px;
    opacity:0.6;
}

.product-title{
    font-family:'Cormorant Garamond',serif;
    font-size:42px;
    font-weight:400;
    color:var(--text);
    letter-spacing:-1px;
    line-height:1.12;
    margin-bottom:24px;
}

.product-description{
    font-size:14px;
    color:var(--muted);
    max-width:440px;
    margin:0 auto;
    line-height:1.9;
    font-weight:200;
}

.page-header{
    text-align:center;
    padding:80px 20px 40px;
    max-width:800px;
    margin:0 auto;
}

.page-title{
    font-family:'Cormorant Garamond',serif;
    font-size:38px;
    font-weight:400;
    color:var(--text);
    letter-spacing:-1px;
    margin-bottom:16px;
}

.page-subtitle{
    font-size:13px;
    color:var(--muted);
    font-weight:200;
    line-height:1.7;
}

.section-title-center{
    color:var(--gold);
    font-size:9px;
    font-weight:300;
    letter-spacing:5px;
    text-transform:uppercase;
    margin-bottom:28px;
    text-align:center;
    opacity:0.6;
}

.feature-card,.info-card,.record-card,.step-card{
    background:var(--card);
    border:1px solid var(--border);
    border-radius:12px;
}

.feature-card{
    padding:36px 24px;
    text-align:center;
    height:100%;
}

.feature-card:hover,.info-card:hover,.record-card:hover{
    background:var(--card-hover);
}

.feature-name{
    font-family:'Cormorant Garamond',serif;
    font-size:18px;
    font-weight:400;
    color:var(--text);
    margin-bottom:14px;
}

.feature-desc,.info-card-desc,.record-holder{
    font-size:12px;
    color:var(--muted);
    line-height:1.7;
    font-weight:200;
}

.info-card,.record-card{
    padding:24px;
    margin-bottom:8px;
}

.info-card-title,.record-title{
    color:var(--gold);
    font-size:8px;
    font-weight:300;
    letter-spacing:4px;
    text-transform:uppercase;
    margin-bottom:10px;
    opacity:0.6;
}

.info-card-value{
    font-family:'Cormorant Garamond',serif;
    color:var(--text);
    font-size:30px;
    font-weight:400;
}

.record-value{
    font-family:'Cormorant Garamond',serif;
    color:var(--text);
    font-size:38px;
    font-weight:300;
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
    padding:14px 40px;
    font-size:9px;
    font-weight:300;
    letter-spacing:5px;
    text-transform:uppercase;
}

.stButton>button:hover{
    background:rgba(184,160,74,0.08);
    border-color:rgba(184,160,74,0.4);
}

.stMetric{
    background:var(--card);
    padding:18px;
    border-radius:12px;
    border:1px solid var(--border);
}

.apple-divider,.apple-divider-sm{
    border:none;
    height:1px;
    background:var(--border);
}

.apple-divider{margin:0;}
.apple-divider-sm{margin:50px 0;}

.clean-footer{
    text-align:center;
    padding:80px 20px;
    border-top:1px solid var(--border);
    margin-top:120px;
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

@media (max-width: 768px) {
    .hero-title{font-size:70px !important; letter-spacing:-3px !important;}
    .product-title{font-size:28px !important;}
    .product-description{font-size:13px !important; padding:0 16px !important;}
    .stats-bar{flex-direction:column !important; gap:25px !important; padding:40px 20px !important;}
    .stat-value{font-size:36px !important;}
    .hero{padding:80px 16px 60px !important;}
    .product-section{padding:60px 16px !important;}
    .page-title{font-size:28px !important;}
    .page-header{padding:60px 16px 30px !important;}
    .stTabs [data-baseweb="tab"]{font-size:9px !important; padding:14px 10px !important; letter-spacing:1px !important;}
    .feature-name{font-size:16px !important;}
    .info-card-value{font-size:24px !important;}
    .record-value{font-size:30px !important;}
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

tabs = st.tabs(["Home","◈ Races","◉ AI Lab","∅ Drivers","◆ Teams","△ Standings","✦ Records","⚔ Head to Head","◎ Roman","⏱ Race Weekend","◇ Circuits","About"])

with tabs[0]:
    st.markdown("""
    <div class='hero'>
        <div class='hero-tag'>THE FORMULA 1 REPOSITORY</div>
        <div class='hero-title'>F1R</div>
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
        ("Race Analysis", "Every race.<br>Every detail.", "Race results, points, and visual breakdowns from across Formula 1 history."),
        ("AI Predictor", "Predict the<br>unpredictable.", "Train a machine learning model on real race data."),
        ("Driver Intelligence", "Know your driver.", "Performance, points, and race outcomes in one clean view."),
        ("Roman", "Your F1<br>companion.", "An AI assistant that knows everything about Formula 1."),
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

with tabs[1]:
    st.markdown("<div class='page-header'><div class='product-label'>Race Analysis</div><div class='page-title'>Explore any Grand Prix.</div></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        ry = st.selectbox("Year", ALL_YEARS, index=len(ALL_YEARS)-2, key="ry")
    rl = get_races_for_year(ry)
    with col2:
        rn = st.selectbox("Grand Prix", rl, key="rn") if rl else None
    if st.button("Analyze Race", key="rb") and rn:
        rnd = rl.index(rn) + 1
        if ry >= 2018:
            try:
                session = fastf1.get_session(ry, rnd, 'R')
                session.load()
                st.markdown(f"<div class='page-header'><div class='page-title'>{ry} {rn}</div></div>", unsafe_allow_html=True)
                c1,c2,c3 = st.columns(3)
                c1.metric("Winner", session.results.iloc[0]["Abbreviation"])
                c2.metric("Second", session.results.iloc[1]["Abbreviation"])
                c3.metric("Third", session.results.iloc[2]["Abbreviation"])
                st.dataframe(session.results[["Abbreviation","TeamName","Position","Points"]], use_container_width=True)
            except:
                st.error("Could not load this race.")
        else:
            df, fn, ci = get_results(ry, rnd)
            if df is not None:
                st.markdown(f"<div class='page-header'><div class='page-title'>{ry} {fn}</div></div>", unsafe_allow_html=True)
                c1,c2,c3 = st.columns(3)
                c1.metric("Winner", df.iloc[0]["Driver"])
                c2.metric("Second", df.iloc[1]["Driver"])
                c3.metric("Third", df.iloc[2]["Driver"])
                st.dataframe(df, use_container_width=True)

with tabs[2]:
    st.markdown("<div class='page-header'><div class='product-label'>AI Predictor</div><div class='page-title'>Predict the unpredictable.</div></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        ay = st.selectbox("Year", list(range(2018, 2027)), index=6, key="ay")
    al = get_races_for_year(ay)
    with col2:
        ar = st.selectbox("Race", al, key="ar") if al else None
    if st.button("Train AI", key="tb") and ar:
        rnd = al.index(ar) + 1
        try:
            session = fastf1.get_session(ay, rnd, 'R')
            session.load()
            data = session.results[["GridPosition","Position"]].dropna()
            model = LinearRegression()
            model.fit(data[["GridPosition"]], data["Position"])
            st.session_state["model"] = model
            st.success(f"AI trained on {ay} {ar}")
        except:
            st.error("Could not train.")
    if "model" in st.session_state:
        grid = st.slider("Grid Position", 1, 20, 1, key="gp")
        pred = max(1, min(20, round(st.session_state["model"].predict([[grid]])[0])))
        c1,c2 = st.columns(2)
        c1.metric("Grid", f"P{grid}")
        c2.metric("Predicted", f"P{pred}")

with tabs[3]:
    st.markdown("<div class='page-header'><div class='product-label'>Drivers</div><div class='page-title'>Know your driver.</div></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        dy = st.selectbox("Year", ALL_YEARS, index=len(ALL_YEARS)-2, key="dy")
    dl = get_races_for_year(dy)
    with col2:
        dr = st.selectbox("Race", dl, key="dr") if dl else None
    if st.button("Load Drivers", key="dl_btn") and dr:
        rnd = dl.index(dr) + 1
        if dy >= 2018:
            try:
                session = fastf1.get_session(dy, rnd, 'R')
                session.load()
                st.session_state["d_session"] = session
                st.session_state["d_drivers"] = session.results["Abbreviation"].tolist()
                st.success("Select a driver.")
            except:
                st.error("Error.")
    if "d_drivers" in st.session_state:
        driver = st.selectbox("Driver", st.session_state["d_drivers"], key="ds")
        if st.button("Show Stats", key="db"):
            result = st.session_state["d_session"].results[st.session_state["d_session"].results["Abbreviation"] == driver]
            if not result.empty:
                row = result.iloc[0]
                tc = get_team_color(row["TeamName"])
                st.markdown(f"<div class='page-header'><div class='page-title' style='color:{tc};'>{driver}</div><div class='page-subtitle'>{row['TeamName']}</div></div>", unsafe_allow_html=True)
                c1,c2,c3 = st.columns(3)
                c1.metric("Grid", f"P{int(row['GridPosition'])}")
                c2.metric("Finish", f"P{int(row['Position'])}")
                c3.metric("Points", int(row["Points"]))

with tabs[4]:
    st.markdown("<div class='page-header'><div class='product-label'>Teams</div><div class='page-title'>Team intelligence.</div></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        ty = st.selectbox("Year", ALL_YEARS, index=len(ALL_YEARS)-2, key="ty")
    tl = get_races_for_year(ty)
    with col2:
        tr_sel = st.selectbox("Race", tl, key="tr") if tl else None
    if st.button("Load Teams", key="tl_btn") and tr_sel:
        rnd = tl.index(tr_sel) + 1
        if ty >= 2018:
            try:
                session = fastf1.get_session(ty, rnd, 'R')
                session.load()
                st.session_state["t_session"] = session
                st.session_state["t_teams"] = session.results["TeamName"].unique().tolist()
                st.success("Select a team.")
            except:
                st.error("Error.")
    if "t_teams" in st.session_state:
        team = st.selectbox("Team", st.session_state["t_teams"], key="ts")
        if st.button("Show Team", key="tsb"):
            tr2 = st.session_state["t_session"].results[st.session_state["t_session"].results["TeamName"] == team]
            if not tr2.empty:
                tc = get_team_color(team)
                st.markdown(f"<div class='page-header'><div class='page-title' style='color:{tc};'>{team}</div></div>", unsafe_allow_html=True)
                c1,c2,c3 = st.columns(3)
                c1.metric("Drivers", len(tr2))
                c2.metric("Points", int(tr2["Points"].sum()))
                c3.metric("Best", f"P{int(tr2['Position'].min())}")

with tabs[5]:
    st.markdown("<div class='page-header'><div class='product-label'>Championships</div><div class='page-title'>Season standings.</div></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        sy = st.selectbox("Year", ALL_YEARS, index=len(ALL_YEARS)-2, key="sy")
    sl = get_races_for_year(sy)
    with col2:
        sr = st.selectbox("After Race", sl, key="sr") if sl else None
    if st.button("Load Standings", key="sb") and sr:
        rnd = sl.index(sr) + 1
        standings = get_standings(sy, rnd)
        if standings is not None and not standings.empty:
            c1,c2,c3 = st.columns(3)
            c1.metric("Leader", standings.iloc[0]["Driver"], f"{int(standings.iloc[0]['Points'])} pts")
            c2.metric("Second", standings.iloc[1]["Driver"], f"{int(standings.iloc[1]['Points'])} pts")
            c3.metric("Third", standings.iloc[2]["Driver"], f"{int(standings.iloc[2]['Points'])} pts")
            st.dataframe(standings, use_container_width=True)

with tabs[6]:
    st.markdown("<div class='page-header'><div class='product-label'>Hall of Fame</div><div class='page-title'>Records.</div></div>", unsafe_allow_html=True)
    records = {
        "Driver": [("MOST CHAMPIONSHIPS","7","Hamilton & Schumacher"),("MOST WINS","103","Hamilton"),("MOST PODIUMS","197","Hamilton"),("MOST POLES","104","Hamilton")],
        "Constructor": [("MOST TITLES","16","Ferrari"),("MOST WINS","243","Ferrari"),("MOST DOMINANT","21/22","Red Bull (2023)")],
    }
    for section, items in records.items():
        st.markdown(f"<div class='section-title-center'>{section} Records</div>", unsafe_allow_html=True)
        cols = st.columns(3)
        for i, (t, v, h) in enumerate(items):
            cols[i % 3].markdown(f"<div class='record-card'><div class='record-title'>{t}</div><div class='record-value'>{v}</div><div class='record-holder'>{h}</div></div>", unsafe_allow_html=True)

with tabs[7]:
    st.markdown("<div class='page-header'><div class='product-label'>Head to Head</div><div class='page-title'>The ultimate battle.</div></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        d1n = st.selectbox("Driver 1", HEAD_TO_HEAD_DRIVERS, index=0, key="d1n")
    with col2:
        d2n = st.selectbox("Driver 2", HEAD_TO_HEAD_DRIVERS, index=1, key="d2n")
    st.info("Career comparison coming soon. Feature is in active development.")

with tabs[8]:
    st.markdown("<div class='page-header'><div class='product-label'>Meet Roman</div><div class='page-title'>Your F1 companion.</div></div>", unsafe_allow_html=True)
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    for msg in st.session_state["chat_history"]:
        border = "#B8A04A" if msg["role"] == "user" else "#00D7B6"
        label = "YOU" if msg["role"] == "user" else "ROMAN"
        st.markdown(f"<div class='info-card' style='border-left:3px solid {border};'><div class='info-card-title'>{label}</div><div class='info-card-desc' style='color:#E8E8E8;font-size:14px;'>{msg['content']}</div></div>", unsafe_allow_html=True)
    user_input = st.text_input("Ask anything about F1...", key="roman_input")
    if st.button("Ask Roman", key="roman_btn") and user_input:
        if not GROQ_KEY or GROQ_KEY == "paste_your_groq_key_here":
            st.error("API key not set.")
        elif Groq is None:
            st.error("Groq not installed.")
        else:
            try:
                client = Groq(api_key=GROQ_KEY)
                result = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are Roman, an expert F1 assistant. Be concise and factual."},
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

with tabs[9]:
    st.markdown("<div class='page-header'><div class='product-label'>Race Weekend</div><div class='page-title'>Race Countdown</div></div>", unsafe_allow_html=True)
        CAL = [
        {"name":"Australian GP","loc":"Melbourne","circuit":"Albert Park","round":1,"date":"2026-03-08 04:00","cancelled":False,"reason":""},
        {"name":"Chinese GP","loc":"Shanghai","circuit":"Shanghai","round":2,"date":"2026-03-15 07:00","cancelled":False,"reason":""},
        {"name":"Japanese GP","loc":"Suzuka","circuit":"Suzuka","round":3,"date":"2026-03-29 05:00","cancelled":False,"reason":""},
        {"name":"Bahrain GP","loc":"Sakhir","circuit":"Bahrain","round":4,"date":"2026-04-12 15:00","cancelled":True,"reason":"Cancelled - West Asia conflicts"},
        {"name":"Saudi Arabian GP","loc":"Jeddah","circuit":"Jeddah","round":5,"date":"2026-04-19 17:00","cancelled":True,"reason":"Cancelled - West Asia conflicts"},
        {"name":"Miami GP","loc":"Miami","circuit":"Miami Autodrome","round":6,"date":"2026-05-03 19:30","cancelled":False,"reason":""},
        {"name":"Canadian GP","loc":"Montreal","circuit":"Gilles Villeneuve","round":7,"date":"2026-05-24 18:00","cancelled":False,"reason":""},
        {"name":"Monaco GP","loc":"Monte Carlo","circuit":"Monaco","round":8,"date":"2026-06-07 13:00","cancelled":False,"reason":""},
        {"name":"Spanish GP","loc":"Barcelona","circuit":"Catalunya","round":9,"date":"2026-06-14 13:00","cancelled":False,"reason":""},
        {"name":"Austrian GP","loc":"Spielberg","circuit":"Red Bull Ring","round":10,"date":"2026-06-28 13:00","cancelled":False,"reason":""},
        {"name":"British GP","loc":"Silverstone","circuit":"Silverstone","round":11,"date":"2026-07-05 14:00","cancelled":False,"reason":""},
        {"name":"Belgian GP","loc":"Spa","circuit":"Spa-Francorchamps","round":12,"date":"2026-07-19 13:00","cancelled":False,"reason":""},
        {"name":"Hungarian GP","loc":"Budapest","circuit":"Hungaroring","round":13,"date":"2026-07-26 13:00","cancelled":False,"reason":""},
        {"name":"Dutch GP","loc":"Zandvoort","circuit":"Zandvoort","round":14,"date":"2026-08-23 13:00","cancelled":False,"reason":""},
        {"name":"Italian GP","loc":"Monza","circuit":"Monza","round":15,"date":"2026-09-06 13:00","cancelled":False,"reason":""},
        {"name":"Madrid GP","loc":"Madrid","circuit":"Madrid Street Circuit","round":16,"date":"2026-09-13 13:00","cancelled":False,"reason":""},
        {"name":"Azerbaijan GP","loc":"Baku","circuit":"Baku City Circuit","round":17,"date":"2026-09-27 11:00","cancelled":False,"reason":""},
        {"name":"Singapore GP","loc":"Singapore","circuit":"Marina Bay","round":18,"date":"2026-10-11 12:00","cancelled":False,"reason":""},
        {"name":"US GP","loc":"Austin","circuit":"COTA","round":19,"date":"2026-10-25 19:00","cancelled":False,"reason":""},
        {"name":"Mexico GP","loc":"Mexico City","circuit":"Hermanos Rodriguez","round":20,"date":"2026-11-01 20:00","cancelled":False,"reason":""},
        {"name":"Sao Paulo GP","loc":"Sao Paulo","circuit":"Interlagos","round":21,"date":"2026-11-08 17:00","cancelled":False,"reason":""},
        {"name":"Las Vegas GP","loc":"Las Vegas","circuit":"Las Vegas Strip","round":22,"date":"2026-11-22 06:00","cancelled":False,"reason":""},
        {"name":"Qatar GP","loc":"Lusail","circuit":"Lusail","round":23,"date":"2026-11-29 17:00","cancelled":False,"reason":""},
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
        st.markdown(f"<div style='text-align:center;padding:30px;'><div class='product-label'>Next Race</div><div class='page-title'>{next_race['name']}</div></div>", unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        c1.metric("Days", diff.days)
        c2.metric("Hours", diff.seconds // 3600)
        c3.metric("Round", f"R{next_race['round']}")
        for race in CAL:
            rdt = datetime.datetime.strptime(race["date"], "%Y-%m-%d %H:%M")
            ic = race.get("cancelled", False)
            color = "#ED1131" if ic else ("#B8A04A" if next_race and race["name"] == next_race["name"] else ("#4a4a4a" if rdt < now else "#E8E8E8"))
            status = "CANCELLED" if ic else ("NEXT" if next_race and race["name"] == next_race["name"] else ("DONE" if rdt < now else "UPCOMING"))
            st.markdown(f"<div class='info-card' style='border-left:3px solid {color};'><div class='info-card-title'>R{race['round']} | {status}</div><div class='info-card-value' style='font-size:20px;color:{color};'>{race['name']}</div></div>", unsafe_allow_html=True)

with tabs[10]:
    st.markdown("<div class='page-header'><div class='product-label'>Circuit Guide</div><div class='page-title'>Every track.</div></div>", unsafe_allow_html=True)
    CIRCUITS = [
        {"name":"Monaco","loc":"Monte Carlo","length":"3.337 km","laps":78,"desc":"The crown jewel of F1."},
        {"name":"Silverstone","loc":"UK","length":"5.891 km","laps":52,"desc":"Home of F1 since 1950."},
        {"name":"Spa","loc":"Belgium","length":"7.004 km","laps":44,"desc":"The greatest circuit. Eau Rouge."},
        {"name":"Monza","loc":"Italy","length":"5.793 km","laps":53,"desc":"Temple of Speed."},
        {"name":"Suzuka","loc":"Japan","length":"5.807 km","laps":53,"desc":"Figure-eight with 130R."},
    ]
    for c in CIRCUITS:
        st.markdown(f"<div class='info-card'><div class='info-card-value' style='font-size:22px;'>{c['name']}</div><div class='info-card-desc'>{c['loc']} | {c['length']} | {c['laps']} laps</div><div class='info-card-desc' style='margin-top:6px;'>{c['desc']}</div></div>", unsafe_allow_html=True)

with tabs[11]:
    st.markdown("<div class='page-header'><div class='product-label'>About</div><div class='page-title'>F1Repository</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='product-section'><div class='product-label'>Mission</div><div class='product-title'>F1 data for everyone.</div><div class='product-description'>Built to give every fan professional-level F1 data. From 1950 to today.</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='product-section'><div class='product-label'>Roman</div><div class='product-title'>Named after a good dog.</div><div class='product-description'>Our AI assistant knows everything about F1. Ask him anything.</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='product-section'><div class='product-label'>Connect</div><div class='product-title'>@f1repository</div><div class='product-description'>Follow on Instagram for daily F1 content.</div></div>", unsafe_allow_html=True)

st.markdown("""
<div class='clean-footer'>
    <p style='font-family:Cormorant Garamond,serif;color:#B8A04A;font-weight:300;font-size:22px;letter-spacing:8px;'>F1R</p>
    <p style='color:#4a4a4a;margin-top:14px;font-size:8px;letter-spacing:5px;text-transform:uppercase;'>The Formula 1 Repository</p>
    <p style='color:#2a2a2a;margin-top:6px;font-size:8px;letter-spacing:4px;'>1950 - 2026</p>
</div>
""", unsafe_allow_html=True)