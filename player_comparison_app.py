import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ─── PAGE CONFIG ───
st.set_page_config(
    page_title="⚽ Football Data",
    page_icon="🏟️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── GLOBAL SETTINGS ───
GLOBAL_FEATURES = [
    'Gls','Ast','xG','xAG','KP','SCA90','GCA90','Sh/90','SoT/90',
    'Succ','Touches','xA','PrgP','PrgC','PrgR','Carries','1/3',
    'PPA','TotDist','TB','Sw','Tkl','Tkl+Int','Blocks','Clr','Err','Int',
    'GA90','PSxG+/-','Cmp%_stats_keeper_adv','AvgDist'
]
scatter_axes_by_role = {
    "FW": ("xG","xAG"),
    "MF": ("PrgP","KP"),
    "DF": ("Tkl","Int"),
    "GK": ("Save%","PSxG+/-")
}
position_feature_map = {
    "FW": ['Gls','Ast','xG','xAG','KP','SCA90','GCA90','Sh/90','SoT/90','Succ','Touches','xA'],
    "MF": ['Ast','xAG','xA','KP','SCA90','GCA90','PrgP','PrgC','PrgR','Carries','Touches','1/3','PPA','TotDist','TB','Sw'],
    "DF": ['Tkl','Tkl+Int','Blocks','Clr','Err','Int','Touches','PrgP','PrgC','Carries'],
    "GK": ['GA90','PSxG+/-','Cmp%_stats_keeper_adv','AvgDist']
}

# ─── LOAD DATA ───
@st.cache_data
def load_data():
    return pd.read_csv("players_data-2024_2025.csv")

df = load_data()

# ─── SIMILARITY ENGINE ───
def get_top_similar_players_by_role(df, name, top_n=5):
    d = df.copy()
    m = d[d['Player'].str.contains(name, case=False, na=False)]
    if m.empty:
        raise ValueError(f"No player matching '{name}'.")
    idx    = m.index[0]
    target = d.at[idx,'Player']
    pos    = d.at[idx,'Pos']
    feats  = [f for f in position_feature_map.get(pos, []) if f in d.columns]
    if not feats:
        raise ValueError(f"No stats for position '{pos}'.")
    X     = d[feats].fillna(0)
    sims  = cosine_similarity([X.loc[idx]], X)[0]
    d['similarity'] = sims
    top = d[d.index != idx].nlargest(top_n, 'similarity')
    return target, pos, feats, top[['Player','Squad','Pos','Comp','similarity']]

# ─── MULTI-PLAYER RADAR ───
def build_multi_radar(position, players, df, feats):
    ranks = {f: df[f].dropna().rank(pct=True, method='max') for f in feats}
    theta = feats + [feats[0]]
    def pct_vals(pl):
        vals = [ranks[f].get(df.loc[df['Player']==pl, f].iat[0], 0) for f in feats]
        return vals + vals[:1]
    fig = go.Figure()
    colors = px.colors.qualitative.Dark24
    for i, pl in enumerate(players):
        fig.add_trace(go.Scatterpolar(
            r=pct_vals(pl), theta=theta, fill='toself', name=pl,
            line=dict(color=colors[i % len(colors)], width=2), opacity=0.5
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=False), angularaxis=dict(rotation=90, direction='clockwise')),
        showlegend=True,
        title=f"Multi-Player Radar ({position})",
        margin=dict(l=30, r=30, t=50, b=30)
    )
    return fig



# ─── CUSTOM SCATTER ───
def build_custom_scatter(df, highlight, x_col, y_col, size_col='Min', color_col='Comp'):
    d = df.dropna(subset=[x_col, y_col, size_col, color_col]).copy()
    d['size_norm'] = d[size_col] / d[size_col].max() * 200
    fig = px.scatter(
        d, x=x_col, y=y_col, size='size_norm', color=color_col,
        hover_name='Player', hover_data=['Squad','Pos',x_col,y_col],
        title=f"{x_col} vs {y_col}", opacity=0.7, size_max=20, template='plotly_white'
    )
    t = d[d['Player']==highlight]
    if not t.empty:
        fig.add_scatter(
            x=t[x_col], y=t[y_col], mode='markers+text',
            marker=dict(size=30, color='red', symbol='star'),
            text=[highlight], textposition='top center', showlegend=False
        )
    return fig

# ─── SIDEBAR FILTERS ───
player_name = st.sidebar.text_input("Player name")
top_n       = st.sidebar.slider("Top N matches", min_value=2, max_value=10, value=5, step=1)
leagues     = st.sidebar.multiselect("League", df['Comp'].unique(), default=df['Comp'].unique())
positions   = st.sidebar.multiselect("Position", df['Pos'].unique(), default=df['Pos'].unique())


# ─── VALIDATION ───
if not player_name:
    st.info("Enter a player name to get started.")
    st.stop()
if not leagues or not positions:
    st.warning("Please select at least one league AND one position.")
    st.stop()

# ─── COMPUTE SIMILARS & OVERVIEW HEADER ───
try:
    target, position, role_feats, similars = get_top_similar_players_by_role(df, player_name, top_n)
except Exception as e:
    st.error(e)
    st.stop()

# Overview Header: show name, position, team, nationality
idx = df.index[df['Player'] == target][0]
team = df.at[idx, 'Squad']
nation = df.at[idx, 'Nation']
mins90 = df.at[idx, '90s']

st.title(f"{target} — {position} @ {team} ({nation})")

# Compute and display top 3 role-specific stats with per90
ranks = {f: df[f].dropna().rank(pct=True, method='max') for f in role_feats}
pctiles = {f: ranks[f].get(df.loc[df['Player']==target, f].iat[0], 0) for f in role_feats}
top_stats = sorted(pctiles.items(), key=lambda x: x[1], reverse=True)[:3]
cols = st.columns(3, gap="small")
for i, (stat, _) in enumerate(top_stats):
    raw = df.loc[df['Player']==target, stat].iat[0]
    per90 = raw / mins90 if mins90 > 0 else 0
    cols[i].metric(stat, f"{raw:.2f} ({per90:.2f}/90)")

# ─── TABS ───
tabs = st.tabs(["Radar","Scatter","Data"])

with tabs[0]:
    st.subheader("Multi-Player Radar")
    pool = df[df['Pos'] == position]['Player'].unique().tolist()
    defaults = [target] + similars['Player'].tolist()[: max(0, top_n-1)]
    default_sel = [p for p in defaults if p in pool]
    if len(default_sel) < 2:
        default_sel = pool[:2] if len(pool) >= 2 else pool
    sel = st.multiselect("Select players to compare (2+)", options=pool, default=default_sel)
    if len(sel) < 2:
        st.warning("Pick at least two players for the radar.")
    else:
        fig = build_multi_radar(position, sel, df, role_feats)
        st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.subheader("Custom Scatter")
    df_sc = df[df['Comp'].isin(leagues) & df['Pos'].isin(positions)]
    x = st.selectbox("X axis", GLOBAL_FEATURES, index=0)
    y = st.selectbox("Y axis", GLOBAL_FEATURES, index=1)
    fig = build_custom_scatter(df_sc, target, x, y)
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.subheader("Top Matches Data")
    st.dataframe(similars, use_container_width=True)
    st.download_button("⬇️ Download CSV", similars.to_csv(index=False), file_name=f"{target}_matches.csv", mime="text/csv")
