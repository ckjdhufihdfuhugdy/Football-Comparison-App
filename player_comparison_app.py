import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity

# ─── GLOBAL FEATURES FOR SCATTER ───
GLOBAL_FEATURES = [
    'Gls','Ast','xG','xAG','KP','SCA90','GCA90','Sh/90','SoT/90',
    'Succ','Touches','xA','PrgP','PrgC','PrgR','Carries','1/3',
    'PPA','TotDist','TB','Sw','Tkl','Tkl+Int','Blocks','Clr','Err','Int',
    'GA90','PSxG+/-','Cmp%_stats_keeper_adv','AvgDist'
]

# ─── ROLE‐SPECIFIC MAPPINGS ───
scatter_axes_by_role = {
    "FW": ("xG", "xAG"),
    "MF": ("PrgP", "KP"),
    "DF": ("Tkl", "Int"),
    "GK": ("Save%", "PSxG+/-")
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
def get_top_similar_players_by_role(df, player_name, top_n=5):
    d = df.copy()
    matches = d[d['Player'].str.contains(player_name, case=False, na=False)]
    if matches.empty:
        raise ValueError(f"No player matching '{player_name}'.")
    idx       = matches.index[0]
    target    = d.at[idx, 'Player']
    position  = d.at[idx, 'Pos']
    if position not in position_feature_map:
        raise ValueError(f"Unsupported position '{position}'.")
    role_feats = [f for f in position_feature_map[position] if f in d.columns]
    if not role_feats:
        raise ValueError(f"No features for position '{position}'.")
    X    = d[role_feats].fillna(0)
    sims = cosine_similarity([X.loc[idx]], X)[0]
    d['similarity'] = sims
    top = d[d.index != idx].nlargest(top_n, 'similarity')
    return target, position, role_feats, top[['Player','Squad','Pos','Comp','similarity']]

# ─── PLOTLY PERCENTILE RADAR ───
def build_plotly_percentile_radar(p1, p2, df, feats):
    pct1, pct2 = [], []
    for f in feats:
        s = df[f].dropna()
        r = s.rank(pct=True, method='max')
        v1 = df.loc[df['Player']==p1, f].iat[0]
        v2 = df.loc[df['Player']==p2, f].iat[0]
        pct1.append(r.get(v1, 0))
        pct2.append(r.get(v2, 0))
    theta = feats + [feats[0]]
    r1 = pct1 + pct1[:1]
    r2 = pct2 + pct2[:1]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=r1, theta=theta, fill='toself', name=p1,
                                  line=dict(color='royalblue'), opacity=0.6))
    fig.add_trace(go.Scatterpolar(r=r2, theta=theta, fill='toself', name=p2,
                                  line=dict(color='darkorange'), opacity=0.6))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=False),
                   angularaxis=dict(rotation=90, direction='clockwise')),
        showlegend=True,
        title=f"{p1} vs {p2}"
    )
    return fig

# ─── CUSTOM PLOTLY SCATTER ───
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
        fig.add_scatter(x=t[x_col], y=t[y_col], mode='markers+text',
                        marker=dict(size=30, color='red', symbol='star'),
                        text=[highlight], textposition='top center', showlegend=False)
    return fig

# ─── STREAMLIT UI ───
st.sidebar.title("Player Comparison App")
player_name = st.sidebar.text_input("Enter a player name")
chart_type  = st.sidebar.radio("Choose chart type", ["Radar","Scatter"])
top_n       = st.sidebar.slider("How many similar players?", 3, 10, 5)

if player_name:
    try:
        target, position, role_feats, similars = get_top_similar_players_by_role(
            df, player_name, top_n
        )
        st.success(f"Top {top_n} matches for {target} ({position})")
        st.dataframe(similars)

        if chart_type == "Radar":
            st.subheader("Position-Specific Radar")
            for _, row in similars.iterrows():
                fig = build_plotly_percentile_radar(target, row['Player'], df, role_feats)
                st.plotly_chart(fig, use_container_width=True)

        else:  # Scatter
            # --- NEW: league & position filters ---
            leagues  = st.sidebar.multiselect("Filter by League", df['Comp'].unique().tolist(), default=df['Comp'].unique().tolist())
            positions= st.sidebar.multiselect("Filter by Position", df['Pos'].unique().tolist(), default=df['Pos'].unique().tolist())

            # apply filters
            df_scatter = df[(df['Comp'].isin(leagues)) & (df['Pos'].isin(positions))]

            x_axis = st.sidebar.selectbox("X axis", GLOBAL_FEATURES, index=0)
            y_axis = st.sidebar.selectbox("Y axis", GLOBAL_FEATURES, index=1)
            st.subheader(f"{x_axis} vs {y_axis} Scatter")
            fig = build_custom_scatter(df_scatter, target, x_axis, y_axis)
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Enter a player name to get started.")
