# app.py
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from regime_detector import MarketRegimeDetector
from datetime import datetime, timedelta
import time

# Auto-refresh functionality
def autorefresh(timeout_seconds=120):
    """Add auto-refresh meta tag to the page"""
    refresh_code = f"""
    <meta http-equiv="refresh" content="{timeout_seconds}">
    """
    st.markdown(refresh_code, unsafe_allow_html=True)

# Page config
st.set_page_config(
    page_title="Market Regime Shift Detector",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .regime-indicator {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .bull { background: linear-gradient(135deg, #00CC96, #00A86B); color: white; }
    .bear { background: linear-gradient(135deg, #FF4B4B, #CC0000); color: white; }
    .sideways { background: linear-gradient(135deg, #FFA500, #FF8C00); color: white; }
    .metric-card {
        background: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
        color: black;
    }
    .metric-card h2 {
        color: black;
    }
    .confidence-high { color: #00CC96; font-weight: bold; }
    .confidence-med { color: #FFA500; font-weight: bold; }
    .confidence-low { color: #FF4B4B; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

def create_regime_chart(df, changes):
    """Create interactive regime timeline"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=('SPY Price with Regimes', 'VIX (Fear Index)', 'Realized Volatility')
    )
    
    # Color mapping
    color_map = {'Bull': '#00CC96', 'Bear': '#FF4B4B', 'Sideways': '#FFA500'}
    
    # Add price line with regime background
    for regime in ['Bull', 'Bear', 'Sideways']:
        mask = df['Regime_Label'] == regime
        if mask.any():
            fig.add_trace(
                go.Scatter(
                    x=df.index[mask],
                    y=df['SPY_Close'][mask],
                    mode='lines',
                    name=f'{regime} Market',
                    line=dict(color=color_map[regime], width=2),
                    showlegend=True,
                    hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<br>Regime: ' + regime
                ),
                row=1, col=1
            )
    
    # Add regime change markers
    if not changes.empty:
        fig.add_trace(
            go.Scatter(
                x=changes.index,
                y=df.loc[changes.index, 'SPY_Close'],
                mode='markers',
                name='Regime Change',
                marker=dict(
                    symbol='diamond',
                    size=12,
                    color='white',
                    line=dict(color='black', width=2)
                ),
                hovertemplate='Regime Change<br>Date: %{x}<br>Price: $%{y:.2f}'
            ),
            row=1, col=1
        )
    
    # VIX
    fig.add_trace(
        go.Scatter(
            x=df.index, 
            y=df['VIX'],
            name='VIX',
            line=dict(color='purple', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(128,0,128,0.1)'
        ),
        row=2, col=1
    )
    
    # Realized Vol
    fig.add_trace(
        go.Scatter(
            x=df.index, 
            y=df['Realized_Vol'] * 100,
            name='Realized Vol %',
            line=dict(color='orange', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(255,165,0,0.1)'
        ),
        row=3, col=1
    )
    
    # Add horizontal reference lines
    fig.add_hline(y=20, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1, annotation_text="VIX Alert")
    fig.add_hline(y=30, line_dash="dash", line_color="darkred", opacity=0.5, row=2, col=1)
    
    fig.update_layout(
        height=800,
        template='plotly_white',
        hovermode='x unified',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="VIX", row=2, col=1)
    fig.update_yaxes(title_text="Vol %", row=3, col=1)
    
    return fig

def create_regime_distribution(df):
    """Create regime distribution pie chart"""
    regime_counts = df['Regime_Label'].value_counts()
    colors = ['#00CC96' if x == 'Bull' else '#FF4B4B' if x == 'Bear' else '#FFA500' 
              for x in regime_counts.index]
    
    fig = go.Figure(data=[go.Pie(
        labels=regime_counts.index,
        values=regime_counts.values,
        hole=0.4,
        marker_colors=colors,
        textinfo='label+percent',
        textfont_size=14,
        hovertemplate='Regime: %{label}<br>Days: %{value}<br>Percentage: %{percent}'
    )])
    
    fig.update_layout(
        title="Regime Distribution (Last 12 Months)",
        template='plotly_white',
        showlegend=False,
        height=400
    )
    
    return fig

def create_feature_importance(importance_df):
    """Create feature importance bar chart"""
    fig = px.bar(
        importance_df.head(8),
        x='importance',
        y='feature',
        orientation='h',
        color='importance',
        color_continuous_scale='Viridis',
        title='Most Important Regime Indicators'
    )
    
    fig.update_layout(
        template='plotly_white',
        height=400,
        yaxis_title="",
        xaxis_title="Importance Score",
        coloraxis_showscale=False
    )
    
    return fig

# ============ NEW VISUALIZATION FUNCTIONS ============

def create_price_sma_chart(df):
    """Price chart with SMA overlays"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['SPY_Close'], name='SPY Price', line=dict(color='black', width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='blue', width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='red', width=1)))
    fig.update_layout(title='SPY Price with Moving Averages', template='plotly_white', height=400)
    return fig

def create_volume_chart(df):
    """Volume bar chart"""
    colors = ['green' if df['Returns'].iloc[i] >= 0 else 'red' for i in range(len(df))]
    fig = go.Figure(data=[go.Bar(x=df.index, y=df['SPY_Volume'], marker_color=colors, name='Volume')])
    fig.add_trace(go.Scatter(x=df.index, y=df['Volume_MA20'], name='Volume MA20', line=dict(color='orange', width=2)))
    fig.update_layout(title='Trading Volume', template='plotly_white', height=350, yaxis_title='Volume')
    return fig

def create_bollinger_bands(df):
    """Bollinger Bands chart"""
    df = df.copy()
    df['BB_Middle'] = df['SPY_Close'].rolling(20).mean()
    df['BB_Std'] = df['SPY_Close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['SPY_Close'], name='SPY', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='Upper Band', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Middle'], name='Middle Band', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='Lower Band', line=dict(color='green', dash='dash')))
    fig.update_layout(title='Bollinger Bands', template='plotly_white', height=400)
    return fig

def create_rsi_chart(df):
    """RSI indicator chart"""
    delta = df['SPY_Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=rsi, name='RSI(14)', line=dict(color='purple', width=2)))
    fig.add_hline(y=70, line_dash='dash', line_color='red', annotation_text='Overbought')
    fig.add_hline(y=30, line_dash='dash', line_color='green', annotation_text='Oversold')
    fig.update_layout(title='Relative Strength Index (RSI)', template='plotly_white', height=350, yaxis_range=[0, 100])
    return fig

def create_macd_chart(df):
    """MACD indicator chart"""
    exp1 = df['SPY_Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['SPY_Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    colors = ['green' if val >= 0 else 'red' for val in histogram]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=histogram, marker_color=colors, name='Histogram'))
    fig.add_trace(go.Scatter(x=df.index, y=macd, name='MACD', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=signal, name='Signal', line=dict(color='orange')))
    fig.update_layout(title='MACD', template='plotly_white', height=350)
    return fig

def create_momentum_chart(df):
    """Multiple period momentum chart"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Momentum_10']*100, name='10-Day Momentum', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Momentum_30']*100, name='30-Day Momentum', line=dict(color='red')))
    fig.add_hline(y=0, line_dash='dash', line_color='black')
    fig.update_layout(title='Price Momentum (%)', template='plotly_white', height=350)
    return fig

def create_roc_chart(df):
    """Rate of Change chart"""
    df = df.copy()
    df['ROC_10'] = ((df['SPY_Close'] - df['SPY_Close'].shift(10)) / df['SPY_Close'].shift(10)) * 100
    df['ROC_20'] = ((df['SPY_Close'] - df['SPY_Close'].shift(20)) / df['SPY_Close'].shift(20)) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['ROC_10'], name='ROC 10', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['ROC_20'], name='ROC 20', line=dict(color='red')))
    fig.add_hline(y=0, line_dash='dash', line_color='black')
    fig.update_layout(title='Rate of Change (%)', template='plotly_white', height=350)
    return fig

def create_volatility_comparison(df):
    """VIX vs Realized Vol comparison"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['VIX'], name='VIX', fill='tozeroy', fillcolor='rgba(128,0,128,0.2)', line=dict(color='purple')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Realized_Vol']*100, name='Realized Vol %', line=dict(color='orange')))
    fig.update_layout(title='VIX vs Realized Volatility', template='plotly_white', height=400)
    return fig

def create_volatility_clustering(df):
    """Volatility clustering chart"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Returns'].abs()*100, name='|Returns|', marker=dict(color='blue', size=3)))
    fig.add_trace(go.Scatter(x=df.index, y=df['Realized_Vol']*100, name='Realized Vol', line=dict(color='red', width=2)))
    fig.update_layout(title='Volatility Clustering', template='plotly_white', height=350)
    return fig

def create_vol_regime_scatter(df):
    """Volatility by regime scatter"""
    df_plot = df.copy()
    df_plot['Realized_Vol_Pct'] = df_plot['Realized_Vol'] * 100
    fig = px.scatter(df_plot, x='VIX', y='Realized_Vol_Pct', color='Regime_Label',
                     color_discrete_map={'Bull': '#00CC96', 'Bear': '#FF4B4B', 'Sideways': '#FFA500'},
                     title='VIX vs Realized Vol by Regime', template='plotly_white', height=400)
    return fig

def create_yield_chart(df):
    """10-Year Treasury Yield chart"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Yield_10Y'], name='10Y Yield', line=dict(color='green'), fill='tozeroy', fillcolor='rgba(0,200,0,0.1)'))
    fig.update_layout(title='10-Year Treasury Yield (%)', template='plotly_white', height=350)
    return fig

def create_gold_chart(df):
    """Gold price chart"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Gold'], name='Gold (GLD)', line=dict(color='gold', width=2), fill='tozeroy', fillcolor='rgba(255,215,0,0.1)'))
    fig.update_layout(title='Gold Price ($)', template='plotly_white', height=350)
    return fig

def create_commodities_chart(df):
    """Commodities index chart"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Commodities'], name='Commodities (DBC)', line=dict(color='brown', width=2), fill='tozeroy', fillcolor='rgba(139,69,19,0.1)'))
    fig.update_layout(title='Commodities Index ($)', template='plotly_white', height=350)
    return fig

def create_usd_chart(df):
    """USD Index chart"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['USD'], name='USD Index (UUP)', line=dict(color='gray', width=2), fill='tozeroy', fillcolor='rgba(128,128,128,0.1)'))
    fig.update_layout(title='US Dollar Index ($)', template='plotly_white', height=350)
    return fig

def create_correlation_matrix(df):
    """Correlation matrix heatmap"""
    cols = ['SPY_Close', 'VIX', 'Yield_10Y', 'Commodities', 'Gold', 'USD', 'Returns', 'Realized_Vol']
    corr = df[cols].corr()
    fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r', title='Correlation Matrix', template='plotly_white', height=450)
    return fig

def create_rolling_correlation(df):
    """Rolling correlation with macro assets"""
    df = df.copy()
    df['Corr_VIX'] = df['SPY_Close'].rolling(30).corr(df['VIX'])
    df['Corr_Yield'] = df['SPY_Close'].rolling(30).corr(df['Yield_10Y'])
    df['Corr_Gold'] = df['SPY_Close'].rolling(30).corr(df['Gold'])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Corr_VIX'], name='SPY vs VIX', line=dict(color='purple')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Corr_Yield'], name='SPY vs Yield', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Corr_Gold'], name='SPY vs Gold', line=dict(color='gold')))
    fig.add_hline(y=0, line_dash='dash', line_color='black')
    fig.update_layout(title='Rolling 30-Day Correlations', template='plotly_white', height=350)
    return fig

def create_inflation_ratio(df):
    """Commodities vs Gold ratio (inflation proxy)"""
    df = df.copy()
    df['Inf_Ratio'] = df['Commodities'] / df['Gold']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Inf_Ratio'], name='DBC/GLD Ratio', line=dict(color='darkred'), fill='tozeroy', fillcolor='rgba(139,0,0,0.1)'))
    fig.update_layout(title='Inflation Proxy (Commodities/Gold Ratio)', template='plotly_white', height=350)
    return fig

def create_regime_transitions(df):
    """Regime transition analysis"""
    df = df.copy()
    transitions = df['Regime_Label'].shift(1) != df['Regime_Label']
    regime_changes = df[transitions].index
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['SPY_Close'], name='SPY', line=dict(color='gray', width=1)))
    
    # Add shapes instead of vline for better compatibility
    for date in regime_changes:
        regime = df.loc[date, 'Regime_Label']
        color = '#00CC96' if regime == 'Bull' else '#FF4B4B' if regime == 'Bear' else '#FFA500'
        # Convert to float timestamp for compatibility
        fig.add_vline(x=float(date.timestamp()), line_color=color, line_dash='dash', annotation_text=regime)
    fig.update_layout(title='Regime Transitions', template='plotly_white', height=400)
    return fig

def create_regime_duration_histogram(df):
    """Regime duration histogram"""
    regimes = df['Regime_Label'].values
    durations = []
    current_regime = regimes[0]
    count = 1
    for i in range(1, len(regimes)):
        if regimes[i] == current_regime:
            count += 1
        else:
            if count > 1:
                durations.append({'regime': current_regime, 'duration': count})
            current_regime = regimes[i]
            count = 1
    if count > 1:
        durations.append({'regime': current_regime, 'duration': count})
    durations_df = pd.DataFrame(durations)
    colors = {'Bull': '#00CC96', 'Bear': '#FF4B4B', 'Sideways': '#FFA500'}
    fig = px.histogram(durations_df, x='duration', color='regime', barmode='overlay',
                       color_discrete_map=colors, title='Regime Duration Distribution', template='plotly_white', height=400)
    return fig

def create_regime_returns_box(df):
    """Returns distribution by regime"""
    df = df.copy()
    df['Returns_Pct'] = df['Returns'] * 100
    colors = {'Bull': '#00CC96', 'Bear': '#FF4B4B', 'Sideways': '#FFA500'}
    fig = px.box(df, x='Regime_Label', y='Returns_Pct', color='Regime_Label', color_discrete_map=colors,
                 title='Daily Returns by Regime', template='plotly_white', height=400)
    return fig

def create_regime_stability(df):
    """Regime stability over time"""
    df = df.copy()
    # Convert regime to numeric for rolling calculation
    regime_map = {'Bull': 2, 'Sideways': 1, 'Bear': 0}
    df['Regime_Num'] = df['Regime_Label'].map(regime_map)
    window = 5
    
    # Calculate stability as the proportion of the most common regime in the window
    def calc_stability(x):
        if len(x) == 0:
            return np.nan
        counts = x.value_counts()
        return counts.iloc[0] / len(x)
    
    df['Regime_Stability'] = df['Regime_Num'].rolling(window).apply(calc_stability, raw=False)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Regime_Stability'], name='Stability', line=dict(color='blue'), fill='tozeroy', fillcolor='rgba(0,0,255,0.2)'))
    fig.update_layout(title=f'Regime Stability ({window}-day rolling)', template='plotly_white', height=350, yaxis_range=[0, 1.1])
    return fig

def create_rolling_sharpe(df, risk_free_rate=0.02):
    """Rolling Sharpe Ratio"""
    df = df.copy()
    df['Rolling_Sharpe'] = (df['Returns'].rolling(60).mean() * 252 - risk_free_rate) / (df['Returns'].rolling(60).std() * np.sqrt(252))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Rolling_Sharpe'], name='Rolling Sharpe', line=dict(color='darkblue')))
    fig.add_hline(y=1, line_dash='dash', line_color='green', annotation_text='Sharpe=1')
    fig.add_hline(y=0, line_dash='dash', line_color='black')
    fig.add_hline(y=-1, line_dash='dash', line_color='red', annotation_text='Sharpe=-1')
    fig.update_layout(title='Rolling 60-Day Sharpe Ratio', template='plotly_white', height=350)
    return fig

def create_drawdown_chart(df):
    """Drawdown chart"""
    df = df.copy()
    df['Cummax'] = df['SPY_Close'].cummax()
    df['Drawdown'] = (df['SPY_Close'] - df['Cummax']) / df['Cummax'] * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Drawdown'], name='Drawdown %', fill='tozeroy', fillcolor='rgba(255,0,0,0.3)', line=dict(color='red')))
    fig.update_layout(title='Drawdown from Peak (%)', template='plotly_white', height=400, yaxis_title='Drawdown %')
    return fig

def create_regime_performance(df):
    """Cumulative returns by regime"""
    df = df.copy()
    df['Cum_Returns'] = (1 + df['Returns']).cumprod()
    colors = {'Bull': '#00CC96', 'Bear': '#FF4B4B', 'Sideways': '#FFA500'}
    fig = px.line(df, y='Cum_Returns', color='Regime_Label', color_discrete_map=colors,
                  title='Cumulative Returns by Current Regime', template='plotly_white', height=400)
    return fig

def create_vix_spike_analysis(df):
    """VIX spike analysis"""
    df = df.copy()
    df['VIX_Spike_Pct'] = df['VIX_Spike'] * 100
    colors = ['red' if x > 20 else 'orange' if x > 0 else 'green' for x in df['VIX_Spike_Pct']]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df['VIX_Spike_Pct'], marker_color=colors, name='VIX Spike %'))
    fig.add_hline(y=0, line_color='black')
    fig.update_layout(title='VIX Spike Analysis (%)', template='plotly_white', height=350)
    return fig

def create_trend_strength_chart(df):
    """Trend strength indicator"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Trend_Strength']*100, name='Trend Strength %', line=dict(color='blue'), fill='tozeroy', fillcolor='rgba(0,0,255,0.1)'))
    fig.add_hline(y=0, line_dash='dash', line_color='black')
    fig.add_hline(y=5, line_dash='dash', line_color='green', annotation_text='Strong Up')
    fig.add_hline(y=-5, line_dash='dash', line_color='red', annotation_text='Strong Down')
    fig.update_layout(title='Trend Strength (% from SMA20)', template='plotly_white', height=350)
    return fig

def create_volume_ratio_chart(df):
    """Volume ratio analysis"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Volume_Ratio'], name='Volume Ratio', line=dict(color='orange'), fill='tozeroy', fillcolor='rgba(255,165,0,0.2)'))
    fig.add_hline(y=1, line_dash='dash', line_color='black', annotation_text='Normal')
    fig.add_hline(y=1.5, line_dash='dash', line_color='red', annotation_text='High Volume')
    fig.update_layout(title='Volume Ratio (vs 20-day MA)', template='plotly_white', height=350)
    return fig

def create_macro_momentum(df):
    """Macro asset momentum comparison"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Gold_Momentum']*100, name='Gold Momentum', line=dict(color='gold')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Commodity_Momentum']*100, name='Commodity Momentum', line=dict(color='brown')))
    fig.add_trace(go.Scatter(x=df.index, y=df['USD_Momentum']*100, name='USD Momentum', line=dict(color='gray')))
    fig.add_hline(y=0, line_dash='dash', line_color='black')
    fig.update_layout(title='Macro Asset Momentum (10-day %)', template='plotly_white', height=350)
    return fig

def create_returns_histogram(df):
    """Returns distribution histogram"""
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df['Returns']*100, nbinsx=50, name='Daily Returns', marker_color='blue', opacity=0.7))
    fig.add_vline(x=df['Returns'].mean()*100, line_color='red', annotation_text=f'Mean: {df["Returns"].mean()*100:.2f}%')
    fig.update_layout(title='Daily Returns Distribution (%)', template='plotly_white', height=400)
    return fig

def create_vol_surface(df):
    """Volatility surface proxy"""
    df = df.copy()
    df['Vol_Regime'] = pd.cut(df['VIX'], bins=[0, 15, 25, 35, 100], labels=['Low', 'Normal', 'Elevated', 'High'])
    vol_by_regime = df.groupby('Vol_Regime')['Realized_Vol'].mean() * 100
    fig = go.Figure(data=[go.Bar(x=vol_by_regime.index, y=vol_by_regime.values, marker_color=['green', 'yellow', 'orange', 'red'], name='Avg Realized Vol')])
    fig.update_layout(title='Realized Vol by VIX Regime', template='plotly_white', height=400, yaxis_title='Realized Vol %')
    return fig

def create_regime_heatmap(df):
    """Regime calendar heatmap"""
    df = df.copy()
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    heatmap_data = df.groupby(['Year', 'Month'])['Regime_Label'].first().unstack()
    regime_map = {'Bull': 2, 'Sideways': 1, 'Bear': 0}
    heatmap_numeric = heatmap_data.replace(regime_map)
    fig = go.Figure(data=go.Heatmap(z=heatmap_numeric.values, x=heatmap_numeric.columns, y=heatmap_numeric.index,
                                     colorscale=[[0, '#FF4B4B'], [0.5, '#FFA500'], [1, '#00CC96']], showscale=False))
    fig.update_layout(title='Regime Calendar Heatmap', template='plotly_white', height=400)
    return fig

def create_candlestick_chart(df):
    """Create candlestick chart with technical indicators"""
    df = df.copy()
    
    # Calculate additional indicators
    df['SMA_20'] = df['SPY_Close'].rolling(20).mean()
    df['SMA_50'] = df['SPY_Close'].rolling(50).mean()
    df['BB_Middle'] = df['SPY_Close'].rolling(20).mean()
    df['BB_Std'] = df['SPY_Close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=('SPY Candlestick Chart', 'Volume', 'Bollinger Bands Width')
    )
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['SPY_Open'],
        high=df['SPY_High'],
        low=df['SPY_Low'],
        close=df['SPY_Close'],
        name='SPY',
        increasing_line_color='green',
        decreasing_line_color='red'
    ), row=1, col=1)
    
    # SMAs
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='blue', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='red', width=1)), row=1, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', line=dict(color='gray', dash='dash'), opacity=0.5), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', line=dict(color='gray', dash='dash'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)
    
    # Volume
    colors = ['green' if df['SPY_Close'].iloc[i] >= df['SPY_Open'].iloc[i] else 'red' for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df['SPY_Volume'], marker_color=colors, name='Volume'), row=2, col=1)
    
    # Bollinger Band Width (volatility indicator)
    bb_width = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle'] * 100
    fig.add_trace(go.Scatter(x=df.index, y=bb_width, name='BB Width %', line=dict(color='purple'), fill='tozeroy', fillcolor='rgba(128,0,128,0.2)'), row=3, col=1)
    
    fig.update_layout(
        height=900,
        template='plotly_white',
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    return fig

def create_advanced_indicators(df):
    """Create advanced technical indicators chart"""
    df = df.copy()
    
    # ATR (Average True Range)
    high_low = df['SPY_High'] - df['SPY_Low']
    high_close = abs(df['SPY_High'] - df['SPY_Close'].shift())
    low_close = abs(df['SPY_Low'] - df['SPY_Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    
    # Stochastic Oscillator
    low_14 = df['SPY_Low'].rolling(14).min()
    high_14 = df['SPY_High'].rolling(14).max()
    df['Stoch_K'] = 100 * ((df['SPY_Close'] - low_14) / (high_14 - low_14))
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
    
    # MACD
    exp12 = df['SPY_Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['SPY_Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # CCI (Commodity Channel Index)
    tp = (df['SPY_High'] + df['SPY_Low'] + df['SPY_Close']) / 3
    sma_tp = tp.rolling(20).mean()
    mad = tp.rolling(20).apply(lambda x: abs(x - x.mean()).mean())
    df['CCI'] = (tp - sma_tp) / (0.015 * mad)
    
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.3, 0.25, 0.25, 0.2],
        subplot_titles=('Price with ATR', 'Stochastic Oscillator', 'MACD', 'CCI')
    )
    
    # Price with ATR
    fig.add_trace(go.Scatter(x=df.index, y=df['SPY_Close'], name='SPY', line=dict(color='black')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['ATR'], name='ATR', line=dict(color='blue'), fill='tozeroy', fillcolor='rgba(0,0,255,0.1)'), row=1, col=1)
    
    # Stochastic
    fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_K'], name='%K', line=dict(color='blue')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_D'], name='%D', line=dict(color='orange')), row=2, col=1)
    fig.add_hline(y=80, line_dash='dash', line_color='red', row=2, col=1)
    fig.add_hline(y=20, line_dash='dash', line_color='green', row=2, col=1)
    
    # MACD
    colors = ['green' if x >= 0 else 'red' for x in df['MACD_Hist']]
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], marker_color=colors, name='MACD Hist'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', line=dict(color='orange')), row=3, col=1)
    
    # CCI
    fig.add_trace(go.Scatter(x=df.index, y=df['CCI'], name='CCI', line=dict(color='purple'), fill='tozeroy', fillcolor='rgba(128,0,128,0.1)'), row=4, col=1)
    fig.add_hline(y=100, line_dash='dash', line_color='red', row=4, col=1)
    fig.add_hline(y=-100, line_dash='dash', line_color='green', row=4, col=1)
    
    fig.update_layout(
        height=1000,
        template='plotly_white',
        showlegend=True
    )
    
    return fig

def create_market_momentum(df):
    """Create market momentum dashboard"""
    df = df.copy()
    
    # On Balance Volume (OBV)
    df['OBV'] = (np.sign(df['SPY_Close'].diff()) * df['SPY_Volume']).fillna(0).cumsum()
    
    # Money Flow Index (MFI)
    typical_price = (df['SPY_High'] + df['SPY_Low'] + df['SPY_Close']) / 3
    money_flow = typical_price * df['SPY_Volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(), 0).rolling(14).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(), 0).rolling(14).sum()
    mfi = 100 - (100 / (1 + positive_flow / negative_flow))
    df['MFI'] = mfi
    
    # Rate of Change
    df['ROC_5'] = ((df['SPY_Close'] - df['SPY_Close'].shift(5)) / df['SPY_Close'].shift(5)) * 100
    df['ROC_10'] = ((df['SPY_Close'] - df['SPY_Close'].shift(10)) / df['SPY_Close'].shift(10)) * 100
    df['ROC_20'] = ((df['SPY_Close'] - df['SPY_Close'].shift(20)) / df['SPY_Close'].shift(20)) * 100
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.4, 0.3, 0.3],
        subplot_titles=('On Balance Volume (OBV)', 'Money Flow Index (MFI)', 'Rate of Change (ROC)')
    )
    
    # OBV
    fig.add_trace(go.Scatter(x=df.index, y=df['OBV'], name='OBV', line=dict(color='blue'), fill='tozeroy', fillcolor='rgba(0,0,255,0.1)'), row=1, col=1)
    
    # MFI
    fig.add_trace(go.Scatter(x=df.index, y=df['MFI'], name='MFI', line=dict(color='green'), fill='tozeroy', fillcolor='rgba(0,200,0,0.1)'), row=2, col=1)
    fig.add_hline(y=80, line_dash='dash', line_color='red', annotation_text='Overbought', row=2, col=1)
    fig.add_hline(y=20, line_dash='dash', line_color='green', annotation_text='Oversold', row=2, col=1)
    
    # ROC
    fig.add_trace(go.Scatter(x=df.index, y=df['ROC_5'], name='ROC 5', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['ROC_10'], name='ROC 10', line=dict(color='orange')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['ROC_20'], name='ROC 20', line=dict(color='red')), row=3, col=1)
    fig.add_hline(y=0, line_dash='dash', line_color='black', row=3, col=1)
    
    fig.update_layout(
        height=800,
        template='plotly_white',
        showlegend=True
    )
    
    return fig

def create_support_resistance(df):
    """Create support and resistance levels"""
    df = df.copy()
    
    # Calculate pivot points
    df['Pivot'] = (df['SPY_High'].shift(1) + df['SPY_Low'].shift(1) + df['SPY_Close'].shift(1)) / 3
    df['R1'] = 2 * df['Pivot'] - df['SPY_Low'].shift(1)
    df['S1'] = 2 * df['Pivot'] - df['SPY_High'].shift(1)
    df['R2'] = df['Pivot'] + (df['SPY_High'].shift(1) - df['SPY_Low'].shift(1))
    df['S2'] = df['Pivot'] - (df['SPY_High'].shift(1) - df['SPY_Low'].shift(1))
    
    fig = go.Figure()
    
    # Price
    fig.add_trace(go.Scatter(x=df.index, y=df['SPY_Close'], name='SPY', line=dict(color='black', width=2)))
    
    # Pivot points
    fig.add_trace(go.Scatter(x=df.index, y=df['Pivot'], name='Pivot', line=dict(color='blue', dash='dot')))
    fig.add_trace(go.Scatter(x=df.index, y=df['R1'], name='R1 (Resistance 1)', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=df.index, y=df['S1'], name='S1 (Support 1)', line=dict(color='green', dash='dash')))
    fig.add_trace(go.Scatter(x=df.index, y=df['R2'], name='R2 (Resistance 2)', line=dict(color='darkred', dash='dash')))
    fig.add_trace(go.Scatter(x=df.index, y=df['S2'], name='S2 (Support 2)', line=dict(color='darkgreen', dash='dash')))
    
    fig.update_layout(
        title='Support & Resistance Levels (Daily Pivot Points)',
        template='plotly_white',
        height=500,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    return fig

def create_relative_strength(df):
    """Create relative strength comparison charts"""
    # Fetch additional tickers for comparison
    tickers = ['SPY', 'QQQ', 'IWM', 'DIA', 'GLD', 'TLT']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=252)
    
    import yfinance as yf
    prices = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
    
    # Normalize to percentage returns
    normalized = (prices / prices.iloc[0]) * 100
    
    fig = go.Figure()
    for ticker in tickers:
        fig.add_trace(go.Scatter(
            x=normalized.index, 
            y=normalized[ticker], 
            name=ticker,
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title='Relative Strength Comparison (Normalized to 100)',
        template='plotly_white',
        height=500,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    return fig


def main():
    st.markdown('<h1 class="main-header">🎯 Market Regime Shift Detector</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: #666; margin-bottom: 2rem;'>
        AI-Powered Macro Regime Detection using Hidden Markov Models & Multi-Factor Analysis
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        lookback = st.slider("Lookback Period (days)", 180, 500, 252)
        n_regimes = st.selectbox("Number of Regimes", [3, 4], index=0)
        
        st.markdown("---")
        st.header("🔄 Auto Refresh")
        auto_refresh = st.toggle("Enable Auto-Refresh", value=True)
        refresh_interval = st.slider("Refresh Interval (seconds)", 30, 300, 120)
        
        if auto_refresh:
            # Use built-in autorefresh function
            autorefresh(timeout_seconds=refresh_interval)
            st.success(f"✅ Auto-refresh: Data refreshes every {refresh_interval}s | Next refresh in ~{refresh_interval}s")
            
            # Show countdown using JavaScript
            countdown_js = f"""
            <div id='countdown' style='text-align: center; font-size: 18px; font-weight: bold; color: #1f77b4; margin: 10px 0;'>
                ⏱️ Next data refresh in: <span id='timer'>{refresh_interval}</span> seconds
            </div>
            <script>
            var seconds = {refresh_interval};
            var timerElement = document.getElementById('timer');
            var countdownInterval = setInterval(function() {{
                seconds--;
                if (timerElement) {{
                    timerElement.innerText = seconds;
                }}
                if (seconds <= 0) {{
                    clearInterval(countdownInterval);
                }}
            }}, 1000);
            </script>
            """
            st.markdown(countdown_js, unsafe_allow_html=True)
        
        st.markdown("---")
        st.header("📊 Data Sources")
        st.markdown("""
        - **SPY**: S&P 500 ETF
        - **^VIX**: Volatility Index
        - **^TNX**: 10-Year Treasury Yield
        - **DBC**: Commodities Index
        - **GLD**: Gold ETF
        - **UUP**: US Dollar Index
        """)
        
        st.markdown("---")
        st.header("🤖 Model Info")
        st.markdown("""
        **Algorithm**: Gaussian Hidden Markov Model  
        **Features**: 12 macro & technical indicators  
        **Update**: Real-time via Yahoo Finance
        """)
        
        run_analysis = st.button("🚀 Run Detection", type="primary", use_container_width=True)
        
        # Manual refresh button
        if st.button("🔄 Refresh Now", use_container_width=True):
            st.rerun()
    
    # Always fetch fresh data when auto-refresh is enabled or button clicked
    # This ensures real-time data updates
    if run_analysis or auto_refresh or 'results' not in st.session_state:
        with st.spinner("Fetching market data and running AI models..."):
            try:
                detector = MarketRegimeDetector(lookback_days=lookback)
                results = detector.run_detection()
                st.session_state['results'] = results
                st.session_state['detector'] = detector
            except Exception as e:
                st.error(f"Error running analysis: {str(e)}")
                return
    
    if 'results' in st.session_state:
        results = st.session_state['results']
        current = results['current']
        df = results['data']
        
        # Current Regime Display
        st.markdown("---")
        cols = st.columns([2, 1, 1])
        
        with cols[0]:
            regime_class = current['regime'].lower()
            detector = st.session_state['detector']
            emoji = detector.regime_emoji[current['regime']]
            
            st.markdown(f"""
            <div class="regime-indicator {regime_class}">
                {emoji} CURRENT REGIME: {current['regime'].upper()} MARKET {emoji}<br>
                <span style='font-size: 1.2rem; opacity: 0.9;'>
                    Confidence: {current['confidence']:.1%}
                </span>
            </div>
            """, unsafe_allow_html=True)
        
        with cols[1]:
            conf_class = 'confidence-high' if current['confidence'] > 0.7 else 'confidence-med' if current['confidence'] > 0.5 else 'confidence-low'
            st.markdown(f"""
            <div class="metric-card">
                <h4>🎯 Model Confidence</h4>
                <h2 class='{conf_class}'>{current['confidence']:.1%}</h2>
                <small>Based on HMM likelihood</small>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>⏱️ Regime Duration</h4>
                <h2>{current['duration_days']} days</h2>
                <small>Since {current['regime_start'].strftime('%Y-%m-%d')}</small>
            </div>
            """, unsafe_allow_html=True)
        
        with cols[2]:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📈 SPY Price</h4>
                <h2>${current['spy_price']:.2f}</h2>
                <small>Real-time</small>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>⚡ VIX Level</h4>
                <h2>{current['vix_level']:.2f}</h2>
                <small>Market Fear Index</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Charts
        st.markdown("---")
        st.subheader("📊 Regime Timeline & Market Indicators")
        
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
            "📈 Price Tech", "🕯️ Candlestick", "📊 Volatility", "💰 Macro", 
            "🔄 Regime Analysis", "⚠️ Risk Metrics", "📉 Returns", "🚀 Advanced", "🎯 Feature Analysis"
        ])
        
        # Tab 1: Price & Technical Analysis
        with tab1:
            fig = create_regime_chart(df, results['changes'])
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                fig_sma = create_price_sma_chart(df)
                st.plotly_chart(fig_sma, use_container_width=True)
            with col2:
                fig_bb = create_bollinger_bands(df)
                st.plotly_chart(fig_bb, use_container_width=True)
            
            col3, col4 = st.columns(2)
            with col3:
                fig_rsi = create_rsi_chart(df)
                st.plotly_chart(fig_rsi, use_container_width=True)
            with col4:
                fig_macd = create_macd_chart(df)
                st.plotly_chart(fig_macd, use_container_width=True)
            
            col5, col6 = st.columns(2)
            with col5:
                fig_mom = create_momentum_chart(df)
                st.plotly_chart(fig_mom, use_container_width=True)
            with col6:
                fig_roc = create_roc_chart(df)
                st.plotly_chart(fig_roc, use_container_width=True)
            
            fig_vol = create_volume_chart(df)
            st.plotly_chart(fig_vol, use_container_width=True)
            
            fig_trend = create_trend_strength_chart(df)
            st.plotly_chart(fig_trend, use_container_width=True)
            
            fig_vol_ratio = create_volume_ratio_chart(df)
            st.plotly_chart(fig_vol_ratio, use_container_width=True)
            
            # Recent changes table
            if not results['changes'].empty:
                st.subheader("🔄 Recent Regime Changes")
                changes_df = results['changes'].tail(5)[['Regime_Label', 'SPY_Close', 'VIX', 'Realized_Vol']].copy()
                changes_df['SPY_Close'] = changes_df['SPY_Close'].round(2)
                changes_df['VIX'] = changes_df['VIX'].round(2)
                changes_df['Realized_Vol'] = (changes_df['Realized_Vol'] * 100).round(2)
                changes_df.columns = ['New Regime', 'SPY Price', 'VIX', 'Realized Vol (%)']
                st.table(changes_df)
        
        # Tab 2: Candlestick Chart
        with tab2:
            st.subheader("🕯️ Candlestick Chart with Technical Indicators")
            fig_candle = create_candlestick_chart(df)
            st.plotly_chart(fig_candle, use_container_width=True)
        
        # Tab 3: Volatility (moved from tab2)
        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                fig_dist = create_regime_distribution(df)
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with col2:
                # Regime statistics
                stats = df.groupby('Regime_Label').agg({
                    'Returns': ['mean', 'std'],
                    'Realized_Vol': 'mean',
                    'VIX': 'mean'
                }).round(4)
                
                stats.columns = ['Avg Return', 'Return Vol', 'Avg Realized Vol', 'Avg VIX']
                stats['Avg Return'] = stats['Avg Return'] * 100  # Convert to %
                stats['Annualized Return'] = (stats['Avg Return'] * 252).round(2)
                
                st.subheader("Regime Characteristics")
                st.dataframe(stats.style.background_gradient(cmap='RdYlGn', subset=['Annualized Return']), use_container_width=True)
        
        # Tab 4: Volatility Analysis
        with tab4:
            col1, col2 = st.columns(2)
            with col1:
                fig_vol = create_volatility_comparison(df)
                st.plotly_chart(fig_vol, use_container_width=True)
            with col2:
                fig_vol_cls = create_volatility_clustering(df)
                st.plotly_chart(fig_vol_cls, use_container_width=True)
            
            col3, col4 = st.columns(2)
            with col3:
                fig_vol_surf = create_vol_surface(df)
                st.plotly_chart(fig_vol_surf, use_container_width=True)
            with col4:
                fig_vol_reg = create_vol_regime_scatter(df)
                st.plotly_chart(fig_vol_reg, use_container_width=True)
            
            fig_vix_spike = create_vix_spike_analysis(df)
            st.plotly_chart(fig_vix_spike, use_container_width=True)
        
        # Tab 5: Macro Analysis
        with tab5:
            col1, col2 = st.columns(2)
            with col1:
                fig_yield = create_yield_chart(df)
                st.plotly_chart(fig_yield, use_container_width=True)
            with col2:
                fig_gold = create_gold_chart(df)
                st.plotly_chart(fig_gold, use_container_width=True)
            
            col3, col4 = st.columns(2)
            with col3:
                fig_comm = create_commodities_chart(df)
                st.plotly_chart(fig_comm, use_container_width=True)
            with col4:
                fig_usd = create_usd_chart(df)
                st.plotly_chart(fig_usd, use_container_width=True)
            
            fig_corr = create_correlation_matrix(df)
            st.plotly_chart(fig_corr, use_container_width=True)
            
            fig_roll_corr = create_rolling_correlation(df)
            st.plotly_chart(fig_roll_corr, use_container_width=True)
            
            fig_inf = create_inflation_ratio(df)
            st.plotly_chart(fig_inf, use_container_width=True)
            
            fig_macro_mom = create_macro_momentum(df)
            st.plotly_chart(fig_macro_mom, use_container_width=True)
        
        # Tab 6: Regime Analysis
        with tab6:
            fig_trans = create_regime_transitions(df)
            st.plotly_chart(fig_trans, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                fig_dur = create_regime_duration_histogram(df)
                st.plotly_chart(fig_dur, use_container_width=True)
            with col2:
                fig_ret_box = create_regime_returns_box(df)
                st.plotly_chart(fig_ret_box, use_container_width=True)
            
            fig_stab = create_regime_stability(df)
            st.plotly_chart(fig_stab, use_container_width=True)
            
            fig_perf = create_regime_performance(df)
            st.plotly_chart(fig_perf, use_container_width=True)
            
            fig_heat = create_regime_heatmap(df)
            st.plotly_chart(fig_heat, use_container_width=True)
        
        # Tab 7: Risk & Returns
        with tab7:
            fig_sharpe = create_rolling_sharpe(df)
            st.plotly_chart(fig_sharpe, use_container_width=True)
            
            fig_dd = create_drawdown_chart(df)
            st.plotly_chart(fig_dd, use_container_width=True)
            
            fig_ret_hist = create_returns_histogram(df)
            st.plotly_chart(fig_ret_hist, use_container_width=True)
        
        # Tab 8: Advanced Technical Analysis
        with tab8:
            st.subheader("🚀 Advanced Technical Indicators")
            
            fig_adv = create_advanced_indicators(df)
            st.plotly_chart(fig_adv, use_container_width=True)
            
            fig_mom = create_market_momentum(df)
            st.plotly_chart(fig_mom, use_container_width=True)
            
            fig_sr = create_support_resistance(df)
            st.plotly_chart(fig_sr, use_container_width=True)
            
            fig_rs = create_relative_strength(df)
            st.plotly_chart(fig_rs, use_container_width=True)
        
        # Tab 9: Feature Analysis (original tab3 content)
        with tab9:
            col1, col2 = st.columns([2, 1])
            with col1:
                fig_imp = create_feature_importance(results['feature_importance'])
                st.plotly_chart(fig_imp, use_container_width=True)
            
            with col2:
                st.subheader("🔍 Key Insights")
                top_feature = results['feature_importance'].iloc[0]
                st.info(f"""
                **Primary Driver**: {top_feature['feature']}
                
                This feature explains {top_feature['importance']:.1%} of regime variance.
                
                **Current Reading**:
                - {df[top_feature['feature']].iloc[-1]:.4f} (latest)
                - Trend: {'↗️ Rising' if df[top_feature['feature']].iloc[-5:].mean() > df[top_feature['feature']].iloc[-20:-5].mean() else '↘️ Falling'}
                """)
                
                # Risk metrics
                st.subheader("⚠️ Risk Metrics")
                var_95 = np.percentile(df['Returns'].dropna(), 5) * 100
                max_dd = ((df['SPY_Close'] / df['SPY_Close'].cummax()) - 1).min() * 100
                
                st.metric("95% VaR (daily)", f"{var_95:.2f}%")
                st.metric("Max Drawdown", f"{max_dd:.2f}%")
        
        # Footer
        st.markdown("---")
        st.caption(f"""
        💡 **How it works**: This model uses a Gaussian Hidden Markov Model to identify latent market states 
        based on volatility patterns, momentum, interest rates, and macro indicators. 
        Unlike simple moving average crossovers, it captures the probability of being in each regime 
        and the likelihood of transitions between them.
        
        Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} IST | Data source: Yahoo Finance | Auto-refresh: {'ON (' + str(refresh_interval) + 's)' if auto_refresh else 'OFF'}
        """)
        
        # Custom Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; padding: 20px; background: linear-gradient(90deg, #1f77b4, #ff7f0e); border-radius: 10px; margin-top: 30px;'>
            <h3 style='color: white; margin: 0;'>Made By Sourish Dey</h3>
            <p style='margin: 5px 0;'>
                <a href='https://sourishdeyportfolio.vercel.app/' target='_blank' style='color: white; text-decoration: underline; font-size: 16px;'>🔗 Visit My Portfolio</a>
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()