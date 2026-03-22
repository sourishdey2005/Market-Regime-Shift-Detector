# regime_detector.py
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class MarketRegimeDetector:
    def __init__(self, lookback_days=252):
        self.lookback_days = lookback_days
        self.hmm_model = None
        self.scaler = StandardScaler()
        self.regime_labels = {0: 'Bear', 1: 'Sideways', 2: 'Bull'}
        self.regime_colors = {'Bear': '#FF4B4B', 'Sideways': '#FFA500', 'Bull': '#00CC96'}
        self.regime_emoji = {'Bear': '🔴', 'Sideways': '🟡', 'Bull': '🟢'}
        
    def fetch_macro_data(self):
        """Fetch VIX, Interest Rates, Inflation proxies, and Price data"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days + 50)
        
        # Force fresh data download (disable caching)
        ticker_kwargs = {
            'start': start_date, 
            'end': end_date, 
            'progress': False,
            'auto_adjust': False,
            'actions': False
        }
        
        # Market data - fetch fresh each time
        spy = yf.download('SPY', **ticker_kwargs)
        vix = yf.download('^VIX', **ticker_kwargs)
        
        # Interest rate proxy (10Y Treasury)
        tnx = yf.download('^TNX', **ticker_kwargs)
        
        # Inflation proxy (TIPS breakeven or commodities)
        # Using DBC (Commodities) as inflation proxy
        dbc = yf.download('DBC', **ticker_kwargs)
        
        # Gold as alternative macro indicator
        gld = yf.download('GLD', **ticker_kwargs)
        
        # USD Strength
        uup = yf.download('UUP', **ticker_kwargs)
        
        data = pd.DataFrame(index=spy.index)
        data['SPY_Open'] = spy['Open']
        data['SPY_High'] = spy['High']
        data['SPY_Low'] = spy['Low']
        data['SPY_Close'] = spy['Close']
        data['SPY_Volume'] = spy['Volume']
        
        # Handle multi-index columns from yfinance
        data['VIX'] = vix['Close'] if not vix.empty else np.nan
        data['Yield_10Y'] = tnx['Close'] if not tnx.empty else np.nan
        data['Commodities'] = dbc['Close'] if not dbc.empty else np.nan
        data['Gold'] = gld['Close'] if not gld.empty else np.nan
        data['USD'] = uup['Close'] if not uup.empty else np.nan
        
        # Forward fill missing values
        data = data.ffill().dropna()
        
        if len(data) < 50:
            raise ValueError("Insufficient data for analysis. Need at least 50 trading days of data.")
        
        return data
    
    def calculate_features(self, data):
        """Calculate technical and macro features for regime detection"""
        df = data.copy()
        
        # Price-based features
        df['Returns'] = df['SPY_Close'].pct_change()
        df['Log_Returns'] = np.log(df['SPY_Close'] / df['SPY_Close'].shift(1))
        
        # Volatility (realized)
        df['Realized_Vol'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
        
        # Trend features
        df['SMA_20'] = df['SPY_Close'].rolling(window=20).mean()
        df['SMA_50'] = df['SPY_Close'].rolling(window=50).mean()
        df['Trend_Strength'] = (df['SPY_Close'] - df['SMA_20']) / df['SMA_20']
        
        # Momentum
        df['Momentum_10'] = df['SPY_Close'].pct_change(10)
        df['Momentum_30'] = df['SPY_Close'].pct_change(30)
        
        # VIX features
        df['VIX_MA20'] = df['VIX'].rolling(window=20).mean()
        df['VIX_Spike'] = (df['VIX'] - df['VIX_MA20']) / df['VIX_MA20']
        
        # Rate change
        df['Yield_Change'] = df['Yield_10Y'].diff()
        
        # Volume features
        df['Volume_MA20'] = df['SPY_Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['SPY_Volume'] / df['Volume_MA20']
        
        # Macro correlations
        df['Gold_Momentum'] = df['Gold'].pct_change(10)
        df['Commodity_Momentum'] = df['Commodities'].pct_change(10)
        df['USD_Momentum'] = df['USD'].pct_change(10)
        
        # Drop NaN
        df = df.dropna()
        
        return df
    
    def prepare_features_for_model(self, df):
        """Select and scale features for HMM"""
        feature_cols = [
            'Returns', 'Realized_Vol', 'Trend_Strength', 
            'Momentum_10', 'Momentum_30', 'VIX', 'VIX_Spike',
            'Yield_Change', 'Volume_Ratio', 'Gold_Momentum',
            'Commodity_Momentum', 'USD_Momentum'
        ]
        
        features = df[feature_cols].copy()
        # Handle any remaining infinities or NaNs
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        return features
    
    def fit_hmm(self, features, n_regimes=3):
        """Fit Hidden Markov Model to detect regimes"""
        # Scale features
        X = self.scaler.fit_transform(features)
        
        # Fit HMM
        self.hmm_model = GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        
        self.hmm_model.fit(X)
        
        # Predict regimes
        regimes = self.hmm_model.predict(X)
        
        # Calculate confidence (log-likelihood normalized)
        score_result = self.hmm_model.score_samples(X)
        if isinstance(score_result, tuple):
            log_likelihood = score_result[1]  # per-sample log-likelihoods
        else:
            log_likelihood = score_result
        # Ensure it's an array
        log_likelihood = np.asarray(log_likelihood)
        # Convert to probability-like confidence (0-1 scale)
        confidence = 1 / (1 + np.exp(-log_likelihood / 10))
        
        return regimes, confidence
    
    def label_regimes(self, df, regimes):
        """Label regimes based on characteristics"""
        df['Regime'] = regimes
        
        # Analyze regime characteristics to label them correctly
        regime_stats = df.groupby('Regime').agg({
            'Returns': 'mean',
            'Realized_Vol': 'mean',
            'VIX': 'mean',
            'Trend_Strength': 'mean'
        })
        
        # Sort regimes by return (Bull > Sideways > Bear)
        sorted_regimes = regime_stats['Returns'].sort_values(ascending=False).index
        
        mapping = {
            sorted_regimes[0]: 'Bull',
            sorted_regimes[1]: 'Sideways',
            sorted_regimes[2]: 'Bear'
        }
        
        df['Regime_Label'] = df['Regime'].map(mapping)
        
        return df, mapping
    
    def detect_regime_change(self, df):
        """Detect points where regime changed"""
        df['Regime_Change'] = df['Regime_Label'] != df['Regime_Label'].shift(1)
        changes = df[df['Regime_Change']].copy()
        return changes
    
    def get_current_regime(self, df, confidence):
        """Get current market regime with confidence"""
        current = df.iloc[-1]
        current_regime = current['Regime_Label']
        current_confidence = float(np.asarray(confidence).flatten()[-1])
        
        # Calculate regime duration
        current_regime_num = current['Regime']
        regime_start = None
        for i in range(len(df)-1, -1, -1):
            if df.iloc[i]['Regime'] != current_regime_num:
                regime_start = df.index[i+1]
                break
        
        if regime_start is None:
            regime_start = df.index[0]
            
        duration_days = (df.index[-1] - regime_start).days
        
        return {
            'regime': current_regime,
            'confidence': current_confidence,
            'duration_days': duration_days,
            'regime_start': regime_start,
            'spy_price': current['SPY_Close'],
            'vix_level': current['VIX'],
            'volatility': current['Realized_Vol']
        }
    
    def run_detection(self):
        """Full pipeline"""
        print("Fetching market data...")
        raw_data = self.fetch_macro_data()
        
        print("Calculating features...")
        features_df = self.calculate_features(raw_data)
        
        print("Preparing model features...")
        X = self.prepare_features_for_model(features_df)
        
        print("Fitting HMM model...")
        regimes, confidence = self.fit_hmm(X)
        
        print("Labeling regimes...")
        labeled_df, mapping = self.label_regimes(features_df, regimes)
        
        print("Detecting regime changes...")
        changes = self.detect_regime_change(labeled_df)
        
        print("Getting current regime...")
        current = self.get_current_regime(labeled_df, confidence)
        
        return {
            'data': labeled_df,
            'changes': changes,
            'current': current,
            'confidence': confidence,
            'feature_importance': self._calculate_feature_importance(X, regimes)
        }
    
    def _calculate_feature_importance(self, X, regimes):
        """Calculate which features distinguish regimes most"""
        from sklearn.ensemble import RandomForestClassifier
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, regimes)
        
        feature_names = [
            'Returns', 'Realized_Vol', 'Trend_Strength', 
            'Momentum_10', 'Momentum_30', 'VIX', 'VIX_Spike',
            'Yield_Change', 'Volume_Ratio', 'Gold_Momentum',
            'Commodity_Momentum', 'USD_Momentum'
        ]
        
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance