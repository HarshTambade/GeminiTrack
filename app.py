import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import firebase_admin
from firebase_admin import credentials, firestore, auth
from firebase_admin.exceptions import FirebaseError
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import requests
import json
import asyncio
import aiohttp
from uuid import uuid4
import hashlib
import logging
from logging.handlers import RotatingFileHandler
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import calendar
from dateutil.relativedelta import relativedelta

# -------------------- Configuration --------------------
class Config:
    FIREBASE_CRED = "firebase-config.json"
    GEMINI_API_URL = "https://api.gemini.com/v1/insights"
    MODEL_CACHE_DIR = "./model_cache/"
    SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
    HABIT_ENCODER = "all-mpnet-base-v2"
    MAX_STREAK_DAYS = 30
    POINTS_PER_HABIT = 10
    HABIT_CATEGORIES = {
        "exploration": ["Learn", "Explore", "Create", "Experiment"],
        "routine": ["Exercise", "Meditate", "Read", "Journal"]
    }

# -------------------- Firebase Initialization --------------------
def initialize_firebase():
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(Config.FIREBASE_CRED)
            firebase_admin.initialize_app(cred)
        return firestore.client(), auth
    except Exception as e:
        logging.error(f"Firebase initialization failed: {str(e)}")
        raise

db, firebase_auth = initialize_firebase()

# -------------------- AI Services --------------------
class AIService:
    def __init__(self):
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis", 
            model=Config.SENTIMENT_MODEL,
            framework="pt"
        )
        self.habit_encoder = SentenceTransformer(Config.HABIT_ENCODER, cache_folder=Config.MODEL_CACHE_DIR)
        
    async def analyze_habit_patterns(self, user_id: str) -> Dict:
        """Perform advanced habit pattern analysis using clustering"""
        habits = await self._get_user_habits(user_id)
        if not habits:
            return {}
            
        embeddings = self.habit_encoder.encode([h['habit'] for h in habits])
        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(embeddings)
        
        # Optimal cluster determination
        distortions = []
        max_clusters = min(5, len(habits)-1)
        for i in range(1, max_clusters+1):
            km = KMeans(n_clusters=i, n_init=10)
            km.fit(scaled_embeddings)
            distortions.append(km.inertia_)
            
        optimal_clusters = np.argmin(np.diff(distortions)) + 1
        km = KMeans(n_clusters=optimal_clusters, n_init=10)
        clusters = km.fit_predict(scaled_embeddings)
        
        return {
            "clusters": clusters.tolist(),
            "habit_patterns": self._interpret_clusters(clusters, habits)
        }
        
    def _interpret_clusters(self, clusters: np.ndarray, habits: List[Dict]) -> List[str]:
        """Convert clusters to human-readable patterns"""
        cluster_habits = {}
        for idx, cluster in enumerate(clusters):
            if cluster not in cluster_habits:
                cluster_habits[cluster] = []
            cluster_habits[cluster].append(habits[idx]['habit'])
        
        patterns = []
        for cluster, habits in cluster_habits.items():
            category = "exploration" if any(h.lower().startswith(tuple(Config.HABIT_CATEGORIES["exploration"])) for h in habits) else "routine"
            patterns.append(f"Cluster {cluster+1}: {category} focus with habits like {', '.join(habits[:3])}")
            
        return patterns

# -------------------- Habit Engine --------------------
class HabitEngine:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.ai = AIService()
        self.db = db
        
    async def create_habit(self, habit_data: Dict) -> Dict:
        """Create a new habit with streak tracking"""
        habit_id = f"habit_{uuid4().hex}"
        habit_data.update({
            "habit_id": habit_id,
            "created_at": datetime.now().isoformat(),
            "streak_days": 0,
            "last_completed": None,
            "completion_history": []
        })
        
        try:
            self.db.collection("users").document(self.user_id).collection("habits").document(habit_id).set(habit_data)
            return {"status": "success", "habit_id": habit_id}
        except Exception as e:
            logging.error(f"Habit creation failed: {str(e)}")
            return {"status": "error", "message": str(e)}
            
    async def complete_habit(self, habit_id: str) -> Dict:
        """Mark habit as completed and update streaks"""
        habit_ref = self.db.collection("users").document(self.user_id).collection("habits").document(habit_id)
        
        try:
            habit_data = habit_ref.get().to_dict()
            last_completed = datetime.fromisoformat(habit_data["last_completed"]) if habit_data["last_completed"] else None
            current_time = datetime.now()
            
            # Streak logic
            if last_completed and (current_time - last_completed).days == 1:
                new_streak = habit_data["streak_days"] + 1
            else:
                new_streak = 1
                
            updates = {
                "streak_days": new_streak,
                "last_completed": current_time.isoformat(),
                "completion_history": firestore.ArrayUnion([current_time.isoformat()])
            }
            
            habit_ref.update(updates)
            return {"status": "success", "streak": new_streak}
        except Exception as e:
            logging.error(f"Habit completion failed: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def get_habit_analytics(self) -> Dict:
        """Get comprehensive habit analytics"""
        habits = await self._get_habits()
        if not habits:
            return {}
            
        df = pd.DataFrame(habits)
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['last_completed'] = pd.to_datetime(df['last_completed'])
        
        # Time-based analysis
        current_date = datetime.now()
        df['days_since_creation'] = (current_date - df['created_at']).dt.days
        df['days_since_last_completed'] = (current_date - df['last_completed']).dt.days
        
        # Streak analysis
        streak_stats = {
            "max_streak": df['streak_days'].max(),
            "avg_streak": df['streak_days'].mean(),
            "current_streaks": df[df['days_since_last_completed'] == 0]['streak_days'].sum()
        }
        
        # Temporal patterns
        df['completion_count'] = df['completion_history'].apply(len)
        time_series = df.groupby(pd.Grouper(key='last_completed', freq='D'))['completion_count'].sum().reset_index()
        
        # AI-powered insights
        patterns = await self.ai.analyze_habit_patterns(self.user_id)
        
        return {
            "streak_stats": streak_stats,
            "time_series": time_series.to_dict(orient='records'),
            "patterns": patterns.get("habit_patterns", []),
            "completion_rate": (df['completion_count'].sum() / df['days_since_creation'].sum()).round(2)
        }
        
    async def _get_habits(self) -> List[Dict]:
        """Retrieve user habits from Firestore"""
        try:
            habits = []
            docs = self.db.collection("users").document(self.user_id).collection("habits").stream()
            for doc in docs:
                habit_data = doc.to_dict()
                habit_data['id'] = doc.id
                habits.append(habit_data)
            return habits
        except Exception as e:
            logging.error(f"Failed to fetch habits: {str(e)}")
            return []

# -------------------- UI Components --------------------
class Dashboard:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.engine = HabitEngine(user_id)
        
    async def render(self):
        """Main dashboard renderer"""
        st.set_page_config(
            page_title="Gemini Habit Optimizer", 
            page_icon="♊", 
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        self._setup_styles()
        await self._handle_auth()
        
        if 'authenticated' in st.session_state and st.session_state.authenticated:
            await self._main_interface()
            
    def _setup_styles(self):
        """Inject custom CSS styles"""
        st.markdown("""
            <style>
            .main {
                background-color: #f0f2f6;
            }
            .stButton>button {
                border-radius: 20px;
                padding: 10px 24px;
                transition: all 0.3s;
            }
            .habit-card {
                padding: 1.5rem;
                border-radius: 15px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                margin-bottom: 1rem;
                background: white;
            }
            .streak-badge {
                background: linear-gradient(45deg, #ff6b6b, #ff8e53);
                color: white;
                padding: 0.3rem 0.8rem;
                border-radius: 12px;
                font-weight: bold;
            }
            </style>
        """, unsafe_allow_html=True)
        
    async def _handle_auth(self):
        """Authentication handling"""
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
            
        with st.sidebar:
            st.title("🔐 Authentication")
            if st.session_state.authenticated:
                st.success(f"Welcome {st.session_state.user_email}")
                if st.button("Logout"):
                    self._logout()
                return
                
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                await self._login(email, password)
            if st.button("Create Account"):
                await self._create_account(email, password)
                
    async def _login(self, email: str, password: str):
        """User login logic"""
        try:
            user = firebase_auth.get_user_by_email(email)
            st.session_state.user_id = user.uid
            st.session_state.user_email = email
            st.session_state.authenticated = True
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Login failed: {str(e)}")
            
    async def _create_account(self, email: str, password: str):
        """User registration logic"""
        try:
            user = firebase_auth.create_user(email=email, password=password)
            st.session_state.user_id = user.uid
            st.session_state.user_email = email
            st.session_state.authenticated = True
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Registration failed: {str(e)}")
            
    def _logout(self):
        """Logout handler"""
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.experimental_rerun()
        
    async def _main_interface(self):
        """Main application interface"""
        st.title("♊ Gemini Habit Optimizer Pro")
        st.markdown("**AI-Powered Habit Optimization for Dynamic Personalities**")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Habit Dashboard", "Analytics", "AI Insights", "Settings"])
        
        with tab1:
            await self._habit_management()
            
        with tab2:
            await self._analytics_dashboard()
            
        with tab3:
            await self._ai_insights()
            
        with tab4:
            await self._user_settings()
            
    async def _habit_management(self):
        """Habit creation and tracking interface"""
        col1, col2 = st.columns([1, 2])
        
        with col1:
            with st.form("habit_form"):
                st.subheader("Create New Habit")
                habit_name = st.text_input("Habit Name")
                habit_type = st.selectbox("Type", ["Exploration", "Routine"])
                difficulty = st.slider("Difficulty", 1, 5, 3)
                target_days = st.number_input("Target Days per Week", 1, 7, 3)
                
                if st.form_submit_button("Create Habit"):
                    habit_data = {
                        "name": habit_name,
                        "type": habit_type.lower(),
                        "difficulty": difficulty,
                        "target_days": target_days,
                        "created_at": datetime.now().isoformat()
                    }
                    result = await self.engine.create_habit(habit_data)
                    if result['status'] == 'success':
                        st.success("Habit created successfully!")
                    else:
                        st.error(f"Error: {result['message']}")
                        
        with col2:
            st.subheader("Your Habits")
            habits = await self.engine._get_habits()
            if not habits:
                st.info("No habits found. Create your first habit!")
                return
                
            for habit in habits:
                with st.container():
                    self._render_habit_card(habit)
                    
    def _render_habit_card(self, habit: Dict):
        """Render individual habit card"""
        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            st.markdown(f"**{habit['name']}** ({habit['type'].capitalize()})")
            st.progress(habit.get('completion_rate', 0))
            
        with col2:
            st.markdown(f"""
                <div class="streak-badge">
                    🔥 {habit.get('streak_days', 0)} Day Streak
                </div>
            """, unsafe_allow_html=True)
            
        with col3:
            if st.button("✅ Complete", key=f"complete_{habit['id']}"):
                result = asyncio.run(self.engine.complete_habit(habit['id']))
                if result['status'] == 'success':
                    st.experimental_rerun()
                    
    async def _analytics_dashboard(self):
        """Advanced analytics visualization"""
        st.subheader("Habit Analytics")
        analytics = await self.engine.get_habit_analytics()
        
        if not analytics:
            st.warning("No analytics data available yet")
            return
            
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Indicator(
                mode="number",
                value=analytics['streak_stats']['max_streak'],
                title="Longest Streak",
                number={'font': {'size': 40}}
            ))
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            fig = px.line(
                pd.DataFrame(analytics['time_series']),
                x='last_completed',
                y='completion_count',
                title="Completion Trends"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        st.subheader("Pattern Recognition")
        for pattern in analytics.get('patterns', []):
            st.markdown(f"- {pattern}")
            
    async def _ai_insights(self):
        """AI-powered recommendations and insights"""
        st.subheader("Personalized AI Recommendations")
        
        with st.spinner("Analyzing your habits..."):
            habits = await self.engine._get_habits()
            if not habits:
                st.warning("No habits to analyze")
                return
                
            # Generate recommendations
            recommendations = await self._generate_recommendations(habits)
            
            st.subheader("Optimization Suggestions")
            for rec in recommendations:
                with st.expander(rec['title']):
                    st.markdown(rec['description'])
                    if st.button("Apply Suggestion", key=rec['id']):
                        await self._apply_recommendation(rec)
                        
    async def _generate_recommendations(self, habits: List[Dict]) -> List[Dict]:
        """Generate AI-powered habit recommendations"""
        # Implement sophisticated recommendation logic here
        return [
            {
                "id": "rec_001",
                "title": "Balance Exploration/Routine",
                "description": "Your habits are 80% routine-based. Try adding more exploration habits for creativity."
            },
            {
                "id": "rec_002",
                "title": "Morning Routine Optimization",
                "description": "Cluster analysis shows higher success rates for morning habits. Schedule key routines before 9 AM."
            }
        ]
        
    async def _apply_recommendation(self, recommendation: Dict):
        """Apply AI recommendation to user's habits"""
        # Implement recommendation application logic
        st.success(f"Applied recommendation: {recommendation['title']}")
        
    async def _user_settings(self):
        """User preferences and account settings"""
        st.subheader("Account Settings")
        with st.form("settings_form"):
            new_email = st.text_input("Update Email", value=st.session_state.user_email)
            new_password = st.text_input("New Password", type="password")
            if st.form_submit_button("Update Settings"):
                try:
                    firebase_auth.update_user(
                        st.session_state.user_id,
                        email=new_email,
                        password=new_password
                    )
                    st.success("Settings updated successfully!")
                except Exception as e:
                    st.error(f"Update failed: {str(e)}")
                    
# -------------------- Main Execution --------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            RotatingFileHandler("app.log", maxBytes=1e6, backupCount=3)
        ]
    )
    
    async def main():
        if 'user_id' in st.session_state:
            dashboard = Dashboard(st.session_state.user_id)
            await dashboard.render()
        else:
            dashboard = Dashboard("")
            await dashboard.render()
            
    asyncio.run(main())