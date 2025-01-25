Here's the raw markdown text you can save directly as `README.md`. Copy all content below between the triple backticks:
# ♊ Gemini Habit Tracker Pro

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Firebase](https://img.shields.io/badge/Firebase-FFCA28?logo=firebase&logoColor=black)](https://firebase.google.com/)

**AI-Powered Productivity App for Gemini Personalities**  
*Balance your dual nature between exploration and routine with intelligent habit optimization*

## 🌟 Features

### Core Functionality
- **Dual Habit System**
  - 🚀 Exploration Mode: Flexible creative habits
  - 📅 Routine Mode: Structured daily rituals
- **AI-Powered Insights**
  - 🤖 Habit recommendations using ML
  - 🔮 Predictive analytics for habit success
  - 🧠 Sentiment-based suggestions
- **Advanced Tracking**
  - 📊 Interactive progress visualizations
  - 🔥 Streak tracking with gamification
  - 📈 Temporal pattern analysis

### Enterprise Features
- 🔐 Firebase Authentication
- 💾 Firestore Database Integration
- 📤 PDF Report Generation
- 🏆 Social Leaderboards
- 🔔 Smart Reminders

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- Firebase project with Firestore
- Gemini API key (optional)

### Setup Guide
```bash
# Clone repository
git clone https://github.com/yourusername/gemini-habit-tracker.git
cd gemini-habit-tracker

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Firebase Configuration
1. Create Firebase project at [console.firebase.google.com](https://console.firebase.google.com/)
2. Enable **Email/Password Authentication**
3. Initialize Firestore in test mode
4. Download service account JSON as `firebase-config.json`

### Environment Setup
Create `.streamlit/secrets.toml`:
```toml
[firebase]
project_id = "your-project-id"

[gemini]
api_key = "your-api-key"
```

## 🚀 Usage
```bash
# Start application
streamlit run app.py

# Access in browser at:
http://localhost:8501
```

## 🧩 Tech Stack

### Frontend
- Streamlit
- Plotly/Altair
- HTML/CSS

### Backend
- Python 3.10
- Firebase Firestore
- Scikit-learn
- Sentence Transformers

### AI/ML
- K-Means Clustering
- TF-IDF Vectorization
- DistilBERT Sentiment Analysis

## 🤝 Contributing

### Development Setup
1. Fork repository
2. Create feature branch:
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. Install dev dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```
4. Follow PEP8 guidelines
5. Submit PR with detailed documentation

## 📄 License
MIT License - See [LICENSE](LICENSE) for details

## 📧 Contact
For support/questions:  
📩 [23207001@apsit.edu.in](mailto:your.email@example.com)

---

**Made with ♊ by [Harsh Tambade]**  
*Empowering Gemini personalities since 2023*
