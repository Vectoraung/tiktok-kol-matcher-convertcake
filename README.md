# TikTok Influencer Finder

A simple Python project for finding TikTok influencers.

---

## Setup Instructions

### 1. Create a virtual environment
```
python3 -m venv venv
```

### 2. Activate the virtual environment
Linux / macOS:
```
source venv/bin/activate
```

Windows (PowerShell):
```
venv\Scripts\activate
```

### 3. Install required libraries
```
pip install -r requirements.txt
```

### 4. Create ```.env``` file and add API keys
```
APIFY_API_TOKEN=<Your Apify API key here>
GOOGLE_API_KEY=<Your google gemini API key here>
```

### 4. Run the app
```
streamlit run app.py
```