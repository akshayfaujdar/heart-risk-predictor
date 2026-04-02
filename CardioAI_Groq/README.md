# 🫀 CardioAI — Heart Disease Analysis & Prevention System

> End-to-end ML system for cardiovascular risk prediction  
> Trained on **70,000 patient records** · Best accuracy **99.31 %** (Gradient Boosting)

---

## 📁 Project Structure

```
CardioAI/
├── backend/
│   ├── main.py            ← FastAPI REST API (5 endpoints)
│   ├── model.py           ← ML training & inference
│   ├── chatbot.py         ← Claude AI health chatbot
│   ├── database.py        ← MySQL schema (SQLAlchemy)
│   ├── requirements.txt   ← Python dependencies
│   └── Dockerfile
├── frontend/
│   └── heart_disease_ai_system.html  ← Full SPA (6 pages)
├── dataset/
│   └── heart_disease_risk_dataset_earlymed.csv
├── docker/
│   ├── docker-compose.yml ← MySQL + Backend + Frontend
│   ├── nginx.conf
│   └── init.sql
├── docs/
│   └── (screenshots, API docs)
├── .env.example
└── README.md
```

---

## 🏆 Model Results

| Model               | Accuracy  | Precision | Recall   | F1       |
|---------------------|-----------|-----------|----------|----------|
| **Gradient Boosting** ⭐ | **99.31 %** | **99.26 %** | **99.37 %** | **99.31 %** |
| Random Forest       | 99.20 %   | 99.24 %   | 99.16 %  | 99.20 %  |
| Logistic Regression | 99.19 %   | 99.14 %   | 99.23 %  | 99.18 %  |
| Decision Tree       | 98.08 %   | 98.15 %   | 98.00 %  | 98.07 %  |

Confusion matrix (Gradient Boosting, 14 000 test samples): **only 96 errors**.

---

## 🚀 Local Setup (without Docker)

### 1. MySQL — create the database

```sql
CREATE DATABASE cardioai_db CHARACTER SET utf8mb4;
```

### 2. Backend

```bash
cd backend
pip install -r requirements.txt

# Copy and edit environment variables
cp ../.env.example ../.env
# Edit DATABASE_URL and ANTHROPIC_API_KEY in .env

# Train the model (first run only — takes ~2 min)
python model.py

# Start the API
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API docs → http://localhost:8000/docs

### 3. Frontend

Simply open `frontend/heart_disease_ai_system.html` in your browser.

---

## 🐳 Docker (Full Stack)

```bash
# 1. Add your Anthropic API key
cp .env.example .env
# Edit ANTHROPIC_API_KEY in .env

# 2. Build & run everything (MySQL + FastAPI + Nginx)
docker-compose -f docker/docker-compose.yml up --build

# 3. Train the model inside the container (first run)
docker exec cardioai_backend python model.py
```

| Service  | URL                        |
|----------|----------------------------|
| Frontend | http://localhost:3000       |
| API      | http://localhost:8000       |
| API Docs | http://localhost:8000/docs  |

---

## 🔌 API Endpoints

| Method | Endpoint           | Description                         |
|--------|--------------------|-------------------------------------|
| GET    | `/`                | Health check                        |
| POST   | `/predict`         | Heart disease risk prediction        |
| POST   | `/chat`            | AI health chatbot (Claude Sonnet)    |
| GET    | `/analysis`        | Dataset stats & model metrics        |
| POST   | `/patient/save`    | Save patient + prediction to MySQL   |
| GET    | `/patient/history` | List recent predictions from MySQL   |

### Example — POST /predict

**Request:**
```json
{
  "Age": 62,
  "Gender": 1,
  "Chest_Pain": 1,
  "Shortness_of_Breath": 1,
  "Fatigue": 1,
  "High_BP": 1,
  "Smoking": 1,
  "Family_History": 1
}
```

**Response:**
```json
{
  "success": true,
  "prediction": {
    "probability": 0.9214,
    "risk_percent": 92.1,
    "prediction": 1,
    "risk_level": "High"
  },
  "model": "Gradient Boosting",
  "model_accuracy": 0.9931
}
```

---

## 🗄️ MySQL Schema

| Table         | Purpose                              |
|---------------|--------------------------------------|
| `users`       | User accounts (optional auth)        |
| `patients`    | Patient clinical records             |
| `predictions` | ML prediction results per patient    |
| `chat_history`| AI chatbot conversation logs         |

Tables are created automatically by SQLAlchemy on first startup.

---

## 📊 Dataset

- **File:** `heart_disease_risk_dataset_earlymed.csv`
- **Records:** 70,000 (50 % high risk / 50 % low risk — perfectly balanced)
- **Features:** 18 clinical attributes (symptoms + lifestyle risk factors)
- **Target:** `Heart_Risk` (0 = low, 1 = high)

### Top Features by Importance

| Rank | Feature              | Importance |
|------|----------------------|------------|
| 1    | Age                  | 14.3 %     |
| 2    | Cold Sweats / Nausea | 11.6 %     |
| 3    | Fatigue              | 11.5 %     |
| 4    | Dizziness            |  9.6 %     |
| 5    | Shortness of Breath  |  9.5 %     |

---

## 🤖 AI Chatbot — Groq (FREE, No Credit Card)

The chatbot uses **Groq's free tier** with `llama-3.3-70b-versatile` — one of the most capable open-source models, running at ultra-fast speed on Groq's LPU hardware.

### How to get your free Groq API key:
1. Go to [https://console.groq.com](https://console.groq.com)
2. Sign up — no credit card required
3. Go to **API Keys** → **Create API Key**
4. Copy the key (starts with `gsk_...`)
5. Add to `.env`: `GROQ_API_KEY=gsk_your_key_here`

### For the frontend HTML chatbot:
Open `frontend/heart_disease_ai_system.html`, find this line and paste your key:
```javascript
const GROQ_API_KEY = 'YOUR_GROQ_API_KEY_HERE';
```

### Free tier limits (very generous):
| Model | Requests/min | Tokens/min | Tokens/day |
|---|---|---|---|
| llama-3.3-70b-versatile | 30 | 6,000 | 500,000 |

---

## 🌐 Production Deployment

| Layer     | Recommended Platform          |
|-----------|-------------------------------|
| Frontend  | Vercel / Netlify              |
| Backend   | Render / Railway / AWS EC2    |
| Database  | PlanetScale (MySQL) / AWS RDS |
| Model     | Loaded from disk via joblib   |

---

## ⚠️ Medical Disclaimer

This system is for **educational and research purposes only**.  
It does not constitute medical advice, diagnosis, or treatment.  
Always consult a qualified healthcare professional.

---

## 📄 License

MIT — free for personal and educational use.
