from fastapi import FastAPI, APIRouter, HTTPException, Header, Depends, Request
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict, EmailStr
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import secrets
import numpy as np
import tensorflow as tf
from io import BytesIO
import base64
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import bcrypt
from contextlib import asynccontextmanager

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# MongoDB connection
mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ.get('DB_NAME', 'finsecure_db')]

# ============= GLOBAL STATE =============
GLOBAL_MODEL = None
MODEL_VERSION = "1.0.0"
CURRENT_ROUND = 0
PREVIOUS_ACCURACY = 0.85
AGGREGATION_THRESHOLD = 2  # Minimum gradients before aggregation

scheduler = AsyncIOScheduler()

# ============= ML SETUP =============

def create_fraud_detection_model():
    """Create a simple neural network for fraud detection"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(30,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    return model

# ============= LIFESPAN MANAGER (Modern Startup/Shutdown) =============

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    global GLOBAL_MODEL, CURRENT_ROUND
    logger.info("Starting up FinSecure Backend...")
    
    # 1. Initialize Model
    GLOBAL_MODEL = create_fraud_detection_model()
    
    # 2. Get latest round info from DB to restore state
    latest_round = await db.training_rounds.find_one(
        {}, 
        sort=[("round_number", -1)]
    )
    if latest_round:
        CURRENT_ROUND = latest_round['round_number'] + 1
        logger.info(f"Restored state: Starting at Round {CURRENT_ROUND}")
    else:
        CURRENT_ROUND = 0
        logger.info("No previous history found. Starting at Round 0")

    # 3. Start Scheduler
    scheduler.add_job(
        auto_aggregate_gradients,
        'interval',
        minutes=5,
        id='auto_aggregate',
        replace_existing=True
    )
    scheduler.start()
    logger.info("Scheduler started.")
    
    yield
    
    # --- SHUTDOWN ---
    logger.info("Shutting down...")
    scheduler.shutdown()
    client.close()

# Create the main app with lifespan
app = FastAPI(lifespan=lifespan)
api_router = APIRouter(prefix="/api")

# ============= MODELS =============

class Company(BaseModel):
    model_config = ConfigDict(extra="ignore")
    company_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    email: EmailStr
    password_hash: str
    api_key: str
    status: str = "active"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class CompanyRegister(BaseModel):
    name: str
    email: EmailStr
    password: str

class CompanyLogin(BaseModel):
    email: EmailStr
    password: str

class CompanyResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    company_id: str
    name: str
    email: str
    api_key: str
    status: str
    created_at: datetime

class LoginResponse(BaseModel):
    success: bool
    company_id: str
    name: str
    email: str
    api_key: str
    message: str

class GradientSubmit(BaseModel):
    gradient_data: str
    metrics: Dict[str, float]

class DashboardStats(BaseModel):
    total_companies: int
    active_companies: int
    total_rounds: int
    current_accuracy: float
    total_updates: int
    latest_round: Optional[Dict[str, Any]]

class Notification(BaseModel):
    model_config = ConfigDict(extra="ignore")
    notification_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    company_id: Optional[str] = None
    title: str
    message: str
    type: str = "info"
    read: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# ============= HELPER FUNCTIONS =============

async def verify_api_key(x_api_key: str = Header(...)):
    """Verify API key from header"""
    company = await db.companies.find_one({"api_key": x_api_key, "status": "active"}, {"_id": 0})
    if not company:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return company

def generate_api_key():
    return f"fs_{secrets.token_urlsafe(32)}"

def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))

def serialize_model_weights(model):
    weights = model.get_weights()
    buffer = BytesIO()
    np.savez_compressed(buffer, *weights)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def deserialize_model_weights(data_str):
    try:
        data = base64.b64decode(data_str)
        buffer = BytesIO(data)
        npz_file = np.load(buffer, allow_pickle=True)
        weights = [npz_file[f'arr_{i}'] for i in range(len(npz_file.files))]
        return weights
    except Exception as e:
        logger.error(f"Error deserializing weights: {str(e)}")
        raise

def federated_averaging(gradient_list):
    if not gradient_list:
        return None
    avg_gradients = []
    for i in range(len(gradient_list[0])):
        layer_gradients = [g[i] for g in gradient_list]
        avg_gradients.append(np.mean(layer_gradients, axis=0))
    return avg_gradients

async def broadcast_notification(title: str, message: str, notification_type: str = "info"):
    companies = await db.companies.find({"status": "active"}, {"_id": 0}).to_list(1000)
    notifications = []
    for company in companies:
        notifications.append({
            "notification_id": str(uuid.uuid4()),
            "company_id": company['company_id'],
            "title": title,
            "message": message,
            "type": notification_type,
            "read": False,
            "created_at": datetime.now(timezone.utc).isoformat()
        })
    if notifications:
        await db.notifications.insert_many(notifications)

# ============= CORE LOGIC =============

async def aggregate_gradients():
    """Aggregate gradients and update global model"""
    global GLOBAL_MODEL, CURRENT_ROUND, MODEL_VERSION, PREVIOUS_ACCURACY
    
    round_id = f"round_{CURRENT_ROUND}"
    
    updates = await db.gradient_updates.find(
        {"round_id": round_id},
        {"_id": 0}
    ).to_list(1000)
    
    if not updates or len(updates) < AGGREGATION_THRESHOLD:
        return {
            "success": False, 
            "message": f"Waiting for gradients ({len(updates)}/{AGGREGATION_THRESHOLD})"
        }
    
    logger.info(f"Aggregating {len(updates)} updates for Round {CURRENT_ROUND}")

    # Deserialize and Average
    gradients = [deserialize_model_weights(update['gradient_data']) for update in updates]
    avg_gradients = federated_averaging(gradients)
    
    # Update Model
    GLOBAL_MODEL.set_weights(avg_gradients)
    
    # Metrics
    avg_accuracy = np.mean([update['metrics'].get('accuracy', 0) for update in updates])
    avg_loss = np.mean([update['metrics'].get('loss', 0) for update in updates])
    
    # Save Round
    training_round = {
        "round_id": round_id,
        "round_number": CURRENT_ROUND,
        "participating_companies": [u['company_id'] for u in updates],
        "avg_accuracy": float(avg_accuracy),
        "avg_loss": float(avg_loss),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    await db.training_rounds.insert_one(training_round)
    
    # Notification Logic
    accuracy_improvement = avg_accuracy - PREVIOUS_ACCURACY
    if accuracy_improvement > 0.01:
        await broadcast_notification(
            title="🎯 Model Performance Improved!",
            message=f"Accuracy +{accuracy_improvement*100:.2f}% (Now {avg_accuracy*100:.2f}%). Round {CURRENT_ROUND} complete.",
            notification_type="success"
        )
    else:
         await broadcast_notification(
            title="✓ Training Round Complete",
            message=f"Round {CURRENT_ROUND} done. Accuracy: {avg_accuracy*100:.2f}%",
            notification_type="info"
        )
    
    PREVIOUS_ACCURACY = avg_accuracy
    CURRENT_ROUND += 1
    
    return {
        "success": True,
        "round_number": CURRENT_ROUND - 1,
        "avg_accuracy": float(avg_accuracy)
    }

async def auto_aggregate_gradients():
    try:
        await aggregate_gradients()
    except Exception as e:
        logger.error(f"Error in auto-aggregation: {str(e)}")

# ============= API ROUTES =============

@api_router.post("/auth/register", response_model=CompanyResponse)
async def register_company(company_input: CompanyRegister):
    existing = await db.companies.find_one({"email": company_input.email})
    if existing:
        raise HTTPException(status_code=400, detail="Company email already exists")
    
    if len(company_input.password) < 8:
        raise HTTPException(status_code=400, detail="Password too short")
    
    api_key = generate_api_key()
    password_hash = hash_password(company_input.password)
    
    company = Company(
        name=company_input.name,
        email=company_input.email,
        password_hash=password_hash,
        api_key=api_key
    )
    
    doc = company.model_dump()
    doc['created_at'] = doc['created_at'].isoformat()
    await db.companies.insert_one(doc)
    return company

@api_router.post("/auth/login", response_model=LoginResponse)
async def login_company(login_input: CompanyLogin):
    company = await db.companies.find_one({"email": login_input.email}, {"_id": 0})
    if not company or not verify_password(login_input.password, company['password_hash']):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if company.get('status') != 'active':
        raise HTTPException(status_code=403, detail="Account inactive")
    
    return LoginResponse(
        success=True,
        company_id=company['company_id'],
        name=company['name'],
        email=company['email'],
        api_key=company['api_key'],
        message="Login successful"
    )

@api_router.get("/auth/verify")
async def verify_key(company: dict = Depends(verify_api_key)):
    return {"valid": True, "company_id": company['company_id'], "name": company['name']}

@api_router.get("/model/download")
async def download_model(company: dict = Depends(verify_api_key)):
    weights_str = serialize_model_weights(GLOBAL_MODEL)
    return {
        "version": MODEL_VERSION,
        "weights": weights_str,
        "round": CURRENT_ROUND
    }

@api_router.post("/federated/submit-gradients")
async def submit_gradients(gradient_submit: GradientSubmit, company: dict = Depends(verify_api_key)):
    global CURRENT_ROUND
    round_id = f"round_{CURRENT_ROUND}"
    
    gradient_update = {
        "update_id": str(uuid.uuid4()),
        "company_id": company['company_id'],
        "round_id": round_id,
        "gradient_data": gradient_submit.gradient_data,
        "metrics": gradient_submit.metrics,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    await db.gradient_updates.insert_one(gradient_update)
    return {"success": True, "round_id": round_id, "message": "Gradients received"}

@api_router.get("/analytics/dashboard", response_model=DashboardStats)
async def get_dashboard_stats():
    total_companies = await db.companies.count_documents({})
    active_companies = await db.companies.count_documents({"status": "active"})
    total_rounds = await db.training_rounds.count_documents({})
    total_updates = await db.gradient_updates.count_documents({})
    
    latest_round = await db.training_rounds.find_one({}, {"_id": 0}, sort=[("round_number", -1)])
    current_accuracy = latest_round.get('avg_accuracy', 0.85) if latest_round else 0.85
    
    return DashboardStats(
        total_companies=total_companies,
        active_companies=active_companies,
        total_rounds=total_rounds,
        current_accuracy=current_accuracy,
        total_updates=total_updates,
        latest_round=latest_round
    )

@api_router.get("/client/script")
async def get_client_script(request: Request, company: dict = Depends(verify_api_key)):
    """Get customized client script"""
    # Dynamically determine the backend URL based on the request
    base_url = str(request.base_url).rstrip('/')
    api_url = f"{base_url}/api"

    script_template = '''#!/usr/bin/env python3
"""
FinSecure Federated Learning - Client Script
Company: {company_name}
"""
import tensorflow as tf
import numpy as np
import requests
import base64
from io import BytesIO

# Configuration
API_KEY = "{api_key}"
BACKEND_URL = "{backend_url}" 

class FederatedClient:
    def __init__(self, api_key, backend_url):
        self.api_key = api_key
        self.backend_url = backend_url
        self.model = None
        self.headers = {{"X-API-Key": self.api_key}}
    
    def download_model(self):
        print(f"Connecting to {{self.backend_url}}...")
        response = requests.get(f"{{self.backend_url}}/model/download", headers=self.headers)
        if response.status_code == 200:
            data = response.json()
            self.model = self._build_model()
            weights = self._deserialize_weights(data['weights'])
            self.model.set_weights(weights)
            print(f"✓ Model downloaded (Round {{data['round']}})")
        else:
            print(f"✗ Failed to download model: {{response.text}}")

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(30,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def _deserialize_weights(self, data_str):
        data = base64.b64decode(data_str)
        buffer = BytesIO(data)
        npz_file = np.load(buffer, allow_pickle=True)
        return [npz_file[f'arr_{{i}}'] for i in range(len(npz_file.files))]

    def _serialize_weights(self, weights):
        buffer = BytesIO()
        np.savez_compressed(buffer, *weights)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')

    def train_local(self, X_train, y_train, epochs=3):
        if not self.model: return
        print(f"Training on {{len(X_train)}} samples...")
        history = self.model.fit(X_train, y_train, epochs=epochs, verbose=0, validation_split=0.2)
        
        metrics = {{
            'accuracy': float(history.history['accuracy'][-1]),
            'loss': float(history.history['loss'][-1])
        }}
        
        # Submit Gradients
        weights = self.model.get_weights()
        gradient_data = self._serialize_weights(weights)
        
        resp = requests.post(
            f"{{self.backend_url}}/federated/submit-gradients",
            headers=self.headers,
            json={{"gradient_data": gradient_data, "metrics": metrics}}
        )
        if resp.status_code == 200:
            print("✓ Gradients submitted successfully")
        else:
            print(f"✗ Submission failed: {{resp.text}}")

# Run
if __name__ == "__main__":
    client = FederatedClient(API_KEY, BACKEND_URL)
    client.download_model()
    
    # DUMMY DATA (Replace with real data)
    X = np.random.randn(100, 30).astype(np.float32)
    y = np.random.randint(0, 2, 100).astype(np.float32)
    
    client.train_local(X, y)
'''
    return {
        "filename": "client.py",
        "content": script_template.format(
            company_name=company['name'],
            api_key=company['api_key'],
            backend_url=api_url
        )
    }

@api_router.get("/notifications", response_model=List[Notification])
async def get_notifications(company: dict = Depends(verify_api_key)):
    notifications = await db.notifications.find(
        {"$or": [{"company_id": company['company_id']}, {"company_id": None}]},
        {"_id": 0}
    ).sort("created_at", -1).limit(50).to_list(50)
    return notifications

# Include API Router
app.include_router(api_router)

# ============= CORS SETUP (CRITICAL FIX) =============
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow ALL origins (Vercel, Localhost, etc.)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
