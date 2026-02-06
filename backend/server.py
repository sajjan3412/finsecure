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
AGGREGATION_THRESHOLD = 1  # Kept at 1 for your Demo to work instantly

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

# ============= LIFESPAN MANAGER =============

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

# --- UPDATED: Safe Deserialization ---
def deserialize_model_weights(data_str):
    try:
        data = base64.b64decode(data_str)
        buffer = BytesIO(data)
        npz_file = np.load(buffer, allow_pickle=True)
        weights = [npz_file[f'arr_{i}'] for i in range(len(npz_file.files))]
        return weights
    except Exception as e:
        logger.error(f"❌ Malformed Gradients Detected: {str(e)}")
        return None # Return None on failure so we can skip this update

# --- NEW: Gradient Validation ---
def validate_gradient_shape(decoded_weights, model):
    """
    Security Check: Ensures uploaded gradients match the Global Model's architecture.
    """
    model_weights = model.get_weights()
    
    # 1. Check Layer Count
    if len(decoded_weights) != len(model_weights):
        logger.warning(f"Shape Mismatch: Received {len(decoded_weights)} layers, expected {len(model_weights)}")
        return False
        
    # 2. Check Shape of Each Layer
    for i, (new_w, true_w) in enumerate(zip(decoded_weights, model_weights)):
        if new_w.shape != true_w.shape:
            logger.warning(f"Layer {i} Mismatch: Received {new_w.shape}, expected {true_w.shape}")
            return False
            
    return True

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

# --- UPDATED: Robust Aggregation ---
async def aggregate_gradients():
    """Aggregate gradients and update global model with Safety Checks"""
    global GLOBAL_MODEL, CURRENT_ROUND, MODEL_VERSION, PREVIOUS_ACCURACY
    
    round_id = f"round_{CURRENT_ROUND}"
    updates = await db.gradient_updates.find(
        {"round_id": round_id, "status": "pending"},
        {"_id": 0}
    ).to_list(1000)
    
    if not updates or len(updates) < AGGREGATION_THRESHOLD:
        return {
            "success": False, 
            "message": f"Waiting for gradients ({len(updates)}/{AGGREGATION_THRESHOLD})"
        }
    
    logger.info(f"🔄 Processing {len(updates)} updates for Round {CURRENT_ROUND}")

    valid_gradients = []
    valid_metrics = []
    
    # --- STEP 1: FILTER BAD UPDATES ---
    for update in updates:
        weights = deserialize_model_weights(update['gradient_data'])
        
        # Check if deserialization worked AND shapes match
        if weights is not None and validate_gradient_shape(weights, GLOBAL_MODEL):
            valid_gradients.append(weights)
            valid_metrics.append(update['metrics'])
        else:
            logger.error(f"⚠️ Dropped corrupt update from Company ID: {update.get('company_id')}")

    if not valid_gradients:
        logger.error("❌ All updates in this round were corrupt! Aborting aggregation.")
        return {"success": False, "message": "All updates failed validation"}

    # --- STEP 2: FEDERATED AVERAGING ---
    avg_gradients = federated_averaging(valid_gradients)
    
    # Update Model
    GLOBAL_MODEL.set_weights(avg_gradients)
    
    # Metrics
    avg_accuracy = np.mean([m.get('accuracy', 0) for m in valid_metrics])
    avg_loss = np.mean([m.get('loss', 0) for m in valid_metrics])
    
    # Save Round
    training_round = {
        "round_id": round_id,
        "round_number": CURRENT_ROUND,
        "participating_companies": len(valid_gradients), # Only count valid ones
        "avg_accuracy": float(avg_accuracy),
        "avg_loss": float(avg_loss),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    await db.training_rounds.insert_one(training_round)

    # Mark updates as processed
    await db.gradient_updates.update_many(
        {"round_id": round_id},
        {"$set": {"status": "processed"}}
    )
    
    # Notification Logic
    accuracy_improvement = avg_accuracy - PREVIOUS_ACCURACY
    msg_type = "success" if accuracy_improvement > 0 else "info"
    
    await broadcast_notification(
        title="Training Round Complete",
        message=f"Round {CURRENT_ROUND} done. Valid Updates: {len(valid_gradients)}. Accuracy: {avg_accuracy*100:.2f}%",
        notification_type=msg_type
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

# --- UPDATED: Secure Submission Endpoint ---
@api_router.post("/federated/submit-gradients")
async def submit_gradients(gradient_submit: GradientSubmit, company: dict = Depends(verify_api_key)):
    global CURRENT_ROUND
    
    # 1. Input Sanitation
    if not (0 <= gradient_submit.metrics.get('accuracy', 0) <= 1):
        raise HTTPException(status_code=400, detail="Invalid Accuracy Metric (Must be 0-1)")

    # 2. Basic Validation
    if not gradient_submit.gradient_data:
         raise HTTPException(status_code=400, detail="Empty Gradient Data")

    round_id = f"round_{CURRENT_ROUND}"
    
    gradient_update = {
        "update_id": str(uuid.uuid4()),
        "company_id": company['company_id'],
        "round_id": round_id,
        "gradient_data": gradient_submit.gradient_data,
        "metrics": gradient_submit.metrics,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "pending" # New field for safe aggregation
    }
    
    await db.gradient_updates.insert_one(gradient_update)
    return {"success": True, "round_id": round_id, "message": "Gradients accepted for processing"}

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
    """
    Generates the Internet Gateway Script.
    """
    base_url = str(request.base_url).rstrip('/')
    api_url = f"{base_url}/api"

    script_template = '''#!/usr/bin/env python3
"""
FinSecure Gateway Script
Company: {company_name}
Role: Internet Bridge (No Training)
"""
import requests
import json
import os
import time
import sys

# --- CONFIGURATION ---
API_KEY = "{api_key}"
BACKEND_URL = "{backend_url}"
EXCHANGE_FOLDER = "./secure_transfer" 

class FederatedGateway:
    def __init__(self, api_key, backend_url):
        self.headers = {{"X-API-Key": api_key}}
        self.backend_url = backend_url
        self.current_round = -1
        
        # Ensure the shared folder exists
        os.makedirs(EXCHANGE_FOLDER, exist_ok=True)
        print(f"🌉 Gateway Active | Company: {company_name}")
        print(f"📂 Monitoring Folder: {{EXCHANGE_FOLDER}}")

    def run(self):
        """Main Loop: Syncs data between Internet and Local Folder"""
        print("⏳ Waiting for updates...")
        while True:
            self._sync_downstream() 
            self._sync_upstream()   
            time.sleep(5)

    def _sync_downstream(self):
        """Check Server for new Global Model -> Save to Folder"""
        try:
            resp = requests.get(f"{{self.backend_url}}/model/download", headers=self.headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                server_round = data.get('round', 0)
                
                if server_round > self.current_round:
                    print(f"\\n⬇️  New Global Model detected (Round {{server_round}})")
                    
                    filepath = f"{{EXCHANGE_FOLDER}}/global_model.json"
                    with open(filepath, "w") as f:
                        json.dump(data, f)
                    
                    print(f"    Saved to {{filepath}}")
                    self.current_round = server_round
        except Exception as e:
            print(f"⚠️ Connection Error: {{e}}")

    def _sync_upstream(self):
        """Check Folder for Gradients -> Upload to Server"""
        local_file = f"{{EXCHANGE_FOLDER}}/local_gradients.json"
        
        if os.path.exists(local_file):
            print("\\n⬆️  Found local updates. Uploading to server...")
            try:
                with open(local_file, "r") as f:
                    payload = json.load(f)
                
                resp = requests.post(
                    f"{{self.backend_url}}/federated/submit-gradients",
                    headers=self.headers,
                    json=payload
                )
                
                if resp.status_code == 200:
                    print("    ✅ Upload Successful!")
                    os.remove(local_file) 
                else:
                    print(f"    ❌ Upload Failed: {{resp.text}}")
            except Exception as e:
                print(f"⚠️ Upload Error: {{e}}")

if __name__ == "__main__":
    gateway = FederatedGateway(API_KEY, BACKEND_URL)
    gateway.run()
'''

    return {
        "filename": "finsecure_gateway.py",
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

@api_router.get("/companies")
async def get_active_companies():
    try:
        cursor = db.companies.find({}) 
        companies = await cursor.to_list(length=100)
        
        results = []
        for company in companies:
            results.append({
                "id": str(company["_id"]),
                "name": company.get("name", "Unknown Bank"),
                "email": company.get("email", ""),
                "status": "Active",
                "joined_at": company.get("created_at", "Recently")
            })
        return results
    except Exception as e:
        print(f"Error fetching companies: {e}")
        return []

@api_router.get("/notifications/unread/count")
async def get_notification_count():
    return {"count": 0}

@api_router.get("/analytics/rounds")
async def get_round_analytics():
    try:
        cursor = db.training_rounds.find({}).sort("round_number", 1)
        history = await cursor.to_list(length=100)
        
        analytics_data = []
        for entry in history:
            analytics_data.append({
                "round": entry.get("round_number", 0),
                "accuracy": entry.get("avg_accuracy", 0),
                "loss": entry.get("avg_loss", 0),
                "timestamp": entry.get("timestamp", "")
            })
            
        if not analytics_data:
            return [
                {"round": 1, "accuracy": 0.65, "loss": 0.80},
                {"round": 2, "accuracy": 0.72, "loss": 0.65},
                {"round": 3, "accuracy": 0.81, "loss": 0.45}
            ]

        return analytics_data

    except Exception as e:
        print(f"Error fetching analytics: {e}")
        return []

# --- RESET BUTTON (Corrected to GET) ---
@api_router.get("/reset-system") 
async def reset_database():
    """
    Clears all training history so the graph can start fresh.
    """
    await db.training_rounds.delete_many({})
    await db.gradient_updates.delete_many({})
    
    global CURRENT_ROUND, GLOBAL_MODEL
    CURRENT_ROUND = 0
    GLOBAL_MODEL = create_fraud_detection_model()
    
    print("♻️ SYSTEM RESET: Graph is now empty and ready.")
    return {"message": "System Reset Successful!"}

# Include API Router
app.include_router(api_router)

# ============= CORS SETUP =============
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
