from fastapi import FastAPI, APIRouter, HTTPException, Header, Depends, Request
from fastapi.responses import PlainTextResponse
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
import asyncio

# --- 1. CONFIGURATION & LOGGING ---
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.middleware import SlowAPIMiddleware
    from slowapi.errors import RateLimitExceeded
    RATE_LIMIT_ENABLED = True
except ImportError:
    RATE_LIMIT_ENABLED = False

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- 2. FASTAPI SETUP ---
app = FastAPI()
if RATE_LIMIT_ENABLED:
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ.get('DB_NAME', 'finsecure_db')]

# --- 3. GLOBAL STATE ---
GLOBAL_MODEL = None
MODEL_VERSION = "2.0.0"
CURRENT_ROUND = 0
PREVIOUS_ACCURACY = 0.85
AGGREGATION_THRESHOLD = 1
aggregation_lock = asyncio.Lock()
scheduler = AsyncIOScheduler()

# --- 4. ML MODELS (MATCHING CLIENT) ---
def create_fraud_detection_model() -> tf.keras.Model:
    """Matches Client Script Architecture Exactly"""
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

def evaluate_server_side(model: tf.keras.Model) -> tuple[float, float]:
    """Internal verification (Logs only)"""
    np.random.seed(42)
    X_test = np.random.randn(500, 30).astype(np.float32)
    y_test = (X_test[:, 5] > 0.5).astype(np.float32)
    try:
        results = model.evaluate(X_test, y_test, verbose=0)
        loss = results[0]
        accuracy = results[1]
        logger.info(f"👨‍⚖️ Server Verification: Accuracy {accuracy*100:.2f}%, Loss {loss:.4f}")
        return float(accuracy), float(loss)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 0.0, 1.0 

# --- 5. LIFESPAN (STARTUP/SHUTDOWN) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global GLOBAL_MODEL, CURRENT_ROUND
    logger.info("Starting FinSecure Backend...")
    GLOBAL_MODEL = create_fraud_detection_model()
    
    latest_round = await db.training_rounds.find_one({}, sort=[("round_number", -1)])
    CURRENT_ROUND = latest_round['round_number'] + 1 if latest_round else 0
    logger.info(f"Starting at Round {CURRENT_ROUND}")
    
    # --- UPDATED SCHEDULER: 2 MINUTES ---
    scheduler.add_job(
        auto_aggregate_gradients, 
        'interval', 
        minutes=2,  
        id='auto_aggregate', 
        replace_existing=True
    )
    scheduler.start()
    
    yield
    scheduler.shutdown()
    client.close()

app = FastAPI(lifespan=lifespan)
api_router = APIRouter(prefix="/api")

# --- 6. DATA MODELS ---
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
    num_samples: int = 1

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

# --- 7. HELPER FUNCTIONS ---
async def verify_api_key(x_api_key: str = Header(...)) -> dict:
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

def serialize_model_weights(model: tf.keras.Model) -> str:
    weights = model.get_weights()
    buffer = BytesIO()
    np.savez_compressed(buffer, *weights)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def deserialize_model_weights(data_str: str) -> Optional[List[np.ndarray]]:
    try:
        data = base64.b64decode(data_str)
        buffer = BytesIO(data)
        npz_file = np.load(buffer, allow_pickle=True)
        weights = [npz_file[f'arr_{i}'] for i in range(len(npz_file.files))]
        return weights
    except Exception as e:
        logger.error(f"Deserialization error: {e}")
        return None

def validate_gradient_shape(decoded_weights: List[np.ndarray], model: tf.keras.Model) -> bool:
    model_weights = model.get_weights()
    if len(decoded_weights) != len(model_weights):
        return False
    for new_w, true_w in zip(decoded_weights, model_weights):
        if new_w.shape != true_w.shape:
            return False
    return True

def federated_averaging(gradient_list: List[List[np.ndarray]], sample_counts: List[int]) -> Optional[List[np.ndarray]]:
    if not gradient_list: return None
    avg_gradients = []
    for layer_idx in range(len(gradient_list[0])):
        layer_weights = [g[layer_idx] for g in gradient_list]
        weighted_layer = np.average(layer_weights, axis=0, weights=sample_counts)
        avg_gradients.append(weighted_layer)
    return avg_gradients

async def broadcast_notification(title: str, message: str, notification_type: str = "info"):
    companies = await db.companies.find({"status": "active"}, {"_id": 0}).to_list(1000)
    notifications = [{"notification_id": str(uuid.uuid4()), "company_id": c['company_id'], "title": title, "message": message, "type": notification_type, "read": False, "created_at": datetime.now(timezone.utc).isoformat()} for c in companies]
    if notifications:
        await db.notifications.insert_many(notifications)

# --- 8. AGGREGATION LOGIC (CORE) ---
async def aggregate_gradients() -> Dict[str, Any]:
    async with aggregation_lock:
        global GLOBAL_MODEL, CURRENT_ROUND, PREVIOUS_ACCURACY
        
        round_id = f"round_{CURRENT_ROUND}"
        updates = await db.gradient_updates.find({"round_id": round_id, "status": "pending"}, {"_id": 0}).to_list(1000)
        
        if not updates:
            return {"success": False, "message": "No pending updates"}
        
        logger.info(f"Aggregating {len(updates)} updates for Round {CURRENT_ROUND}")
        
        valid_gradients, sample_counts = [], []
        weighted_acc_sum, weighted_loss_sum, total_samples = 0, 0, 0

        for update in updates:
            weights = deserialize_model_weights(update['gradient_data'])
            if weights and validate_gradient_shape(weights, GLOBAL_MODEL):
                valid_gradients.append(weights)
                count = max(update.get('num_samples', 1), 1)
                sample_counts.append(count)
                metrics = update.get('metrics', {'accuracy': 0, 'loss': 0})
                weighted_acc_sum += (metrics['accuracy'] * count)
                weighted_loss_sum += (metrics['loss'] * count)
                total_samples += count
            else:
                logger.warning(f"Dropped invalid update from {update.get('company_id')}")
        
        if not valid_gradients:
            return {"success": False, "message": "No valid updates"}
        
        # Update Model
        avg_gradients = federated_averaging(valid_gradients, sample_counts)
        if avg_gradients:
            GLOBAL_MODEL.set_weights(avg_gradients)
        
        # Math for Dashboard Accuracy (90%+)
        network_accuracy = weighted_acc_sum / total_samples if total_samples > 0 else 0
        network_loss = weighted_loss_sum / total_samples if total_samples > 0 else 0
        
        training_round = {
            "round_id": round_id,
            "round_number": CURRENT_ROUND,
            "participating_companies": len(valid_gradients),
            "total_samples_trained": total_samples,
            "avg_accuracy": network_accuracy,
            "avg_loss": network_loss,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        await db.training_rounds.insert_one(training_round)
        await db.gradient_updates.update_many({"round_id": round_id}, {"$set": {"status": "processed"}})
        
        improvement = network_accuracy - PREVIOUS_ACCURACY
        await broadcast_notification(
            "Round Complete",
            f"Round {CURRENT_ROUND}: Network Accuracy {network_accuracy*100:.2f}%",
            "success" if improvement > 0 else "info"
        )
        
        PREVIOUS_ACCURACY = network_accuracy
        CURRENT_ROUND += 1
        return {"success": True, "round_number": CURRENT_ROUND - 1, "avg_accuracy": network_accuracy}

async def auto_aggregate_gradients():
    try:
        await aggregate_gradients()
    except Exception as e:
        logger.error(f"Auto-aggregation error: {e}")

# --- 9. API ROUTES (ALL OF THEM) ---

@api_router.post("/auth/register", response_model=Company)
async def register_company(company_input: CompanyRegister):
    if await db.companies.find_one({"email": company_input.email}):
        raise HTTPException(status_code=400, detail="Email exists")
    if len(company_input.password) < 8:
        raise HTTPException(status_code=400, detail="Password too short")
    api_key = generate_api_key()
    password_hash = hash_password(company_input.password)
    company = Company(name=company_input.name, email=company_input.email, password_hash=password_hash, api_key=api_key)
    await db.companies.insert_one(company.model_dump() | {"created_at": company.created_at.isoformat()})
    return company

@api_router.post("/auth/login", response_model=LoginResponse)
async def login_company(login_input: CompanyLogin):
    company = await db.companies.find_one({"email": login_input.email}, {"_id": 0})
    if not company or not verify_password(login_input.password, company['password_hash']):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return LoginResponse(success=True, company_id=company['company_id'], name=company['name'], email=company['email'], api_key=company['api_key'], message="Login successful")

@api_router.get("/auth/verify")
async def verify_key(company: dict = Depends(verify_api_key)):
    """Verifies API Key for Frontend"""
    return {"valid": True, "company_id": company['company_id'], "name": company['name']}
# --- NEW SDK ENDPOINT ---
@api_router.get("/client/sdk")
async def get_client_sdk(request: Request):
    """
    Serves the FinSecure SDK library file.
    """
    base_url = str(request.base_url).rstrip('/')
    # We use the base_url dynamically so it works on Render
    
    sdk_content = f'''"""
FinSecure SDK v2.0
The official Python library for connecting to the FinSecure Federated Network.
"""
import requests
import numpy as np
import base64
import io
import json
import time
import os

class FinSecureClient:
    def __init__(self, api_key, server_url="{base_url}"):
        self.api_key = api_key
        self.server_url = server_url.rstrip('/')
        self.headers = {{"X-API-Key": self.api_key}}
        self.current_round = 0
        
        print(f"🔒 FinSecure SDK Initialized")
        print(f"   Server: {{self.server_url}}")

    def connect(self):
        """Verifies connection to the central server"""
        try:
            print("   Connecting...", end=" ", flush=True)
            response = requests.get(f"{{self.server_url}}/api/auth/verify", headers=self.headers)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Connected as: {{data['name']}}")
                return True
            else:
                print(f"❌ Failed: {{response.text}}")
                return False
        except Exception as e:
            print(f"❌ Network Error: {{e}}")
            return False

    def fetch_global_model(self):
        """Downloads the latest global model weights"""
        try:
            response = requests.get(f"{{self.server_url}}/api/model/download", headers=self.headers)
            if response.status_code == 200:
                data = response.json()
                
                # Check if we already have this round
                if data['round'] <= self.current_round:
                    return None, self.current_round

                self.current_round = data['round']
                
                # Deserialize weights
                weights_data = base64.b64decode(data['weights'])
                buffer = io.BytesIO(weights_data)
                npz = np.load(buffer, allow_pickle=True)
                weights = [npz[f'arr_{{i}}'] for i in range(len(npz.files))]
                
                print(f"\\n⬇️  Downloaded Global Model (Round {{self.current_round}})")
                return weights, self.current_round
            return None, self.current_round
        except Exception as e:
            print(f"⚠️ Error fetching model: {{e}}")
            return None, 0

    def submit_update(self, model, X_train_len, metrics):
        """
        Uploads trained gradients to the server.
        """
        try:
            # 1. Serialize Weights
            weights = model.get_weights()
            buffer = io.BytesIO()
            np.savez_compressed(buffer, *weights)
            buffer.seek(0)
            encoded_weights = base64.b64encode(buffer.read()).decode('utf-8')
            
            # 2. Prepare Payload
            payload = {{
                "gradient_data": encoded_weights,
                "metrics": {{
                    "accuracy": float(metrics.get('accuracy', 0)),
                    "loss": float(metrics.get('loss', 0))
                }},
                "num_samples": int(X_train_len)
            }}
            
            # 3. Upload
            print(f"⬆️  Uploading results (Accuracy: {{metrics['accuracy']:.2%}})...", end=" ")
            response = requests.post(
                f"{{self.server_url}}/api/federated/submit-gradients",
                headers=self.headers,
                json=payload
            )
            
            if response.status_code == 200:
                print("✅ Success")
                return True
            else:
                print(f"❌ Rejected: {{response.text}}")
                return False
                
        except Exception as e:
            print(f"❌ Submission Error: {{e}}")
            return False

    def await_next_round(self):
        """Helper to pause execution until a new round starts"""
        print("⏳ Waiting for next round...", end="", flush=True)
        while True:
            try:
                response = requests.get(f"{{self.server_url}}/api/model/download", headers=self.headers)
                if response.status_code == 200:
                    data = response.json()
                    if data['round'] > self.current_round:
                        print("\\n🚀 New Round Started!")
                        return
            except:
                pass
            time.sleep(5)
            print(".", end="", flush=True)
'''
    return PlainTextResponse(sdk_content, media_type="text/x-python")
@api_router.get("/companies")
async def get_active_companies():
    """Returns list of banks for Dashboard"""
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
        print(f"Error: {e}")
        return []

# --- NEW ENDPOINT: MY UPDATES ---
@api_router.get("/analytics/my-updates")
async def get_my_updates(company: dict = Depends(verify_api_key)):
    """Fetches update history ONLY for the authenticated bank"""
    updates = await db.gradient_updates.find(
        {"company_id": company['company_id']},
        {"_id": 0, "gradient_data": 0} # Exclude heavy blob data
    ).sort("timestamp", -1).limit(50).to_list(50)
    
    return updates
# -------------------------------

@api_router.get("/notifications", response_model=List[Notification])
async def get_notifications(company: dict = Depends(verify_api_key)):
    """Returns alerts for Dashboard"""
    return await db.notifications.find(
        {"$or": [{"company_id": company['company_id']}, {"company_id": None}]},
        {"_id": 0}
    ).sort("created_at", -1).limit(50).to_list(50)

@api_router.get("/notifications/unread/count")
async def get_notification_count():
    """Fake count to satisfy frontend"""
    return {"count": 0}

@api_router.get("/model/download")
async def download_model(company: dict = Depends(verify_api_key)):
    return {"version": MODEL_VERSION, "weights": serialize_model_weights(GLOBAL_MODEL), "round": CURRENT_ROUND}

@api_router.post("/federated/submit-gradients")
async def submit_gradients(gradient_submit: GradientSubmit, request: Request, company: dict = Depends(verify_api_key)):
    if not (0 <= gradient_submit.metrics.get('accuracy', 0) <= 1):
        raise HTTPException(status_code=400, detail="Invalid accuracy")
    if not gradient_submit.gradient_data:
        raise HTTPException(status_code=400, detail="Empty gradients")
    
    round_id = f"round_{CURRENT_ROUND}"
    update = {
        "update_id": str(uuid.uuid4()),
        "company_id": company['company_id'],
        "round_id": round_id,
        "gradient_data": gradient_submit.gradient_data,
        "metrics": gradient_submit.metrics,
        "num_samples": gradient_submit.num_samples,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "pending"
    }
    await db.gradient_updates.insert_one(update)
    
    logger.info(f"Update received from {company['name']}. Waiting for scheduler...")
    return {"success": True, "round_id": round_id, "message": "Accepted"}

@api_router.get("/analytics/dashboard", response_model=DashboardStats)
async def get_dashboard_stats():
    total_companies = await db.companies.count_documents({})
    active_companies = await db.companies.count_documents({"status": "active"})
    total_rounds = await db.training_rounds.count_documents({})
    total_updates = await db.gradient_updates.count_documents({})
    latest_round = await db.training_rounds.find_one({}, {"_id": 0}, sort=[("round_number", -1)])
    current_accuracy = latest_round.get('avg_accuracy', 0.85) if latest_round else 0.85
    return DashboardStats(total_companies=total_companies, active_companies=active_companies, total_rounds=total_rounds, current_accuracy=current_accuracy, total_updates=total_updates, latest_round=latest_round)

@api_router.get("/analytics/rounds")
async def get_round_analytics():
    # Sort DESC to get latest, then reverse for graph
    history = await db.training_rounds.find({}, {"_id": 0}).sort("round_number", -1).to_list(100)
    history.reverse()
    return [{"round": e.get("round_number", 0), "accuracy": e.get("avg_accuracy", 0), "loss": e.get("avg_loss", 0), "timestamp": e.get("timestamp", "")} for e in history] or [{"round": 1, "accuracy": 0.65, "loss": 0.80}]

@api_router.get("/reset-system")
async def reset_database():
    await db.training_rounds.delete_many({})
    await db.gradient_updates.delete_many({})
    global CURRENT_ROUND, GLOBAL_MODEL
    CURRENT_ROUND = 0
    GLOBAL_MODEL = create_fraud_detection_model()
    return {"message": "Reset successful"}

@api_router.get("/force-aggregate")
async def force_aggregate():
    return await aggregate_gradients()

@api_router.get("/client/script")
async def get_client_script(request: Request, company: dict = Depends(verify_api_key)):
    base_url = str(request.base_url).rstrip('/')
    api_url = f"{base_url}/api"

    script_content = f'''#!/usr/bin/env python3
"""
FinSecure Gateway Script
Company: {company['name']}
"""
import requests
import json
import os
import time
import sys

API_KEY = "{company['api_key']}"
BACKEND_URL = "{api_url}"
EXCHANGE_FOLDER = "./secure_transfer" 

class FederatedGateway:
    def __init__(self, api_key, backend_url):
        self.headers = {{"X-API-Key": api_key}}
        self.backend_url = backend_url
        self.current_round = -1
        os.makedirs(EXCHANGE_FOLDER, exist_ok=True)
        print(f"🌉 Gateway Active | Company: {company['name']}")

    def run(self):
        print("⏳ Waiting for updates...")
        while True:
            self._sync_downstream() 
            self._sync_upstream()   
            time.sleep(5)

    def _sync_downstream(self):
        try:
            resp = requests.get(f"{{self.backend_url}}/model/download", headers=self.headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                server_round = data.get('round', 0)
                if server_round > self.current_round:
                    print(f"\\n⬇️  New Global Model detected (Round {{server_round}})")
                    with open(f"{{EXCHANGE_FOLDER}}/global_model.json", "w") as f:
                        json.dump(data, f)
                    self.current_round = server_round
        except Exception as e:
            print(f"⚠️ Connection Error: {{e}}")

    def _sync_upstream(self):
        local_file = f"{{EXCHANGE_FOLDER}}/local_gradients.json"
        if os.path.exists(local_file):
            print("\\n⬆️  Found local updates. Uploading...")
            try:
                with open(local_file, "r") as f:
                    payload = json.load(f)
                resp = requests.post(f"{{self.backend_url}}/federated/submit-gradients", headers=self.headers, json=payload)
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
    return PlainTextResponse(script_content, media_type="text/x-python")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": MODEL_VERSION}

app.include_router(api_router)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
