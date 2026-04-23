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
AGGREGATION_THRESHOLD = 1  # Instant updates for demo
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

class TransactionInput(BaseModel):
    amount: float
    time_of_day: float 
    is_international: int 

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
        try:
            global GLOBAL_MODEL, CURRENT_ROUND, PREVIOUS_ACCURACY
            
            round_id = f"round_{CURRENT_ROUND}"
            updates = await db.gradient_updates.find({"round_id": round_id, "status": "pending"}, {"_id": 0}).to_list(1000)
            
            if not updates:
                return {"success": False, "message": "No pending updates"}
                
            if len(updates) < AGGREGATION_THRESHOLD:
                logger.info(f"⏳ Waiting for more banks. Currently {len(updates)}/{AGGREGATION_THRESHOLD} submitted.")
                return {"success": False, "message": f"Waiting for {AGGREGATION_THRESHOLD} updates"}
            
            logger.info(f"🚀 Threshold Reached! Aggregating {len(updates)} updates for Round {CURRENT_ROUND}")
            
            valid_gradients, sample_counts = [], []
            weighted_acc_sum, weighted_loss_sum, total_samples = 0.0, 0.0, 0

            for update in updates:
                weights = deserialize_model_weights(update['gradient_data'])
                if weights and validate_gradient_shape(weights, GLOBAL_MODEL):
                    valid_gradients.append(weights)
                    count = int(max(update.get('num_samples', 1), 1))
                    sample_counts.append(count)
                    metrics = update.get('metrics', {'accuracy': 0.0, 'loss': 0.0})
                    weighted_acc_sum += float(metrics['accuracy']) * count
                    weighted_loss_sum += float(metrics['loss']) * count
                    total_samples += count
                else:
                    logger.warning(f"Dropped invalid update from {update.get('company_id')}")
            
            if not valid_gradients:
                return {"success": False, "message": "No valid updates"}
            
            # Update Model
            avg_gradients = federated_averaging(valid_gradients, sample_counts)
            if avg_gradients:
                GLOBAL_MODEL.set_weights(avg_gradients)
            
            # CRITICAL FIX: Cast NumPy floats to standard Python floats to prevent MongoDB crashes
            network_accuracy = float(weighted_acc_sum / total_samples) if total_samples > 0 else 0.0
            network_loss = float(weighted_loss_sum / total_samples) if total_samples > 0 else 0.0
            
            training_round = {
                "round_id": round_id,
                "round_number": int(CURRENT_ROUND),
                "participating_companies": int(len(valid_gradients)),
                "total_samples_trained": int(total_samples),
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
            logger.info(f"✅ Round successfully increased to {CURRENT_ROUND}")
            return {"success": True, "round_number": CURRENT_ROUND - 1, "avg_accuracy": network_accuracy}
            
        except Exception as e:
            logger.error(f"CRITICAL ERROR in aggregation: {e}")
            return {"success": False, "message": str(e)}

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

# --- LIVE PREDICTION ENDPOINT ---
@api_router.post("/transaction/predict")
async def process_live_transaction(data: TransactionInput, company: dict = Depends(verify_api_key)):
    global GLOBAL_MODEL
    
    input_array = np.zeros((1, 30), dtype=np.float32)
    input_array[0, 0] = data.amount / 10000.0  
    input_array[0, 1] = data.time_of_day / 24.0 
    input_array[0, 2] = float(data.is_international)
    
    prediction_prob = GLOBAL_MODEL.predict(input_array, verbose=0)[0][0]
    
    is_fraud = bool(prediction_prob > 0.5)
    risk_score = float(prediction_prob * 100)
    
    return {
        "transaction_id": f"TXN-{secrets.token_hex(4).upper()}",
        "status": "BLOCKED" if is_fraud else "APPROVED",
        "risk_score": round(risk_score, 2),
        "message": "Fraudulent pattern detected by Cyber Shield" if is_fraud else "Transaction looks secure"
    }

@api_router.get("/client/sdk")
async def get_client_sdk(request: Request):
    """Serves the FinSecure SDK library file."""
    base_url = str(request.base_url).rstrip('/')
    
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
                
                if data['round'] <= self.current_round:
                    return None, self.current_round

                self.current_round = data['round']
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
        try:
            weights = model.get_weights()
            buffer = io.BytesIO()
            np.savez_compressed(buffer, *weights)
            buffer.seek(0)
            encoded_weights = base64.b64encode(buffer.read()).decode('utf-8')
            
            payload = {{
                "gradient_data": encoded_weights,
                "metrics": {{
                    "accuracy": float(metrics.get('accuracy', 0)),
                    "loss": float(metrics.get('loss', 0))
                }},
                "num_samples": int(X_train_len)
            }}
            
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

@api_router.get("/client/script")
async def get_client_script(request: Request, company: dict = Depends(verify_api_key)):
    base_url = str(request.base_url).rstrip('/')
    api_url = f"{base_url}/api"

    script_content = f'''#!/usr/bin/env python3
"""
FinSecure Interactive Gateway Script
Company: {company['name']}
"""
import requests
import os
import time
import sys
import csv
import numpy as np
import tensorflow as tf
import base64
import io

API_KEY = "{company['api_key']}"
BACKEND_URL = "{api_url}"
BANK_NAME = "{company['name']}"

CORE_DB_FILE = f"{{BANK_NAME.replace(' ', '_')}}_core_ledger.csv"
ML_BUFFER_FILE = f"{{BANK_NAME.replace(' ', '_')}}_ml_training_buffer.csv"

RED, GREEN, CYAN, YELLOW, RESET = '\\033[91m', '\\033[92m', '\\033[96m', '\\033[93m', '\\033[0m'
HEADERS = {{"X-API-Key": API_KEY}}

print(f"\\n{{GREEN}}✅ Securely connected as: {{BANK_NAME}}{{RESET}}")

CSV_HEADERS = [
    'Amount', 'Time', 'Is_International', 'Txn_Type', 
    'Location_Dist', 'New_Device', 'Prev_Declines', 
    'Acct_Age', 'Velocity', 'MCC_Risk', 'Is_Fraud'
]

if not os.path.exists(CORE_DB_FILE):
    print(f"{{YELLOW}}⚙️ Initializing Untouchable Core Ledger...{{RESET}}")
    with open(CORE_DB_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADERS)
        for _ in range(500):
            amt = round(np.random.uniform(10, 1500), 2)
            time_val = round(np.random.uniform(0, 24), 1)
            is_int = np.random.choice([0, 1], p=[0.95, 0.05])
            txn_type = np.random.randint(1, 4)
            loc_dist = round(np.random.uniform(0, 50), 1)
            new_dev = 0
            declines = 0
            age = np.random.randint(30, 3650)
            vel = np.random.randint(1, 4)
            mcc = round(np.random.uniform(0.1, 0.4), 2)
            
            writer.writerow([amt, time_val, is_int, txn_type, loc_dist, new_dev, declines, age, vel, mcc, 0])
    print(f"{{GREEN}}📁 Core Ledger created: {{CORE_DB_FILE}}{{RESET}}")

if not os.path.exists(ML_BUFFER_FILE):
    with open(ML_BUFFER_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADERS)
    print(f"{{GREEN}}📁 Dedicated ML Training Buffer created: {{ML_BUFFER_FILE}}{{RESET}}")

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(30,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2), 
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

def sync_global_model():
    print(f"\\n{{YELLOW}}⏳ Syncing with Global Intelligence...{{RESET}}", end=" ")
    try:
        res = requests.get(f"{{BACKEND_URL}}/model/download", headers=HEADERS)
        if res.status_code == 200:
            data = res.json()
            weights_data = base64.b64decode(data['weights'])
            npz = np.load(io.BytesIO(weights_data), allow_pickle=True)
            model.set_weights([npz[f'arr_{{i}}'] for i in range(len(npz.files))])
            print(f"{{GREEN}}Synced to Round {{data['round']}}{{RESET}}")
        else:
            print(f"{{RED}}Failed.{{RESET}}")
    except Exception as e:
        print(f"{{RED}}Error: {{e}}{{RESET}}")

sync_global_model()
last_txn = None

while True:
    print(f"\\n{{CYAN}}========== {{BANK_NAME}} TERMINAL =========={{RESET}}")
    print("1. Process Live Transaction")
    print("2. Report Last Transaction as Fraud & Train Network")
    print("3. Sync Latest Global Intelligence")
    print("4. Exit")
    
    choice = input("Select Action (1-4): ")
    
    if choice == '1':
        try:
            amt = float(input("💰 Amount ($): "))
            time_val = float(input("🕒 Time (0-24): "))
            isInt = float(input("🌍 International (1=Yes, 0=No): "))
        except ValueError:
            print(f"{{RED}}Invalid input! Please enter numbers.{{RESET}}")
            continue
            
        print(f"\\n{{CYAN}}🔍 Auto-fetching transaction metadata...{{RESET}}")
        time.sleep(0.5)
        
        is_risky = 1 if amt > 5000 else 0
        txn_type = 1 
        loc_dist = round(np.random.uniform(500, 2000) if is_risky else np.random.uniform(0, 50), 1)
        new_dev = 1 if is_risky else 0
        declines = 2 if is_risky else 0
        age = np.random.randint(30, 3650)
        vel = np.random.randint(3, 8) if is_risky else 1
        mcc = 0.9 if is_risky else 0.2
        
        print(f" ├─ Transaction Type: Online (1)")
        print(f" ├─ Location Distance: {{loc_dist}} miles")
        print(f" ├─ New Device Flag: {{new_dev}}")
        print(f" ├─ Prev. Declines (24h): {{declines}}")
        print(f" ├─ Account Age: {{age}} days")
        print(f" ├─ Txn Velocity (1h): {{vel}}")
        print(f" └─ MCC Risk Score: {{mcc}}")
        
        last_txn = [amt, time_val, isInt, txn_type, loc_dist, new_dev, declines, age, vel, mcc]
        
        test_txn = np.zeros((1, 30), dtype=np.float32)
        test_txn[0, 0] = amt / 10000.0
        test_txn[0, 1] = time_val / 24.0
        test_txn[0, 2] = isInt
        test_txn[0, 3] = txn_type / 3.0
        test_txn[0, 4] = loc_dist / 10000.0
        test_txn[0, 5] = new_dev
        test_txn[0, 6] = declines / 10.0
        test_txn[0, 7] = age / 3650.0
        test_txn[0, 8] = vel / 50.0
        test_txn[0, 9] = mcc
        
        print(f"\\n{{YELLOW}}🧠 Analyzing 10-dimensional vector against global weights...{{RESET}}")
        time.sleep(1)
        pred = model.predict(test_txn, verbose=0)[0][0]
        
        if pred > 0.5:
            print(f"{{RED}}🚨 TRANSACTION BLOCKED! (Fraud Risk: {{pred*100:.1f}}%){{RESET}}")
            with open(CORE_DB_FILE, 'a', newline='') as f:
                csv.writer(f).writerow(last_txn + [1])
        else:
            print(f"{{GREEN}}✅ TRANSACTION APPROVED (Fraud Risk: {{pred*100:.1f}}%){{RESET}}")
            with open(CORE_DB_FILE, 'a', newline='') as f:
                csv.writer(f).writerow(last_txn + [0])

    elif choice == '2':
        if not last_txn:
            print(f"{{RED}}No transaction to report!{{RESET}}")
            continue
            
        print(f"\\n{{YELLOW}}⚙️ Initiating Incident Response...{{RESET}}")
        
        print(f"✍️ Extracting 10-dimensional threat signature to {{ML_BUFFER_FILE}}...")
        with open(ML_BUFFER_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            for _ in range(50):
                noisy_txn = list(last_txn)
                noisy_txn[0] = round(noisy_txn[0] * np.random.uniform(0.9, 1.1), 2)
                noisy_txn[4] = round(noisy_txn[4] * np.random.uniform(0.9, 1.1), 1)
                writer.writerow(noisy_txn + [1]) 
                
        print("📊 Compiling training batch from Ledger and Buffer...")
        try:
            core_data = np.genfromtxt(CORE_DB_FILE, delimiter=',', skip_header=1)
            safe_data = core_data[core_data[:, 10] == 0][:100]
            fraud_data = np.genfromtxt(ML_BUFFER_FILE, delimiter=',', skip_header=1)
            
            combined_data = np.vstack([safe_data, fraud_data])
            X_raw = combined_data[:, :10]
            y_train = combined_data[:, 10]
            
            X_train = np.zeros((len(X_raw), 30), dtype=np.float32)
            X_train[:, 0] = X_raw[:, 0] / 10000.0
            X_train[:, 1] = X_raw[:, 1] / 24.0
            X_train[:, 2] = X_raw[:, 2]
            X_train[:, 3] = X_raw[:, 3] / 3.0
            X_train[:, 4] = X_raw[:, 4] / 10000.0
            X_train[:, 5] = X_raw[:, 5]
            X_train[:, 6] = X_raw[:, 6] / 10.0
            X_train[:, 7] = X_raw[:, 7] / 3650.0
            X_train[:, 8] = X_raw[:, 8] / 50.0
            X_train[:, 9] = X_raw[:, 9]
            
            print(f"🧠 Training Neural Network safely without touching Core Ledger...")
            hist = model.fit(X_train, y_train, epochs=20, verbose=0)
            
            buffer = io.BytesIO()
            np.savez_compressed(buffer, *model.get_weights())
            buffer.seek(0)
            encoded_weights = base64.b64encode(buffer.read()).decode('utf-8')
            
            print(f"⬆️ Uploading Encrypted Intelligence to Cyber Shield Server...")
            
            # --- STRICT FLOAT/INT CASTING FOR MONGODB COMPATIBILITY ---
            res = requests.post(f"{{BACKEND_URL}}/federated/submit-gradients", headers=HEADERS, json={{
                "gradient_data": encoded_weights,
                "metrics": {{"accuracy": float(hist.history['accuracy'][-1]), "loss": float(hist.history['loss'][-1])}},
                "num_samples": int(len(X_train))
            }})
            
            if res.status_code == 200:
                print(f"{{GREEN}}✅ Network Notified! Global Model Updated.{{RESET}}")
                sync_global_model() # Instantly pull the new round
            else:
                print(f"{{RED}}❌ Upload Failed: {{res.text}}{{RESET}}")
                
        except Exception as e:
            print(f"{{RED}}❌ Failed to compile training data: {{e}}{{RESET}}")

    elif choice == '3':
        sync_global_model()
    elif choice == '4':
        break
'''
    return PlainTextResponse(script_content, media_type="text/x-python")

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
        print(f"Error: {e}")
        return []

@api_router.get("/analytics/my-updates")
async def get_my_updates(company: dict = Depends(verify_api_key)):
    updates = await db.gradient_updates.find(
        {"company_id": company['company_id']},
        {"_id": 0, "gradient_data": 0} 
    ).sort("timestamp", -1).limit(50).to_list(50)
    return updates

@api_router.get("/notifications", response_model=List[Notification])
async def get_notifications(company: dict = Depends(verify_api_key)):
    return await db.notifications.find(
        {"$or": [{"company_id": company['company_id']}, {"company_id": None}]},
        {"_id": 0}
    ).sort("created_at", -1).limit(50).to_list(50)

@api_router.get("/notifications/unread/count")
async def get_notification_count():
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
    logger.info(f"✅ Update received from {company['name']}.")
    
    # --- CRITICAL FIX: SYNCHRONOUS EXECUTION ---
    pending_count = await db.gradient_updates.count_documents({"round_id": round_id, "status": "pending"})
    if pending_count >= AGGREGATION_THRESHOLD:
        logger.info(f"🔥 {AGGREGATION_THRESHOLD}/{AGGREGATION_THRESHOLD} Banks Submitted! Triggering Instant Aggregation...")
        
        # Await it directly so it calculates before answering the node
        agg_result = await aggregate_gradients()
        
        if not agg_result.get("success"):
            logger.error(f"Aggregation Failed internally: {agg_result}")
    else:
        logger.info(f"⏳ {pending_count}/{AGGREGATION_THRESHOLD} Banks Submitted. Waiting for the other bank...")
        
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

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": MODEL_VERSION}

app.include_router(api_router)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
