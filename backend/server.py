from fastapi import FastAPI, APIRouter, HTTPException, Header, Depends, Request, BackgroundTasks
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
from datetime import datetime, timezone, timedelta
import secrets
import numpy as np
import tensorflow as tf
from io import BytesIO
import base64
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import bcrypt
from contextlib import asynccontextmanager
import asyncio
import smtplib
from email.mime.text import MIMEText

# --- 1. CONFIGURATION, LOGGING & RATE LIMITING ---
# We keep this verbose to help you debug on Vercel/Render
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
logger = logging.getLogger("FinSecure_Global_Server")

# --- ENVIRONMENT VARIABLES ---
EMAIL_ADDRESS = os.environ.get("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
MONGO_URL = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
DB_NAME = os.environ.get('DB_NAME', 'finsecure_db')

# --- 2. DATABASE & STATE INITIALIZATION ---
client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]

GLOBAL_MODEL = None
MODEL_VERSION = "2.0.0"
CURRENT_ROUND = 0
PREVIOUS_ACCURACY = 0.85
AGGREGATION_THRESHOLD = 1 # Minimum nodes required to aggregate
aggregation_lock = asyncio.Lock()
scheduler = AsyncIOScheduler()

# --- 3. DEEP LEARNING ARCHITECTURE ---
# This must match the Client Edge Node exactly.
def create_fraud_detection_model() -> tf.keras.Model:
    """
    Constructs the standard 6-layer DNN for Cyber Shield.
    Architecture: 30 (Input) -> 64 -> 32 -> 16 -> 1 (Output)
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(30,), name="dense_64"),
        tf.keras.layers.Dropout(0.2, name="dropout_1"),
        tf.keras.layers.Dense(32, activation='relu', name="dense_32"),
        tf.keras.layers.Dropout(0.2, name="dropout_2"), 
        tf.keras.layers.Dense(16, activation='relu', name="dense_16"),
        tf.keras.layers.Dense(1, activation='sigmoid', name="output_layer")
    ])
    model.compile(
        optimizer='adam', 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )
    return model

# --- 4. LIFESPAN MANAGEMENT ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global GLOBAL_MODEL, CURRENT_ROUND
    logger.info("Initializing FinSecure Global Model and Scheduler...")
    
    # Initialize blank model
    GLOBAL_MODEL = create_fraud_detection_model()
    
    # Resume from the last known round in MongoDB
    latest_round_doc = await db.training_rounds.find_one({}, sort=[("round_number", -1)])
    if latest_round_doc:
        CURRENT_ROUND = latest_round_doc['round_number'] + 1
        logger.info(f"Resuming training from Round {CURRENT_ROUND}")
    else:
        CURRENT_ROUND = 0
        logger.info("Starting fresh training at Round 0")
    
    # Auto-aggregation Job (Runs every 2 minutes)
    scheduler.add_job(
        auto_aggregate_gradients, 
        'interval', 
        minutes=2,  
        id='federated_auto_aggregator', 
        replace_existing=True
    )
    scheduler.start()
    
    yield
    # Cleanup
    logger.info("Shutting down FinSecure Backend...")
    scheduler.shutdown()
    client.close()

app = FastAPI(lifespan=lifespan, title="FinSecure Federated Server")

# Rate Limiting setup
if RATE_LIMIT_ENABLED:
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

# CORS for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_router = APIRouter(prefix="/api")

# --- 5. DATA MODELS (PYDANTIC) ---
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

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    email: EmailStr
    otp: str
    new_password: str

# --- 6. UTILITY / SECURITY FUNCTIONS ---
def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))

def generate_api_key():
    return f"fs_{secrets.token_urlsafe(32)}"

async def verify_api_key(x_api_key: str = Header(...)) -> dict:
    company = await db.companies.find_one({"api_key": x_api_key, "status": "active"})
    if not company:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid API Key")
    return company

# --- 7. TENSORFLOW WEIGHT SERIALIZATION ---
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
        return [npz_file[f'arr_{i}'] for i in range(len(npz_file.files))]
    except Exception as e:
        logger.error(f"Matrix Deserialization Error: {e}")
        return None

def validate_gradient_shape(decoded_weights: List[np.ndarray], model: tf.keras.Model) -> bool:
    target_weights = model.get_weights()
    if len(decoded_weights) != len(target_weights): return False
    for d, t in zip(decoded_weights, target_weights):
        if d.shape != t.shape: return False
    return True

# --- 8. EMAIL SYSTEM (FORGOT PASSWORD) ---
def send_otp_email(receiver: str, otp_code: str):
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        logger.error("Email credentials missing. Please check .env file.")
        return
    
    msg = MIMEText(f"Your Cyber Shield Reset OTP: {otp_code}\nValid for 10 minutes.")
    msg['Subject'] = "Cyber Shield - Password Reset Request"
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = receiver

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
            logger.info(f"OTP successfully sent to {receiver}")
    except Exception as e:
        logger.error(f"SMTP Error: {e}")

# --- 9. FEDERATED AGGREGATION LOGIC ---
async def aggregate_gradients():
    async with aggregation_lock:
        global GLOBAL_MODEL, CURRENT_ROUND, PREVIOUS_ACCURACY
        
        rid = f"round_{CURRENT_ROUND}"
        updates = await db.gradient_updates.find({"round_id": rid, "status": "pending"}).to_list(1000)
        
        if not updates or len(updates) < AGGREGATION_THRESHOLD:
            logger.info(f"Aggregation Round {CURRENT_ROUND} skipped: Not enough updates.")
            return

        logger.info(f"Processing Round {CURRENT_ROUND} with {len(updates)} node updates...")
        
        valid_grads, sample_weights = [], []
        sum_acc, total_n = 0.0, 0

        for up in updates:
            w = deserialize_model_weights(up['gradient_data'])
            if w and validate_gradient_shape(w, GLOBAL_MODEL):
                valid_grads.append(w)
                n = int(up.get('num_samples', 1))
                sample_weights.append(n)
                sum_acc += float(up['metrics'].get('accuracy', 0)) * n
                total_n += n
        
        if not valid_grads: return

        # Federated Averaging Math
        new_global_weights = []
        for i in range(len(valid_grads[0])):
            layer_set = [g[i] for g in valid_grads]
            avg_layer = np.average(layer_set, axis=0, weights=sample_weights)
            new_global_weights.append(avg_layer)
        
        GLOBAL_MODEL.set_weights(new_global_weights)
        
        # Save Round Metrics
        final_acc = sum_acc / total_n if total_n > 0 else 0
        await db.training_rounds.insert_one({
            "round_number": CURRENT_ROUND,
            "avg_accuracy": final_acc,
            "participating_nodes": len(valid_grads),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Mark processed
        await db.gradient_updates.update_many({"round_id": rid}, {"$set": {"status": "processed"}})
        
        logger.info(f"Round {CURRENT_ROUND} Complete. Accuracy: {final_acc:.4f}")
        CURRENT_ROUND += 1

async def auto_aggregate_gradients():
    try: await aggregate_gradients()
    except Exception as e: logger.error(f"Scheduler Error: {e}")

# --- 10. AUTH & ACCOUNT ROUTES ---
@api_router.post("/auth/register")
async def register_company(data: CompanyRegister):
    existing = await db.companies.find_one({"email": data.email})
    if existing: raise HTTPException(status_code=400, detail="Account already exists")
    
    new_comp = {
        "company_id": str(uuid.uuid4()),
        "name": data.name,
        "email": data.email,
        "password_hash": hash_password(data.password),
        "api_key": generate_api_key(),
        "status": "active",
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    await db.companies.insert_one(new_comp)
    return {"message": "Registration successful"}

@api_router.post("/auth/login", response_model=LoginResponse)
async def login_company(data: CompanyLogin):
    user = await db.companies.find_one({"email": data.email})
    if not user or not verify_password(data.password, user['password_hash']):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    return LoginResponse(
        success=True,
        company_id=user['company_id'],
        name=user['name'],
        email=user['email'],
        api_key=user['api_key'],
        message="Login successful"
    )

@api_router.post("/auth/forgot-password")
async def forgot_password(req: ForgotPasswordRequest, bg_tasks: BackgroundTasks):
    user = await db.companies.find_one({"email": req.email})
    if not user: raise HTTPException(status_code=404, detail="Email not found")
    
    otp = secrets.token_hex(3).upper()
    expires = datetime.now(timezone.utc) + timedelta(minutes=10)
    
    await db.companies.update_one(
        {"email": req.email},
        {"$set": {"reset_otp": otp, "otp_expires": expires}}
    )
    bg_tasks.add_task(send_otp_email, req.email, otp)
    return {"success": True, "message": "OTP sent to email"}

@api_router.post("/auth/reset-password")
async def reset_password(req: ResetPasswordRequest):
    user = await db.companies.find_one({"email": req.email})
    if not user or user.get("reset_otp") != req.otp:
        raise HTTPException(status_code=400, detail="Invalid OTP")
    
    exp = user["otp_expires"]
    if exp.tzinfo is None: exp = exp.replace(tzinfo=timezone.utc)
    if datetime.now(timezone.utc) > exp:
        raise HTTPException(status_code=400, detail="OTP Expired")
    
    await db.companies.update_one(
        {"email": req.email},
        {
            "$set": {"password_hash": hash_password(req.new_password)},
            "$unset": {"reset_otp": "", "otp_expires": ""}
        }
    )
    return {"success": True, "message": "Password updated"}

# --- 11. FEDERATED ML ROUTES ---
@api_router.get("/model/download")
async def download_model(comp: dict = Depends(verify_api_key)):
    return {
        "weights": serialize_model_weights(GLOBAL_MODEL),
        "round": CURRENT_ROUND,
        "version": MODEL_VERSION
    }

@api_router.post("/federated/submit-gradients")
async def submit_gradients(sub: GradientSubmit, comp: dict = Depends(verify_api_key)):
    await db.gradient_updates.insert_one({
        "company_id": comp['company_id'],
        "round_id": f"round_{CURRENT_ROUND}",
        "gradient_data": sub.gradient_data,
        "metrics": sub.metrics,
        "num_samples": sub.num_samples,
        "status": "pending",
        "timestamp": datetime.now(timezone.utc).isoformat()
    })
    logger.info(f"Gradients received from {comp['name']} for Round {CURRENT_ROUND}")
    return {"success": True}

# --- 12. ANALYTICS & DASHBOARD ---
@api_router.get("/analytics/dashboard", response_model=DashboardStats)
async def get_dashboard():
    latest = await db.training_rounds.find_one({}, sort=[("round_number", -1)])
    return DashboardStats(
        total_companies=await db.companies.count_documents({}),
        active_companies=await db.companies.count_documents({"status": "active"}),
        total_rounds=await db.training_rounds.count_documents({}),
        current_accuracy=latest['avg_accuracy'] if latest else 0.85,
        total_updates=await db.gradient_updates.count_documents({}),
        latest_round=latest
    )

@api_router.get("/analytics/rounds")
async def get_round_history():
    history = await db.training_rounds.find({}, {"_id": 0}).sort("round_number", 1).to_list(100)
    return history

@api_router.get("/reset-system")
async def reset_system():
    # Dangerous: Only for development
    await db.training_rounds.delete_many({})
    await db.gradient_updates.delete_many({})
    global CURRENT_ROUND, GLOBAL_MODEL
    CURRENT_ROUND = 0
    GLOBAL_MODEL = create_fraud_detection_model()
    return {"status": "System Wiped"}

@api_router.get("/client/script")
async def get_client_script(request: Request, comp: dict = Depends(verify_api_key)):
    # Dynamically generates the Python gateway script for the bank
    base = f"{str(request.base_url).rstrip('/')}/api"
    script = f"""
import requests, json, os, time
# Cyber Shield Gateway for {comp['name']}
API_KEY = "{comp['api_key']}"
URL = "{base}"
def sync():
    try:
        r = requests.get(f"{{URL}}/model/download", headers={{"X-API-Key": API_KEY}}, timeout=30)
        if r.status_code == 200:
            with open("global_model.json", "w") as f: json.dump(r.json(), f)
            print("Downstream: Global model updated.")
        if os.path.exists("local_gradients.json"):
            with open("local_gradients.json", "r") as f:
                payload = json.load(f)
            resp = requests.post(f"{{URL}}/federated/submit-gradients", headers={{"X-API-Key": API_KEY}}, json=payload, timeout=60)
            if resp.status_code == 200:
                print("Upstream: Local gradients uploaded.")
                os.remove("local_gradients.json")
    except Exception as e: print(f"Gateway Error: {{e}}")

print("Cyber Shield Gateway is running...")
while True: sync(); time.sleep(15)
"""
    return PlainTextResponse(script, media_type="text/x-python")

app.include_router(api_router)
