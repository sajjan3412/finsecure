from fastapi import FastAPI, APIRouter, HTTPException, Header, Depends
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
import hashlib
import numpy as np
import tensorflow as tf
from io import BytesIO
import base64
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import asyncio

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app
app = FastAPI()
api_router = APIRouter(prefix="/api")

# ============= MODELS =============

class Company(BaseModel):
    model_config = ConfigDict(extra="ignore")
    company_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    email: EmailStr
    api_key: str
    status: str = "active"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class CompanyRegister(BaseModel):
    name: str
    email: EmailStr

class CompanyResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    company_id: str
    name: str
    email: str
    api_key: str
    status: str
    created_at: datetime

class GradientUpdate(BaseModel):
    model_config = ConfigDict(extra="ignore")
    update_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    company_id: str
    round_id: str
    gradient_data: str  # Base64 encoded
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class GradientSubmit(BaseModel):
    gradient_data: str
    metrics: Dict[str, float]

class TrainingRound(BaseModel):
    model_config = ConfigDict(extra="ignore")
    round_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    round_number: int
    participating_companies: List[str]
    avg_accuracy: float
    avg_loss: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ModelVersion(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    version: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

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
    company_id: Optional[str] = None  # None means broadcast to all
    title: str
    message: str
    type: str = "info"  # info, success, warning, error
    read: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class NotificationCreate(BaseModel):
    company_id: Optional[str] = None
    title: str
    message: str
    type: str = "info"

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

# Initialize global model
GLOBAL_MODEL = create_fraud_detection_model()
MODEL_VERSION = "1.0.0"
CURRENT_ROUND = 0
PREVIOUS_ACCURACY = 0.85
AGGREGATION_THRESHOLD = 2  # Minimum number of gradients before aggregation

# ============= NOTIFICATION HELPERS =============

async def create_notification(title: str, message: str, notification_type: str = "info", company_id: Optional[str] = None):
    """Create a notification for a company or broadcast to all"""
    notification = {
        "notification_id": str(uuid.uuid4()),
        "company_id": company_id,
        "title": title,
        "message": message,
        "type": notification_type,
        "read": False,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    await db.notifications.insert_one(notification)
    logger.info(f"Created notification: {title} for company: {company_id or 'ALL'}")

async def broadcast_notification(title: str, message: str, notification_type: str = "info"):
    """Send notification to all active companies"""
    companies = await db.companies.find({"status": "active"}, {"_id": 0}).to_list(1000)
    for company in companies:
        await create_notification(title, message, notification_type, company['company_id'])

# ============= HELPER FUNCTIONS =============

async def verify_api_key(x_api_key: str = Header(...)):
    """Verify API key from header"""
    company = await db.companies.find_one({"api_key": x_api_key, "status": "active"}, {"_id": 0})
    if not company:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return company

def generate_api_key():
    """Generate a secure API key"""
    return f"fs_{secrets.token_urlsafe(32)}"

def serialize_model_weights(model):
    """Serialize model weights to base64 string"""
    weights = model.get_weights()
    buffer = BytesIO()
    np.savez_compressed(buffer, *weights)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def deserialize_model_weights(data_str):
    """Deserialize model weights from base64 string"""
    try:
        data = base64.b64decode(data_str)
        buffer = BytesIO(data)
        buffer.seek(0)
        npz_file = np.load(buffer, allow_pickle=True)
        weights = [npz_file[f'arr_{i}'] for i in range(len(npz_file.files))]
        return weights
    except Exception as e:
        logger.error(f"Error deserializing weights: {str(e)}")
        raise

def federated_averaging(gradient_list):
    """Perform federated averaging of gradients"""
    if not gradient_list:
        return None
    
    # Average all gradients
    avg_gradients = []
    for i in range(len(gradient_list[0])):
        layer_gradients = [g[i] for g in gradient_list]
        avg_gradients.append(np.mean(layer_gradients, axis=0))
    
    return avg_gradients

# ============= API ROUTES =============

@api_router.post("/auth/register", response_model=CompanyResponse)
async def register_company(company_input: CompanyRegister):
    """Register a new fintech company"""
    # Check if email already exists
    existing = await db.companies.find_one({"email": company_input.email})
    if existing:
        raise HTTPException(status_code=400, detail="Company with this email already exists")
    
    # Generate API key
    api_key = generate_api_key()
    
    # Create company
    company = Company(
        name=company_input.name,
        email=company_input.email,
        api_key=api_key
    )
    
    doc = company.model_dump()
    doc['created_at'] = doc['created_at'].isoformat()
    
    await db.companies.insert_one(doc)
    
    return company

@api_router.get("/auth/verify")
async def verify_key(company: dict = Depends(verify_api_key)):
    """Verify API key validity"""
    return {
        "valid": True,
        "company_id": company['company_id'],
        "name": company['name']
    }

@api_router.get("/companies", response_model=List[CompanyResponse])
async def get_companies():
    """Get all registered companies"""
    companies = await db.companies.find({}, {"_id": 0}).to_list(1000)
    for company in companies:
        if isinstance(company['created_at'], str):
            company['created_at'] = datetime.fromisoformat(company['created_at'])
    return companies

@api_router.get("/model/current")
async def get_current_model():
    """Get current model information"""
    model_info = await db.model_versions.find_one(
        {"version": MODEL_VERSION},
        {"_id": 0}
    )
    
    if not model_info:
        # Create initial model info
        model_info = {
            "model_id": str(uuid.uuid4()),
            "version": MODEL_VERSION,
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88,
            "f1_score": 0.85,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        await db.model_versions.insert_one(model_info)
    
    return model_info

@api_router.get("/model/download")
async def download_model(company: dict = Depends(verify_api_key)):
    """Download current model weights"""
    weights_str = serialize_model_weights(GLOBAL_MODEL)
    
    return {
        "version": MODEL_VERSION,
        "weights": weights_str,
        "architecture": {
            "input_shape": [30],
            "layers": [
                {"type": "dense", "units": 64, "activation": "relu"},
                {"type": "dropout", "rate": 0.2},
                {"type": "dense", "units": 32, "activation": "relu"},
                {"type": "dropout", "rate": 0.2},
                {"type": "dense", "units": 16, "activation": "relu"},
                {"type": "dense", "units": 1, "activation": "sigmoid"}
            ]
        }
    }

@api_router.post("/federated/submit-gradients")
async def submit_gradients(gradient_submit: GradientSubmit, company: dict = Depends(verify_api_key)):
    """Submit gradient updates from client"""
    global CURRENT_ROUND
    
    # Get or create current round
    round_id = f"round_{CURRENT_ROUND}"
    
    # Save gradient update
    gradient_update = {
        "update_id": str(uuid.uuid4()),
        "company_id": company['company_id'],
        "round_id": round_id,
        "gradient_data": gradient_submit.gradient_data,
        "metrics": gradient_submit.metrics,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    await db.gradient_updates.insert_one(gradient_update)
    
    return {
        "success": True,
        "round_id": round_id,
        "message": "Gradients received successfully"
    }

@api_router.post("/federated/aggregate")
async def aggregate_gradients():
    """Aggregate gradients and update global model"""
    global GLOBAL_MODEL, CURRENT_ROUND, MODEL_VERSION, PREVIOUS_ACCURACY
    
    round_id = f"round_{CURRENT_ROUND}"
    
    # Get all gradients for current round
    updates = await db.gradient_updates.find(
        {"round_id": round_id},
        {"_id": 0}
    ).to_list(1000)
    
    if not updates:
        return {"success": False, "message": "No gradients to aggregate"}
    
    if len(updates) < AGGREGATION_THRESHOLD:
        return {
            "success": False, 
            "message": f"Waiting for more gradients ({len(updates)}/{AGGREGATION_THRESHOLD})"
        }
    
    # Deserialize gradients
    gradients = [deserialize_model_weights(update['gradient_data']) for update in updates]
    
    # Perform federated averaging
    avg_gradients = federated_averaging(gradients)
    
    # Update global model
    GLOBAL_MODEL.set_weights(avg_gradients)
    
    # Calculate metrics
    avg_accuracy = np.mean([update['metrics'].get('accuracy', 0) for update in updates])
    avg_loss = np.mean([update['metrics'].get('loss', 0) for update in updates])
    
    # Save training round
    training_round = {
        "round_id": round_id,
        "round_number": CURRENT_ROUND,
        "participating_companies": [u['company_id'] for u in updates],
        "avg_accuracy": float(avg_accuracy),
        "avg_loss": float(avg_loss),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    await db.training_rounds.insert_one(training_round)
    
    # Check for performance improvement
    accuracy_improvement = avg_accuracy - PREVIOUS_ACCURACY
    if accuracy_improvement > 0.01:  # 1% improvement threshold
        improvement_pct = accuracy_improvement * 100
        await broadcast_notification(
            title="🎯 Model Performance Improved!",
            message=f"Global model accuracy increased by {improvement_pct:.2f}% to {avg_accuracy*100:.2f}%. Round {CURRENT_ROUND} completed with {len(updates)} contributors.",
            notification_type="success"
        )
    elif accuracy_improvement > 0:
        await broadcast_notification(
            title="✓ Training Round Complete",
            message=f"Round {CURRENT_ROUND} completed. Model accuracy: {avg_accuracy*100:.2f}% ({len(updates)} contributors).",
            notification_type="info"
        )
    
    PREVIOUS_ACCURACY = avg_accuracy
    CURRENT_ROUND += 1
    
    logger.info(f"Aggregation complete: Round {CURRENT_ROUND-1}, Accuracy: {avg_accuracy:.4f}")
    
    return {
        "success": True,
        "round_number": CURRENT_ROUND - 1,
        "avg_accuracy": float(avg_accuracy),
        "participating_companies": len(updates),
        "improvement": float(accuracy_improvement)
    }

@api_router.get("/analytics/dashboard", response_model=DashboardStats)
async def get_dashboard_stats():
    """Get dashboard analytics"""
    total_companies = await db.companies.count_documents({})
    active_companies = await db.companies.count_documents({"status": "active"})
    total_rounds = await db.training_rounds.count_documents({})
    total_updates = await db.gradient_updates.count_documents({})
    
    # Get latest round
    latest_round = await db.training_rounds.find_one(
        {},
        {"_id": 0},
        sort=[("round_number", -1)]
    )
    
    # Get current model accuracy
    model_info = await db.model_versions.find_one({"version": MODEL_VERSION}, {"_id": 0})
    current_accuracy = model_info.get('accuracy', 0.85) if model_info else 0.85
    
    if latest_round:
        current_accuracy = latest_round.get('avg_accuracy', current_accuracy)
    
    return DashboardStats(
        total_companies=total_companies,
        active_companies=active_companies,
        total_rounds=total_rounds,
        current_accuracy=current_accuracy,
        total_updates=total_updates,
        latest_round=latest_round
    )

@api_router.get("/analytics/rounds")
async def get_training_rounds():
    """Get training round history"""
    rounds = await db.training_rounds.find(
        {},
        {"_id": 0}
    ).sort("round_number", 1).to_list(1000)
    
    return rounds

@api_router.get("/client/script")
async def get_client_script(company: dict = Depends(verify_api_key)):
    """Get client script for company"""
    script_template = '''#!/usr/bin/env python3
"""
FinSecure Federated Learning - Client Script
Company: {company_name}
Generated: {timestamp}
"""

import tensorflow as tf
import numpy as np
import requests
import base64
from io import BytesIO
import json

# Configuration
API_KEY = "{api_key}"
BACKEND_URL = "https://fintech-defender.preview.emergentagent.com/api"

class FederatedClient:
    def __init__(self, api_key, backend_url):
        self.api_key = api_key
        self.backend_url = backend_url
        self.model = None
        self.headers = {{"X-API-Key": self.api_key}}
    
    def download_model(self):
        """Download current global model"""
        response = requests.get(
            f"{{self.backend_url}}/model/download",
            headers=self.headers
        )
        if response.status_code == 200:
            data = response.json()
            self.model = self._build_model()
            weights = self._deserialize_weights(data['weights'])
            self.model.set_weights(weights)
            print(f"✓ Model downloaded (version {{data['version']}})")
        else:
            print(f"✗ Failed to download model: {{response.status_code}}")
    
    def _build_model(self):
        """Build model architecture"""
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
    
    def _serialize_weights(self, weights):
        """Serialize model weights"""
        buffer = BytesIO()
        np.savez_compressed(buffer, *weights)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')
    
    def _deserialize_weights(self, data_str):
        """Deserialize model weights"""
        data = base64.b64decode(data_str)
        buffer = BytesIO(data)
        npz_file = np.load(buffer)
        return [npz_file[f'arr_{{i}}'] for i in range(len(npz_file.files))]
    
    def train_local(self, X_train, y_train, epochs=5, batch_size=32):
        """Train model on local data (PRIVATE - never shared)"""
        if self.model is None:
            print("✗ Model not loaded. Call download_model() first.")
            return
        
        print(f"Training on {{len(X_train)}} local samples...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=0
        )
        
        final_accuracy = history.history['accuracy'][-1]
        final_loss = history.history['loss'][-1]
        
        print(f"✓ Training complete - Accuracy: {{final_accuracy:.4f}}, Loss: {{final_loss:.4f}}")
        
        return {{
            'accuracy': float(final_accuracy),
            'loss': float(final_loss)
        }}
    
    def submit_gradients(self, metrics):
        """Submit only gradients to central server (NO RAW DATA)"""
        if self.model is None:
            print("✗ Model not trained. Train first.")
            return
        
        # Serialize ONLY model weights/gradients
        weights = self.model.get_weights()
        gradient_data = self._serialize_weights(weights)
        
        # Submit to server
        response = requests.post(
            f"{{self.backend_url}}/federated/submit-gradients",
            headers={{**self.headers, "Content-Type": "application/json"}},
            json={{
                "gradient_data": gradient_data,
                "metrics": metrics
            }}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Gradients submitted successfully (Round: {{result['round_id']}})")
        else:
            print(f"✗ Failed to submit gradients: {{response.status_code}}")
    
    def predict(self, X):
        """Make fraud predictions locally"""
        if self.model is None:
            print("✗ Model not loaded.")
            return None
        return self.model.predict(X)

# Example usage
if __name__ == "__main__":
    print("="*60)
    print("FinSecure Federated Learning Client")
    print("Company: {company_name}")
    print("="*60)
    
    client = FederatedClient(API_KEY, BACKEND_URL)
    
    # Step 1: Download global model
    print("\\n[1] Downloading global model...")
    client.download_model()
    
    # Step 2: Load YOUR local transaction data
    print("\\n[2] Loading local transaction data...")
    # IMPORTANT: Replace this with your actual transaction data
    # This is just dummy data for demonstration
    X_train = np.random.randn(1000, 30).astype(np.float32)
    y_train = np.random.randint(0, 2, 1000).astype(np.float32)
    print(f"✓ Loaded {{len(X_train)}} transactions (PRIVATE - stays local)")
    
    # Step 3: Train on local data
    print("\\n[3] Training on local data...")
    metrics = client.train_local(X_train, y_train, epochs=3)
    
    # Step 4: Submit ONLY gradients (not data)
    print("\\n[4] Submitting gradients to central server...")
    client.submit_gradients(metrics)
    
    print("\\n" + "="*60)
    print("✓ Federated learning cycle complete!")
    print("Your transaction data remained completely private.")
    print("="*60)
'''
    
    script_content = script_template.format(
        company_name=company['name'],
        api_key=company['api_key'],
        timestamp=datetime.now(timezone.utc).isoformat()
    )
    
    return {
        "filename": f"finsecure_client_{company['company_id'][:8]}.py",
        "content": script_content
    }

@api_router.get("/notifications", response_model=List[Notification])
async def get_notifications(company: dict = Depends(verify_api_key), unread_only: bool = False):
    """Get notifications for a company"""
    query = {
        "$or": [
            {"company_id": company['company_id']},
            {"company_id": None}  # Broadcast notifications
        ]
    }
    
    if unread_only:
        query["read"] = False
    
    notifications = await db.notifications.find(
        query,
        {"_id": 0}
    ).sort("created_at", -1).limit(50).to_list(50)
    
    for notification in notifications:
        if isinstance(notification['created_at'], str):
            notification['created_at'] = datetime.fromisoformat(notification['created_at'])
    
    return notifications

@api_router.post("/notifications/{notification_id}/read")
async def mark_notification_read(notification_id: str, company: dict = Depends(verify_api_key)):
    """Mark a notification as read"""
    result = await db.notifications.update_one(
        {"notification_id": notification_id},
        {"$set": {"read": True}}
    )
    
    if result.modified_count > 0:
        return {"success": True}
    else:
        raise HTTPException(status_code=404, detail="Notification not found")

@api_router.get("/notifications/unread/count")
async def get_unread_count(company: dict = Depends(verify_api_key)):
    """Get count of unread notifications"""
    count = await db.notifications.count_documents({
        "$or": [
            {"company_id": company['company_id']},
            {"company_id": None}
        ],
        "read": False
    })
    
    return {"unread_count": count}

# Include router
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============= SCHEDULER =============

scheduler = AsyncIOScheduler()

async def auto_aggregate_gradients():
    """Scheduled job to automatically aggregate gradients"""
    try:
        logger.info("Running scheduled gradient aggregation...")
        result = await aggregate_gradients()
        if result.get('success'):
            logger.info(f"Auto-aggregation successful: {result}")
        else:
            logger.info(f"Auto-aggregation skipped: {result.get('message')}")
    except Exception as e:
        logger.error(f"Error in auto-aggregation: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Start scheduler on app startup"""
    # Run aggregation every 5 minutes
    scheduler.add_job(
        auto_aggregate_gradients,
        'interval',
        minutes=5,
        id='auto_aggregate',
        replace_existing=True
    )
    scheduler.start()
    logger.info("Scheduler started - auto-aggregation every 5 minutes")

@app.on_event("shutdown")
async def shutdown_db_client():
    scheduler.shutdown()
    client.close()
