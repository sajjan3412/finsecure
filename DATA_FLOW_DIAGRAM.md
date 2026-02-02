# FinSecure Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMPANY'S INFRASTRUCTURE                      │
│                    (Your Private Environment)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Your Transaction Data (PRIVATE - NEVER LEAVES)              │
│     ┌──────────────────────────────────────────┐               │
│     │ Transaction ID: 12345                     │               │
│     │ Amount: $1,250.00                         │               │
│     │ Customer: John Doe                        │               │
│     │ Card: **** **** **** 4321                │               │
│     │ Timestamp: 2026-02-02 10:30:45           │               │
│     │ Merchant: Store XYZ                       │               │
│     │ Fraud: No                                 │               │
│     └──────────────────────────────────────────┘               │
│                          ↓                                       │
│  2. Local Training (HAPPENS ON YOUR MACHINE)                    │
│     ┌──────────────────────────────────────────┐               │
│     │  model.fit(X_train, y_train)             │               │
│     │  - Learns patterns from YOUR data        │               │
│     │  - Updates neural network weights        │               │
│     │  - NO data sent anywhere yet             │               │
│     └──────────────────────────────────────────┘               │
│                          ↓                                       │
│  3. Extract ONLY Model Weights (Mathematical Parameters)        │
│     ┌──────────────────────────────────────────┐               │
│     │  weights = model.get_weights()           │               │
│     │                                           │               │
│     │  Example weights (just numbers):         │               │
│     │  [0.234, -0.891, 0.445, 0.123, ...]     │               │
│     │  [1.234, 0.567, -0.234, ...]             │               │
│     │                                           │               │
│     │  ❌ NO transaction amounts                │               │
│     │  ❌ NO customer names                     │               │
│     │  ❌ NO card numbers                       │               │
│     │  ✅ ONLY learned patterns (numbers)      │               │
│     └──────────────────────────────────────────┘               │
│                          ↓                                       │
│  4. Serialize & Encode Weights                                  │
│     ┌──────────────────────────────────────────┐               │
│     │  gradient_data = base64.encode(weights)  │               │
│     │                                           │               │
│     │  Result:                                  │               │
│     │  "UEsDBC0AAAAIAAAAIQDpm1zc..."          │               │
│     └──────────────────────────────────────────┘               │
│                          ↓                                       │
└─────────────────────────────────────────────────────────────────┘
                           ↓
                   [HTTPS Encrypted]
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│               FINSECURE CENTRAL SERVER (Cloud)                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  5. Receives ONLY Gradient Data (Encrypted)                     │
│     ┌──────────────────────────────────────────┐               │
│     │  POST /api/federated/submit-gradients    │               │
│     │  {                                        │               │
│     │    "gradient_data": "UEsDBC0AAAAI...",  │               │
│     │    "metrics": {                          │               │
│     │      "accuracy": 0.87,                   │               │
│     │      "loss": 0.35                        │               │
│     │    }                                      │               │
│     │  }                                        │               │
│     │                                           │               │
│     │  ✅ Contains: Model weights (numbers)    │               │
│     │  ❌ Does NOT contain: Transaction data   │               │
│     └──────────────────────────────────────────┘               │
│                          ↓                                       │
│  6. Aggregate Gradients from Multiple Companies                 │
│     ┌──────────────────────────────────────────┐               │
│     │  Company A weights: [0.234, -0.891, ...]│               │
│     │  Company B weights: [0.256, -0.845, ...]│               │
│     │  Company C weights: [0.223, -0.912, ...]│               │
│     │           ↓                               │               │
│     │  Average: [0.238, -0.883, ...]           │               │
│     └──────────────────────────────────────────┘               │
│                          ↓                                       │
│  7. Update Global Model                                         │
│     ┌──────────────────────────────────────────┐               │
│     │  global_model.set_weights(avg_weights)   │               │
│     │  - Improved accuracy: 87% → 89%          │               │
│     │  - Notify all companies of improvement   │               │
│     └──────────────────────────────────────────┘               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Key Privacy Points

### ✅ What IS Shared
- Model weights (floating point numbers)
- Aggregate metrics (accuracy, loss)
- Number of training samples (count only)

### ❌ What is NEVER Shared
- Transaction amounts
- Customer names or IDs
- Card numbers
- Account numbers
- Merchant information
- Timestamps
- Any personally identifiable information (PII)

### 🔒 Security Layers

1. **Local Training**: Data never leaves your infrastructure
2. **Gradient Only**: Only mathematical weights are extracted
3. **Serialization**: Weights are compressed and encoded
4. **HTTPS Encryption**: All transmission is encrypted
5. **API Authentication**: Every request requires valid API key
6. **Aggregation**: Multiple companies' gradients are averaged together
7. **No Reverse Engineering**: Impossible to recover training data from weights

### 📊 Real Example

**Your Private Data:**
```
Transaction 1: $100, Fraud=Yes
Transaction 2: $50, Fraud=No
Transaction 3: $200, Fraud=Yes
```

**What Gets Sent:**
```json
{
  "gradient_data": "UEsDBC0AAAAIAAAAIQDpm1zc...",
  "metrics": {"accuracy": 0.87}
}
```

The gradient_data is just a blob of numbers representing learned patterns. No transaction amounts, no fraud labels, no customer data!

---

**FinSecure: Collaborative fraud detection without compromising privacy.**
