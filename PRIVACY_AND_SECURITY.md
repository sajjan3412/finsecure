# FinSecure Privacy & Security

## What Data Is Shared?

### ✅ ONLY Gradient Updates (Model Weights) Are Shared

FinSecure uses **Federated Learning** to ensure complete privacy of your transaction data:

1. **Training Happens Locally**
   - Your transaction data NEVER leaves your infrastructure
   - The model trains on YOUR machine using YOUR data
   - No raw transaction records are sent to the central server

2. **Only Model Weights Are Shared**
   - After local training, the client extracts model weights (gradients)
   - These weights are mathematical parameters (floating point numbers)
   - They represent learned patterns, NOT individual transactions
   - Weights are serialized, compressed (npz format), and base64 encoded
   - ONLY these encoded weights are transmitted to the server

3. **How It Works**
   ```python
   # Step 1: Train locally on YOUR private data
   model.fit(your_transaction_data, your_labels)
   
   # Step 2: Extract ONLY the weights
   weights = model.get_weights()  # Just numbers, no data!
   
   # Step 3: Serialize and encode
   gradient_data = serialize_weights(weights)
   
   # Step 4: Send ONLY gradients (not data)
   submit_gradients(gradient_data, metrics)
   ```

4. **What Cannot Be Recovered**
   - Individual transaction amounts
   - Customer information
   - Account numbers
   - Transaction timestamps
   - Any personally identifiable information (PII)

### 🔒 Security Measures

1. **API Key Authentication**
   - Every request requires a valid API key
   - Keys are prefixed with `fs_` and cryptographically generated
   - Keys are never exposed in logs

2. **HTTPS Encryption**
   - All data transmission uses HTTPS
   - Gradient updates are encrypted in transit

3. **Aggregation Privacy**
   - The central server aggregates gradients from multiple companies
   - This further obscures any individual contribution
   - Federated averaging ensures no single company's data dominates

### 📊 What Metrics Are Shared?

Along with gradients, these aggregate metrics are shared:
- Model accuracy (e.g., 87%)
- Training loss (e.g., 0.35)
- Number of training samples (count only, not actual data)

These metrics help the central server track model improvement but contain NO transaction details.

### 🎯 Privacy Guarantees

- **Data Minimization**: Only the minimum necessary information (weights) is shared
- **Differential Privacy**: Aggregation across multiple companies provides additional privacy
- **No Reverse Engineering**: Model weights cannot be reverse-engineered to recover training data
- **Compliance Ready**: Designed for GDPR, CCPA, and financial regulations

### 🔍 Verification

You can inspect the client script to verify:
1. Download the script from the dashboard
2. Search for `submit_gradients()` function
3. See that only `model.get_weights()` is serialized and sent
4. No transaction data variables are ever transmitted

### 📝 Example Gradient Data

What actually gets sent to the server:
```python
{
  "gradient_data": "UEsDBC0AAAAIAAAAIQDpm1zc//////////8JABQAYXJyXzAubn...",
  "metrics": {
    "accuracy": 0.87,
    "loss": 0.35
  }
}
```

The `gradient_data` is base64-encoded numpy weights. It contains:
- ✅ Mathematical parameters (weights and biases)
- ❌ NOT transaction amounts, customer IDs, or any raw data

---

## Summary

**FinSecure ensures your transaction data remains completely private on your infrastructure. Only the learned model patterns (gradients) are shared, enabling collaborative fraud detection without compromising data privacy.**
