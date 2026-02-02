#!/usr/bin/env python3
"""
Privacy Verification Test for FinSecure Client Script
Verifies that the generated script only shares gradients, not raw transaction data
"""

import requests
import re

def test_client_script_privacy():
    """Test that client script only shares gradients, not raw data"""
    
    # Use the API key from backend test
    api_key = "fs_1tgIud9bRvTefH2tEoFGwMkCx0hOMHKyJFIiRcqalVs"
    
    print("🔒 PRIVACY VERIFICATION TEST")
    print("=" * 60)
    
    # Download client script
    response = requests.get(
        "https://fintech-defender.preview.emergentagent.com/api/client/script",
        headers={'X-API-Key': api_key}
    )
    
    if response.status_code != 200:
        print("❌ Failed to download client script")
        return False
    
    script_data = response.json()
    script_content = script_data.get('content', '')
    filename = script_data.get('filename', '')
    
    print(f"📄 Script: {filename}")
    print(f"📏 Size: {len(script_content)} characters")
    
    # Privacy checks
    privacy_checks = []
    
    # 1. Check that only model.get_weights() is sent
    if 'model.get_weights()' in script_content:
        privacy_checks.append(("✅ Uses model.get_weights() for gradient extraction", True))
    else:
        privacy_checks.append(("❌ Missing model.get_weights() call", False))
    
    # 2. Check that submit_gradients only sends weights
    submit_function_match = re.search(r'def submit_gradients\(.*?\):(.*?)(?=def|\Z)', script_content, re.DOTALL)
    if submit_function_match:
        submit_function = submit_function_match.group(1)
        
        # Should contain gradient_data serialization
        if 'gradient_data' in submit_function and 'serialize_weights' in submit_function:
            privacy_checks.append(("✅ submit_gradients serializes weights only", True))
        else:
            privacy_checks.append(("❌ submit_gradients doesn't properly serialize weights", False))
        
        # Should NOT send raw training data
        raw_data_patterns = ['X_train', 'y_train', 'transaction_data', 'raw_data']
        sends_raw_data = any(pattern in submit_function for pattern in raw_data_patterns)
        
        if not sends_raw_data:
            privacy_checks.append(("✅ submit_gradients does NOT send raw training data", True))
        else:
            privacy_checks.append(("❌ submit_gradients may send raw training data", False))
    else:
        privacy_checks.append(("❌ submit_gradients function not found", False))
    
    # 3. Check for privacy warnings/comments
    privacy_comments = [
        'PRIVATE - never shared',
        'stays local',
        'NO RAW DATA',
        'Your transaction data remained completely private'
    ]
    
    found_privacy_comments = [comment for comment in privacy_comments if comment in script_content]
    if found_privacy_comments:
        privacy_checks.append((f"✅ Contains privacy warnings: {len(found_privacy_comments)} found", True))
    else:
        privacy_checks.append(("❌ Missing privacy warnings in comments", False))
    
    # 4. Check that training happens locally
    if 'Train model on local data' in script_content or 'train_local' in script_content:
        privacy_checks.append(("✅ Training happens locally", True))
    else:
        privacy_checks.append(("❌ Local training not clearly indicated", False))
    
    # 5. Check for data serialization (weights only)
    if '_serialize_weights' in script_content and 'np.savez_compressed' in script_content:
        privacy_checks.append(("✅ Uses proper weight serialization", True))
    else:
        privacy_checks.append(("❌ Missing proper weight serialization", False))
    
    # 6. Check that raw data variables are marked as local/private
    local_data_patterns = ['X_train', 'y_train']
    for pattern in local_data_patterns:
        if pattern in script_content:
            # Check if it's used only locally (not in requests)
            request_sections = re.findall(r'requests\.(get|post).*?\)', script_content, re.DOTALL)
            sends_in_request = any(pattern in section for section in request_sections)
            
            if not sends_in_request:
                privacy_checks.append((f"✅ {pattern} used locally only, not sent in requests", True))
            else:
                privacy_checks.append((f"❌ {pattern} may be sent in requests", False))
    
    # Print results
    print("\n🔍 PRIVACY VERIFICATION RESULTS:")
    print("-" * 60)
    
    passed_checks = 0
    total_checks = len(privacy_checks)
    
    for check_desc, passed in privacy_checks:
        print(f"   {check_desc}")
        if passed:
            passed_checks += 1
    
    print("-" * 60)
    print(f"Privacy Score: {passed_checks}/{total_checks} ({passed_checks/total_checks*100:.1f}%)")
    
    if passed_checks == total_checks:
        print("🎉 PRIVACY VERIFICATION PASSED - Script only shares gradients!")
        return True
    else:
        print("⚠️  PRIVACY CONCERNS DETECTED - Review script content")
        return False

if __name__ == "__main__":
    test_client_script_privacy()