#!/usr/bin/env python3
"""
FinSecure Federated Learning - Backend API Tests
Tests all endpoints for the federated fraud detection system
"""

import requests
import sys
import json
import base64
import numpy as np
from datetime import datetime
from io import BytesIO

class FinSecureAPITester:
    def __init__(self, base_url="https://fintech-defender.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.api_key = None
        self.company_id = None
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []

    def log_test(self, name, success, details=""):
        """Log test result"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {name}")
        else:
            print(f"❌ {name} - {details}")
        
        self.test_results.append({
            "test": name,
            "success": success,
            "details": details
        })

    def run_test(self, name, method, endpoint, expected_status, data=None, headers=None):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}"
        test_headers = {'Content-Type': 'application/json'}
        
        if self.api_key:
            test_headers['X-API-Key'] = self.api_key
        
        if headers:
            test_headers.update(headers)

        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=test_headers, timeout=30)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=test_headers, timeout=30)
            elif method == 'PUT':
                response = requests.put(url, json=data, headers=test_headers, timeout=30)

            success = response.status_code == expected_status
            
            if success:
                self.log_test(name, True)
                try:
                    return True, response.json()
                except:
                    return True, response.text
            else:
                error_msg = f"Expected {expected_status}, got {response.status_code}"
                try:
                    error_detail = response.json()
                    error_msg += f" - {error_detail}"
                except:
                    error_msg += f" - {response.text[:200]}"
                
                self.log_test(name, False, error_msg)
                return False, {}

        except Exception as e:
            self.log_test(name, False, f"Request failed: {str(e)}")
            return False, {}

    def serialize_dummy_gradients(self):
        """Create dummy gradient data for testing"""
        # Create dummy weights similar to the model structure
        weights = [
            np.random.randn(30, 64).astype(np.float32),  # Dense layer 1
            np.random.randn(64).astype(np.float32),      # Bias 1
            np.random.randn(64, 32).astype(np.float32),  # Dense layer 2
            np.random.randn(32).astype(np.float32),      # Bias 2
            np.random.randn(32, 16).astype(np.float32),  # Dense layer 3
            np.random.randn(16).astype(np.float32),      # Bias 3
            np.random.randn(16, 1).astype(np.float32),   # Output layer
            np.random.randn(1).astype(np.float32)        # Output bias
        ]
        
        buffer = BytesIO()
        np.savez_compressed(buffer, *weights)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')

    def test_company_registration(self):
        """Test company registration endpoint"""
        test_company = {
            "name": f"Test Fintech {datetime.now().strftime('%H%M%S')}",
            "email": f"test_{datetime.now().strftime('%H%M%S')}@testfintech.com"
        }
        
        success, response = self.run_test(
            "Company Registration",
            "POST",
            "auth/register",
            200,
            data=test_company
        )
        
        if success and 'api_key' in response:
            self.api_key = response['api_key']
            self.company_id = response['company_id']
            print(f"   Generated API Key: {self.api_key[:20]}...")
            return True
        return False

    def test_duplicate_registration(self):
        """Test duplicate email registration"""
        if not self.api_key:
            return False
            
        # Try to register with same email
        duplicate_company = {
            "name": "Duplicate Test",
            "email": f"test_{datetime.now().strftime('%H%M%S')}@testfintech.com"
        }
        
        success, response = self.run_test(
            "Duplicate Registration Prevention",
            "POST", 
            "auth/register",
            400,  # Should fail with 400
            data=duplicate_company
        )
        return success

    def test_api_key_verification(self):
        """Test API key verification"""
        if not self.api_key:
            return False
            
        success, response = self.run_test(
            "API Key Verification",
            "GET",
            "auth/verify",
            200
        )
        
        if success and response.get('valid'):
            print(f"   Verified company: {response.get('name')}")
            return True
        return False

    def test_invalid_api_key(self):
        """Test invalid API key handling"""
        old_key = self.api_key
        self.api_key = "invalid_key_12345"
        
        success, response = self.run_test(
            "Invalid API Key Rejection",
            "GET",
            "auth/verify", 
            401  # Should fail with 401
        )
        
        self.api_key = old_key  # Restore valid key
        return success

    def test_get_companies(self):
        """Test getting all companies"""
        success, response = self.run_test(
            "Get Companies List",
            "GET",
            "companies",
            200
        )
        
        if success and isinstance(response, list):
            print(f"   Found {len(response)} companies")
            return True
        return False

    def test_get_current_model(self):
        """Test getting current model info"""
        success, response = self.run_test(
            "Get Current Model Info",
            "GET",
            "model/current",
            200
        )
        
        if success and 'version' in response:
            print(f"   Model version: {response.get('version')}")
            print(f"   Accuracy: {response.get('accuracy', 'N/A')}")
            return True
        return False

    def test_download_model(self):
        """Test model download (requires auth)"""
        if not self.api_key:
            return False
            
        success, response = self.run_test(
            "Download Model Weights",
            "GET",
            "model/download",
            200
        )
        
        if success and 'weights' in response and 'architecture' in response:
            print(f"   Downloaded model version: {response.get('version')}")
            print(f"   Architecture layers: {len(response.get('architecture', {}).get('layers', []))}")
            return True
        return False

    def test_submit_gradients(self):
        """Test gradient submission"""
        if not self.api_key:
            return False
            
        gradient_data = self.serialize_dummy_gradients()
        
        gradient_payload = {
            "gradient_data": gradient_data,
            "metrics": {
                "accuracy": 0.87,
                "loss": 0.23,
                "precision": 0.85,
                "recall": 0.89
            }
        }
        
        success, response = self.run_test(
            "Submit Gradient Updates",
            "POST",
            "federated/submit-gradients",
            200,
            data=gradient_payload
        )
        
        if success and response.get('success'):
            print(f"   Round ID: {response.get('round_id')}")
            return True
        return False

    def test_aggregate_gradients(self):
        """Test gradient aggregation"""
        success, response = self.run_test(
            "Aggregate Gradients",
            "POST",
            "federated/aggregate",
            200
        )
        
        if success and response.get('success'):
            print(f"   Round number: {response.get('round_number')}")
            print(f"   Avg accuracy: {response.get('avg_accuracy')}")
            print(f"   Participating companies: {response.get('participating_companies')}")
            return True
        return False

    def test_dashboard_analytics(self):
        """Test dashboard analytics endpoint"""
        success, response = self.run_test(
            "Dashboard Analytics",
            "GET",
            "analytics/dashboard",
            200
        )
        
        if success:
            print(f"   Total companies: {response.get('total_companies')}")
            print(f"   Active companies: {response.get('active_companies')}")
            print(f"   Training rounds: {response.get('total_rounds')}")
            print(f"   Current accuracy: {response.get('current_accuracy')}")
            print(f"   Total updates: {response.get('total_updates')}")
            return True
        return False

    def test_training_rounds(self):
        """Test training rounds history"""
        success, response = self.run_test(
            "Training Rounds History",
            "GET",
            "analytics/rounds",
            200
        )
        
        if success and isinstance(response, list):
            print(f"   Found {len(response)} training rounds")
            return True
        return False

    def test_api_key_prefix(self):
        """Test that API keys start with 'fs_' prefix"""
        if not self.api_key:
            return False
            
        if self.api_key.startswith('fs_'):
            self.log_test("API Key Prefix Validation", True)
            print(f"   ✓ API key has correct 'fs_' prefix")
            return True
        else:
            self.log_test("API Key Prefix Validation", False, f"API key should start with 'fs_' but starts with '{self.api_key[:3]}'")
            return False

    def test_client_script_branding(self):
        """Test client script contains FinSecure branding"""
        if not self.api_key:
            return False
            
        success, response = self.run_test(
            "Client Script FinSecure Branding",
            "GET",
            "client/script",
            200
        )
        
        if success and 'content' in response and 'filename' in response:
            content = response.get('content', '')
            filename = response.get('filename', '')
            
            # Check filename uses finsecure_client_ prefix
            if filename.startswith('finsecure_client_'):
                print(f"   ✓ Filename has correct prefix: {filename}")
            else:
                self.log_test("Client Script Branding", False, f"Filename should start with 'finsecure_client_' but is '{filename}'")
                return False
            
            # Check content contains FinSecure branding
            if 'FinSecure' in content:
                print(f"   ✓ Script contains FinSecure branding")
                return True
            else:
                self.log_test("Client Script Branding", False, "Script content should contain 'FinSecure' branding")
                return False
        return False
    def test_client_script_download(self):
        """Test client script download"""
        if not self.api_key:
            return False
            
        success, response = self.run_test(
            "Client Script Download",
            "GET",
            "client/script",
            200
        )
        
        if success and 'content' in response and 'filename' in response:
            print(f"   Script filename: {response.get('filename')}")
            print(f"   Script size: {len(response.get('content', ''))} characters")
            # Verify script contains API key
            if self.api_key in response.get('content', ''):
                print(f"   ✓ API key properly embedded in script")
                return True
            else:
                print(f"   ❌ API key not found in script")
                return False
        return False
    def test_notifications_endpoints(self):
        """Test notification system endpoints"""
        if not self.api_key:
            return False
            
        # Test get notifications
        success, response = self.run_test(
            "Get Notifications",
            "GET",
            "notifications",
            200
        )
        
        if not success:
            return False
            
        print(f"   Found {len(response)} notifications")
        
        # Test unread count
        success, response = self.run_test(
            "Get Unread Count",
            "GET",
            "notifications/unread/count",
            200
        )
        
        if success and 'unread_count' in response:
            print(f"   Unread notifications: {response.get('unread_count')}")
            return True
        return False

def main():
    print("=" * 80)
    print("🛡️  FINSECURE FEDERATED LEARNING - API TESTING")
    print("=" * 80)
    
    tester = FinSecureAPITester()
    
    # Test sequence
    tests = [
        ("Company Registration", tester.test_company_registration),
        ("API Key Prefix Validation", tester.test_api_key_prefix),
        ("API Key Verification", tester.test_api_key_verification),
        ("Invalid API Key Handling", tester.test_invalid_api_key),
        ("Companies List", tester.test_get_companies),
        ("Current Model Info", tester.test_get_current_model),
        ("Model Download", tester.test_download_model),
        ("Gradient Submission", tester.test_submit_gradients),
        ("Gradient Aggregation", tester.test_aggregate_gradients),
        ("Dashboard Analytics", tester.test_dashboard_analytics),
        ("Training Rounds", tester.test_training_rounds),
        ("Client Script Download", tester.test_client_script_download),
        ("Client Script Branding", tester.test_client_script_branding),
        ("Notifications System", tester.test_notifications_endpoints)
    ]
    
    print(f"\n🚀 Starting {len(tests)} API tests...\n")
    
    for test_name, test_func in tests:
        try:
            test_func()
        except Exception as e:
            tester.log_test(test_name, False, f"Exception: {str(e)}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    print(f"Tests Run: {tester.tests_run}")
    print(f"Tests Passed: {tester.tests_passed}")
    print(f"Tests Failed: {tester.tests_run - tester.tests_passed}")
    print(f"Success Rate: {(tester.tests_passed / tester.tests_run * 100):.1f}%")
    
    if tester.api_key:
        print(f"\n🔑 Generated API Key: {tester.api_key}")
        print("   (Save this for frontend testing)")
    
    # Print failed tests
    failed_tests = [r for r in tester.test_results if not r['success']]
    if failed_tests:
        print(f"\n❌ FAILED TESTS ({len(failed_tests)}):")
        for test in failed_tests:
            print(f"   • {test['test']}: {test['details']}")
    
    print("\n" + "=" * 80)
    
    return 0 if tester.tests_passed == tester.tests_run else 1

if __name__ == "__main__":
    sys.exit(main())