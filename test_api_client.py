"""
Test client for the Address Structuring FastAPI REST API.

This script demonstrates how to use the API endpoints:
1. Health check
2. Single address structuring
3. Batch address structuring

Requirements:
    pip install requests
"""

import requests
import json
import sys

BASE_URL = "http://127.0.0.1:5000"


def test_health_check():
    """Test the health check endpoint."""
    print("\n" + "="*60)
    print("TEST 1: Health Check")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_single_address():
    """Test structuring a single address."""
    print("\n" + "="*60)
    print("TEST 2: Single Address Structuring")
    print("="*60)
    
    address = """1600 Pennsylvania Ave NW
Washington, DC 20500
USA"""
    
    payload = {
        "address": address
    }
    
    print(f"Request:")
    print(json.dumps(payload, indent=2))
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/structure-address",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        print(f"\nStatus Code: {response.status_code}")
        print(f"Response:")
        print(json.dumps(response.json(), indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_batch_addresses():
    """Test structuring multiple addresses in batch."""
    print("\n" + "="*60)
    print("TEST 3: Batch Address Structuring")
    print("="*60)
    
    addresses = [
        """1600 Pennsylvania Ave NW
Washington, DC 20500
USA""",
        """42 Main Street
New York, NY 10001
USA""",
        """10 Downing Street
London, SW1A 2AA
UK"""
    ]
    
    payload = {
        "addresses": addresses
    }
    
    print(f"Request: {len(addresses)} addresses")
    for i, addr in enumerate(addresses, 1):
        print(f"\n  Address {i}:")
        for line in addr.split("\n"):
            print(f"    {line}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/structure-addresses-batch",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        print(f"\nStatus Code: {response.status_code}")
        print(f"Response:")
        print(json.dumps(response.json(), indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_error_handling():
    """Test error handling with invalid input."""
    print("\n" + "="*60)
    print("TEST 4: Error Handling")
    print("="*60)
    
    all_passed = True
    
    # Test 1: Missing address field
    print("\n1. Missing 'address' field:")
    try:
        response = requests.post(
            f"{BASE_URL}/api/structure-address",
            json={},
            headers={"Content-Type": "application/json"}
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        all_passed = all_passed and response.status_code == 422
    except Exception as e:
        print(f"Error: {e}")
        all_passed = False
    
    # Test 2: Invalid address type
    print("\n2. Invalid address type (not a string):")
    try:
        response = requests.post(
            f"{BASE_URL}/api/structure-address",
            json={"address": 12345},
            headers={"Content-Type": "application/json"}
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        all_passed = all_passed and response.status_code == 422
    except Exception as e:
        print(f"Error: {e}")
        all_passed = False
    
    # Test 3: Invalid endpoint
    print("\n3. Invalid endpoint:")
    try:
        response = requests.get(f"{BASE_URL}/api/invalid-endpoint")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        all_passed = all_passed and response.status_code == 404
    except Exception as e:
        print(f"Error: {e}")
        all_passed = False
    
    # Test 4: Empty address
    print("\n4. Empty address string:")
    try:
        response = requests.post(
            f"{BASE_URL}/api/structure-address",
            json={"address": ""},
            headers={"Content-Type": "application/json"}
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        all_passed = all_passed and response.status_code == 422
    except Exception as e:
        print(f"Error: {e}")
        all_passed = False
    
    return all_passed


if __name__ == "__main__":
    print("\n" + "="*60)
    print("ADDRESS STRUCTURING FASTAPI TEST SUITE")
    print("="*60)
    print("Make sure the API server is running:")
    print("  python api_server.py")
    print("\nOr with uvicorn directly:")
    print("  uvicorn api_server:app --reload --host 127.0.0.1 --port 5000")
    
    
    
    try:
        results = []
        results.append(("Health Check", test_health_check()))
        results.append(("Single Address", test_single_address()))
        results.append(("Batch Addresses", test_batch_addresses()))
        results.append(("Error Handling", test_error_handling()))
        
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        for test_name, passed in results:
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{test_name}: {status}")
        
        all_passed = all(passed for _, passed in results)
        print("\n" + "="*60)
        if all_passed:
            print("ALL TESTS PASSED ✅")
        else:
            print("SOME TESTS FAILED ❌")
        print("="*60)
    
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
