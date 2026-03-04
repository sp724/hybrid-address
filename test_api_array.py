"""
Test script for updated /api/structure-address endpoint
Tests both single address (legacy) and array of addresses (new) formats
"""

import requests
import json

BASE_URL = "http://127.0.0.1:5000"

def test_single_address_format():
    """Test legacy single address format"""
    print("\n" + "="*80)
    print("TEST 1: Single Address Format (Legacy)")
    print("="*80)
    
    request_data = {
        "address": "1600 Pennsylvania Ave NW\nWashington, DC 20500\nUSA"
    }
    
    print("Request:")
    print(json.dumps(request_data, indent=2))
    
    response = requests.post(f"{BASE_URL}/api/structure-address", json=request_data)
    print(f"\nStatus Code: {response.status_code}")
    print("Response:")
    print(json.dumps(response.json(), indent=2))
    
    return response.status_code == 200


def test_multiple_addresses_format():
    """Test new array of addresses format"""
    print("\n" + "="*80)
    print("TEST 2: Multiple Addresses Format (New Array)")
    print("="*80)
    
    request_data = {
        "addresses": [
            "1600 Pennsylvania Ave NW\nWashington, DC 20500\nUSA",
            "10 Downing Street\nLondon, SW1A 2AA\nUK",
            "Eiffel Tower\nParis, France"
        ]
    }
    
    print("Request:")
    print(json.dumps(request_data, indent=2))
    
    response = requests.post(f"{BASE_URL}/api/structure-address", json=request_data)
    print(f"\nStatus Code: {response.status_code}")
    print("Response:")
    print(json.dumps(response.json(), indent=2))
    
    return response.status_code == 200


def test_empty_array_error():
    """Test error handling for empty array"""
    print("\n" + "="*80)
    print("TEST 3: Error Handling - Empty Array")
    print("="*80)
    
    request_data = {
        "addresses": []
    }
    
    print("Request:")
    print(json.dumps(request_data, indent=2))
    
    response = requests.post(f"{BASE_URL}/api/structure-address", json=request_data)
    print(f"\nStatus Code: {response.status_code}")
    print("Response:")
    print(json.dumps(response.json(), indent=2))
    
    return response.status_code == 400


def test_no_input_error():
    """Test error handling when neither address nor addresses provided"""
    print("\n" + "="*80)
    print("TEST 4: Error Handling - No Input Provided")
    print("="*80)
    
    request_data = {}
    
    print("Request:")
    print(json.dumps(request_data, indent=2))
    
    response = requests.post(f"{BASE_URL}/api/structure-address", json=request_data)
    print(f"\nStatus Code: {response.status_code}")
    print("Response:")
    print(json.dumps(response.json(), indent=2))
    
    # Should be 400 since neither address nor addresses provided
    return response.status_code == 400


def main():
    print("\n" + "="*80)
    print("TESTING UPDATED /api/structure-address ENDPOINT")
    print("Supports both single address (string) and array of addresses")
    print("="*80)
    
    results = {
        "Single Address Format": test_single_address_format(),
        "Multiple Addresses Format": test_multiple_addresses_format(),
        "Empty Array Error": test_empty_array_error(),
        "No Input Error": test_no_input_error(),
    }
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")


if __name__ == "__main__":
    main()
