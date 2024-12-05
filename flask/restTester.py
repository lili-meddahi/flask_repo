import requests

# Define the base URL of the Flask app
BASE_URL = "http://localhost:9090"

# Test data for the `/predict` endpoint
test_data = {
    'Brand': 'Toyota',
    'model': 'Corolla',
    'Year': '2015',
    'kmDriven': '50000',
    'Transmission': 'Manual',
    'FuelType': 'Petrol'
}

def test_predict_endpoint():
    print("Testing `/predict` endpoint...")
    response = requests.post(f"{BASE_URL}/predict", data=test_data)
    
    if response.status_code == 200:
        print("Response:", response.json())
    else:
        print(f"Error: Status code {response.status_code}, Response: {response.text}")

if __name__ == "__main__":
    print("Starting REST tester...")
    
    # Test the `/predict` endpoint
    test_predict_endpoint()
