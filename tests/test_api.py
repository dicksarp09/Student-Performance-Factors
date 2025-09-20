import subprocess
import time
import requests

CONTAINER_NAME = "test_student_api"
IMAGE_NAME = "dicksonml/student-perf-app:latest"
BASE_URL = "http://127.0.0.1:5002"

def wait_for_api(url, timeout=120, interval=5):  # wait up to 2 minutes
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = requests.get(url)
            # Ensure API responds with 200 and not HTML fallback (like MLflow UI)
            if response.status_code == 200 and "Welcome" in response.text or "Student Performance" in response.text:
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(interval)
    return False

def setup_module(module):
    """Start the container before tests."""
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], check=False)

    subprocess.run([
        "docker", "run", "-d", "-p", "5002:5002",
        "--name", CONTAINER_NAME, IMAGE_NAME
    ], check=True)

    if not wait_for_api(f"{BASE_URL}/"):
        subprocess.run(["docker", "logs", CONTAINER_NAME])
        assert False, "API did not become ready in time"

def teardown_module(module):
    """Stop container after tests."""
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], check=False)

def test_root_endpoint():
    r = requests.get(f"{BASE_URL}/")
    assert r.status_code == 200
    assert "Welcome" in r.text or "Student Performance" in r.text

def test_predict_endpoint():
    payload = {
        "Parental_Involvement": "High",
        "Access_to_Resources": "Medium",
        "Extracurricular_Activities": "Yes",
        "Motivation_Level": "High",
        "Internet_Access": "Yes",
        "Family_Income": "Medium",
        "Teacher_Quality": "High",
        "School_Type": "Public",
        "Peer_Influence": "Positive",
        "Learning_Disabilities": "No",
        "Parental_Education_Level": "College",
        "Distance_from_Home": "Near",
        "Gender": "Female",
        "Hours_Studied": 15,
        "Attendance": 90,
        "Sleep_Hours": 7,
        "Previous_Scores": 78,
        "Tutoring_Sessions": 3,
        "Physical_Activity": 4
    }
    r = requests.post(f"{BASE_URL}/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "prediction" in data
