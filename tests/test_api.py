import subprocess
import time
import requests
import socket

CONTAINER_NAME = "test_student_api"
IMAGE_NAME = "dicksonml/student-perf-app:latest"

def get_free_port():
    """Find a free port on the host machine."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

# dynamically pick a free port
HOST_PORT = get_free_port()
BASE_URL = f"http://127.0.0.1:{HOST_PORT}"

def wait_for_api(url, timeout=120, interval=5):  # wait up to 2 minutes
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(interval)
    return False

def setup_module(module):
    """Start the container before tests."""
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], check=False)

    docker_cmd = [
        "docker", "run", "-d",
        "-p", f"{HOST_PORT}:5002",  # map free host port to container port
        "--name", CONTAINER_NAME,
        IMAGE_NAME
    ]
    subprocess.run(docker_cmd, check=True)

    if not wait_for_api(BASE_URL):
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
    assert "prediction" in r.json()
