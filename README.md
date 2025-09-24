<img width="1024" height="1024" alt="Gemini_Generated_Image_3l9tnz3l9tnz3l9t" src="https://github.com/user-attachments/assets/04507d12-05b1-4f38-8b8c-470f93e228a7" />



# 📊 Student Performance Project

This project is a full-cycle Machine Learning application designed to **predict student performance** using both linear and polynomial models. Beyond just training models, it showcases **modular code design, experiment tracking, MLOps practices, containerization, cloud deployment, and demo hosting**.

---

## 🚀 Project Overview

The goal was simple but ambitious: take raw student performance data and build a scalable, production-ready ML system that can predict outcomes, track experiments, and deploy models in real-world environments.

---

## 🏗 Phases of the Project

| Phase                               | Description                                                                                                                                                                               |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Model Development**            | Trained **Linear Regression** and **Polynomial Regression** models. Modularized code into separate scripts, with `train.py` for training and comparing results against Jupyter Notebooks. |
| **2. Data & Experiment Management** | Versioned datasets using **DVC**, tracked experiments with **MLflow** for easy comparison of model performance and metrics.                                                               |
| **3. API Development**              | Built a **Flask API** to serve predictions. Integrated **DVC and MLflow** into the API to handle data and model versions dynamically.                                                     |
| **4. Containerization**             | Containerized the application using **Docker**, pushed images to **DockerHub**. Tested CI/CD workflows with **GitHub Actions** to ensure builds run smoothly.                             |
| **5. Cloud Deployment**             | Pushed Docker images from DockerHub to **AWS ECR** using IAM roles. Deployed to **AWS ECS** with cluster, task definition, and service setup. Accessed API via **EC2** instances.         |
| **6. Model Registry & Demo**        | Registered models with **AWS SageMaker Model Registry** for version control. Deployed a **demo on Hugging Face Spaces** for interactive testing.                                          |

---

## 🛠 Tech Stack

| Layer                        | Technology                               |
| ---------------------------- | ---------------------------------------- |
| **ML Models**                | Linear Regression, Polynomial Regression |
| **Experiment Tracking**      | MLflow                                   |
| **Data Versioning**          | DVC                                      |
| **Backend/API**              | Flask                                    |
| **Containerization**         | Docker                                   |
| **CI/CD**                    | GitHub Actions                           |
| **Cloud Deployment**         | AWS ECR, ECS, EC2                        |
| **Model Registry**           | AWS SageMaker                            |
| **Demo Hosting**             | Hugging Face Spaces                      |
| **Programming Language**     | Python                                   |
| **Visualization / Notebook** | Jupyter Notebook                         |

---

## 📈 Project Highlights

* Modularized ML pipeline for **easy experimentation and reuse**
* **Versioned datasets and tracked experiments** for reproducibility
* Full **CI/CD pipeline** from local Docker builds to GitHub Actions testing
* **AWS Cloud deployment** with ECR → ECS → EC2 access
* Demo available publicly via **Hugging Face Spaces**
* **Integration of MLOps tools** (MLflow, DVC, SageMaker) for production-ready workflow

---

