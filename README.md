# Web-Based Deepfake Detection System

This project is a full-stack web application designed to detect deepfakes in images. It features a machine learning backend that performs the analysis and a modern web frontend for user interaction.

## Architecture

The system is built with a decoupled architecture, making it scalable and easy to maintain.

-   **Frontend:** A [Next.js](https://nextjs.org/) application located in the `/frontend` directory. It provides a user-friendly interface for uploading images and displaying the prediction results, including a heatmap visualization.
-   **Backend:** A [Flask](https://flask.palletsprojects.com/) API located in the `/backend` directory. It serves a machine learning model to perform deepfake detection and generates a [Grad-CAM](http://grad-cam.cloudcv.org/) heatmap to explain the model's prediction.

The application is designed for cloud deployment:
-   The frontend is configured for deployment on **Vercel**.
-   The backend is containerized with Docker and configured for deployment on **Render**.

## Features

-   **Image Deepfake Detection:** Upload an image to get a REAL/FAKE prediction.
-   **Explainable AI (XAI):** Visualizes the parts of the image that the model focused on using a Grad-CAM heatmap.
-   **REST API:** A clean and simple API for predictions.
-   **Cloud-Ready:** Pre-configured for easy deployment on modern cloud platforms.

---

## Deployment Instructions

This project is optimized for deployment on Vercel (frontend) and Render (backend).

### 1. Backend Deployment (Render)

The backend is a Dockerized Flask application. Render can automatically build and deploy it using the provided `render.yaml` configuration.

**Prerequisites:**
-   A GitHub account.
-   A Render account.

**Steps:**
1.  **Fork this repository** to your own GitHub account.
2.  On the Render Dashboard, click **"New +"** and select **"Blueprint"**.
3.  Connect the GitHub repository you just forked. Render will automatically detect and use the `render.yaml` file in the root of the repository.
4.  Give your backend service a name (e.g., `deepfake-backend`).
5.  Render will read the `render.yaml` file and configure the service. It will:
    -   Use the `Dockerfile` in the `backend/` directory.
    -   Set up a free-tier web service.
    -   Configure the health check to use the `/` endpoint.
6.  Click **"Apply"** to create and deploy the service. The first build may take several minutes as it needs to download the Docker image and all Python dependencies, including TensorFlow.
7.  Once deployed, Render will provide you with the public URL for your backend service (e.g., `https://your-backend-service.onrender.com`). **Copy this URL.**

### 2. Frontend Deployment (Vercel)

The frontend is a Next.js application. Vercel will automatically detect, build, and deploy it.

**Prerequisites:**
-   A Vercel account.

**Steps:**
1.  On the Vercel Dashboard, click **"Add New..."** and select **"Project"**.
2.  Import the same forked GitHub repository.
3.  Vercel will automatically detect that it is a Next.js project. It will use the `vercel.json` file to understand the monorepo structure.
4.  **Configure Environment Variables:**
    -   Before deploying, you need to tell the frontend where the backend is running.
    -   Go to the "Environment Variables" section.
    -   Add a new environment variable:
        -   **Name:** `NEXT_PUBLIC_API_URL`
        -   **Value:** The URL of your deployed Render backend service (the one you copied in the previous section).
5.  Click **"Deploy"**. Vercel will build and deploy the frontend application.
6.  Once deployed, you will get a public URL for your web application. You can now use the live deepfake detection system!

---

## Local Development (Best-Effort)

**Disclaimer:** The execution environment used during the development of this project was unstable, particularly for running Python-based web servers. The following instructions are provided on a "best-effort" basis and may not work in all environments. Deployment to Render and Vercel is the recommended way to run this application.

### Backend
```bash
# Navigate to the backend directory
cd backend

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the Flask server
# (This command consistently failed in the development environment)
flask run --port 5001
```

### Frontend
```bash
# Navigate to the frontend directory
cd frontend

# Install dependencies
npm install

# Set the backend API URL
# Create a .env.local file in the frontend directory with this content:
# NEXT_PUBLIC_API_URL=http://localhost:5001

# Run the Next.js development server
# (This command also failed in the development environment)
npm run dev
```

---

## API Documentation

### Predict Image

-   **Endpoint:** `/api/predict`
-   **Method:** `POST`
-   **Body:** `multipart/form-data`
    -   `file`: The image file to be analyzed.
-   **Success Response (200):**
    ```json
    {
      "prediction": "FAKE",
      "confidence": 0.92,
      "heatmap": "data:image/jpeg;base64,..."
    }
    ```
-   **Error Response (400, 500):**
    ```json
    {
      "error": "Error message describing the issue."
    }
    ```
