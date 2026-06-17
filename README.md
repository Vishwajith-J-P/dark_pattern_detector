# Dark Pattern Detector

## Overview

Dark Pattern Detector is a Chromium-based web extension that detects potentially manipulative user interface patterns on E-Commerce websites. It uses a fine-tuned DistilBERT model to analyze webpage content and identify common dark patterns such as scarcity, urgency, hidden costs, misdirection, and confirmshaming.

## Components

### Chrome Extension

* Extracts webpage elements
* Sends data to the backend
* Displays analysis results
* Highlights detected dark patterns

### FastAPI Backend

* Receives webpage data
* Runs model inference
* Calculates risk scores
* Returns detection results

### DistilBERT Model

* Classifies webpage text into dark pattern categories
* Produces confidence scores for each prediction

## How It Works

1. User clicks **Scan Page**.
2. The extension extracts visible webpage elements.
3. Data is sent to the FastAPI backend.
4. The DistilBERT model analyzes the content.
5. Detected dark patterns are returned.
6. The extension highlights suspicious elements and displays results.

## Running the Project

### Creating a Virtual Environemnt

Although it's completely optional, it is recommended to install all the python packages in an environment

#### Using Conda

conda create -n dpd python=3.11
conda activate dpd

#### Using Python venv

python -m venv dpd
dpd\Scripts\activate

### Start the Backend

```bash
pip install -r requirements.txt
uvicorn classifier.predict:app --host 127.0.0.1 --port 8000 --reload
```

The backend will run at:

```text
http://127.0.0.1:8000
```

### Load the Extension

1. Open Chrome.
2. Go to `chrome://extensions`.
3. Enable **Developer Mode**.
4. Click **Load Unpacked**.
5. Select the extension folder.

### Use

1. Open a website.
2. Click the DarkLens extension.
3. Press **Scan Page**.
4. View detected dark patterns and risk score.
