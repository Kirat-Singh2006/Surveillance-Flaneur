# Surveillance-Flâneur-
A lightweight, multi-modal counter-surveillance aesthetic engine designed to evaluate human style architectures against automated corporate tracking grids. Built entirely to run locally on edge hardware with zero API token latency.
# graph TD
    A[streetfit.jpg] --> B[YOLOv8 nano]
    A --> C[CLIP ViT-B/32]
    B -->|Human Bounding Box Confidence| D[Stealth Metrics Engine]
    C -->|Zero-Shot Style Archetype Matching| E[Style Matrix Engine]
    D --> F[Deterministic Aesthetic Modifiers]
    E --> F
    F --> G[Dystopian Voice Critique]
# Features

* **YOLOv8 Tracking Scan:** Leverages an edge-optimized computer vision model to calculate real-time machine tracking confidence.
* **CLIP Style Matrix:** Utilizes a multi-modal text/image model to analyze outfit telemetry against curated fashion archetypes via zero-shot classification.
* **Aesthetic Compensation Modifiers:** Emulates geometric silhouette distortion, granting tactical bonuses for techwear structure or oversized layers.
* **Zero-Cost Telemetry Synthesis:** Features a deterministic, zero-latency local voice engine that delivers cynical, tech-dystopian feedback without relying on external LLM APIs.
* **Adaptive Dynamic UI:** A low-light, command-line terminal interface built with pure semantic HTML/CSS that resolves networking paths automatically across your local network.

# The Tech Stack

* **Backend:** FastAPI, Uvicorn, PyTorch
* **Computer Vision:** Ultralytics YOLOv8n, HuggingFace Transformers (OpenAI CLIP ViT-B/32)
* **Frontend:** Vanilla JS, HTML5, CSS3 (Material-inspired dark matrix)

# Deployment

1. **Fire up the localized edge engine:**
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
