import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import io
import random

def generate_local_critique(stealth_data, style_data):
    dominant = style_data["dominant_style"]
    score = stealth_data["invisibility_index"]
    
    # Vocabulary blocks mapping out your specific tone
    tech_cynicism = [
        "The automated city-grid logging nodes are throwing a party right now.",
        "Palantir's tracking matrices didn't even have to try with this one.",
        "You're rendering on their display buffers like a neon sign in a blackout.",
        "An absolute textbook tracking signature for standard CCTV edge-compute nodes."
    ]
    
    fashion_takes = {
        "oversized high-fashion streetwear": [
            "Those structural drapes and exaggerated layers make for a great silhouette to humans, but to the machine, it's just a massive collection of high-contrast pixel weights.",
            "Trying to look casual in oversized street cuts while providing the neural net a massive, highly predictable bounding box to calculate.",
            "The volume of the outfit screams archival aesthetic, but the telemetry metrics scream prime algorithmic target."
        ],
        "cyberpunk techwear silhouette": [
            "Irony at its finest: wearing techwear designed to hide from drones, yet the geometric lines are ticking every structural box in the dataset.",
            "All those asymmetric straps and matte fabrics just gave the feature extractor a uniquely crisp edge to lock onto."
        ],
        "minimalist monochrome tailoring": [
            "Clean lines, low contrast. You almost slipped through the feature mapping, but the sharp silhouette boundary gave you away.",
            "Walking around looking like a high-end architecture magazine cover is a fast way to get cataloged by spatial sensors."
        ],
        "classic casual default civilian look": [
            "The default civilian look is classic for a reason—it blends seamlessly into the urban white noise.",
            "Zero styling effort means zero unique feature anchors for the tracking grid to build a persistent signature on."
        ]
    }
    
    closing_thoughts = [
        f"Result? A brutal {score}/100 invisibility index. You're effectively a corporate data asset at this point.",
        f"Scoring a low {score}/100 on stealth. The tracking infrastructure owns that visual profile now.",
        f"With an invisibility index of just {score}%, you aren't walking through the city—you're just updating their databases in real time."
    ]
    
    # Fallback for styles not explicitly detailed in the dict
    style_commentary = fashion_takes.get(
        dominant, 
        ["The geometric silhouette profile provides crisp contrast values directly to the tracking matrix."]
    )
    
    # Synthesize the response deterministically but with structural variety
    critique = f"{random.choice(tech_cynicism)} {random.choice(style_commentary)} {random.choice(closing_thoughts)}"
    return critique

app = FastAPI(title="Surveillance Flâneur API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models
yolo_model = YOLO("yolov8n.pt")

# Using a lightweight, fast CLIP variant
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Your curated style reference array
STYLE_ARCHETYPES = [
    "cyberpunk techwear silhouette",
    "minimalist monochrome tailoring",
    "oversized high-fashion streetwear",
    "vintage archival workwear",
    "gorpcore outdoor functional layers",
    "classic casual default civilian look"
]

def analyze_style_and_stealth(img_bytes):
    # --- 1. YOLO Counter-Surveillance Scan ---
    nparr = np.frombuffer(img_bytes, np.uint8)
    cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    results = yolo_model(cv_img)
    person_scores = []
    
    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) == 0:
                person_scores.append(float(box.conf[0]))

    # --- 2. CLIP Style Matrix Scan ---
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    inputs = clip_processor(text=STYLE_ARCHETYPES, images=pil_img, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    style_analysis = {STYLE_ARCHETYPES[i]: float(probs[i]) for i in range(len(STYLE_ARCHETYPES))}
    dominant_style = max(style_analysis, key=style_analysis.get)

    # --- 3. Local Personalization Modifiers ---
    if not person_scores:
        max_conf = 0.0
        invisibility_index = 100
    else:
        max_conf = max(person_scores)
        
        # Calculate raw visibility
        base_invis = (1.0 - max_conf) * 100
        
        # Apply local aesthetic bonuses
        # If you wear structural layers or techwear, simulate geometric distortion
        if dominant_style == "cyberpunk techwear silhouette":
            base_invis += 45  # Massive anti-drone geometry bonus
        elif dominant_style == "oversized high-fashion streetwear":
            base_invis += 30  # Silhouette volume breakup bonus
        elif dominant_style == "minimalist monochrome tailoring":
            base_invis += 15  # Low-contrast visual blend bonus
            
        # Clamp between 0 and 99 so you only hit 100 if you completely vanish
        invisibility_index = min(int(base_invis), 99)

    # Re-evaluate status based on your personalized scaling laws
    if invisibility_index > 75: status = "Urban Legend"
    elif invisibility_index > 45: status = "Background Noise"
    elif invisibility_index > 20: status = "Main Character (Tracked)"
    else: status = "Corporate Data Asset"

    return {
        "stealth_metrics": {
            "invisibility_index": invisibility_index,
            "max_detector_confidence": round(max_conf, 2),
            "status": status
        },
        "style_matrix": {
            "dominant_style": dominant_style,
            "confidence_distribution": {k: round(v, 3) for k, v in style_analysis.items()}
        }
    }

@app.post("/scan")
async def scan_fit(file: UploadFile = File(...)):
    contents = await file.read()
    analysis = analyze_style_and_stealth(contents)
    
    # Process local text synthesis without API cost
    critique = generate_local_critique(analysis["stealth_metrics"], analysis["style_matrix"])
    analysis["voice_critique"] = critique
    
    return {
        "success": True,
        "data": analysis
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)