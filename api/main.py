import traceback
import sys
from pathlib import Path

import cv2
import numpy as np
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.Visualization import save_overlay_image
from src.Inference import DamagePredictor, PartPredictor
from src.FeatureExtractor import FeatureExtractor, FeatureAggregator
from src.EstimateModel import EstimateModel


app = FastAPI(title = "Car Damage Estimate API")

STATIC_DIR = PROJECT_ROOT / "static"
RESULT_DIR = STATIC_DIR / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Extract damage masks
damage_predictor = DamagePredictor(
    weight_paths={
        "Scratched": "weights/damage/[DAMAGE][Scratched_0]Unet.pt",
        "Separated": "weights/damage/[DAMAGE][Separated_1]Unet.pt",
        "Crushed": "weights/damage/[DAMAGE][Crushed_2]Unet.pt",
        "Breakage": "weights/damage/[DAMAGE][Breakage_3]Unet.pt"
    },
    size=256
)

# Extract part mask
part_predictor =PartPredictor(
    weight_path="weights/part/[PART]Unet.pt",
    size=256
)
# Extract features
feature_extractor = FeatureExtractor()
# Features aggregation
feature_aggregator = FeatureAggregator()

# Estimate cost by features
estimate_model = EstimateModel(
    labor_model_path="weights/estimate/pure_labor_regressor.cbm",
    paint_classifier_path="weights/estimate/paint_classifier.cbm",
    paint_regressor_path="weights/estimate/paint_regressor.cbm",
    part_classifier_path="weights/estimate/part_classifier.cbm",
    part_regressor_path="weights/estimate/part_regressor.cbm",
    used_features_path="weights/estimate/used_features_3models.csv",
    paint_threshold=0.5,
    part_threshold=0.5,
)


def read_upload_image(file_bytes):
    np_arr = np.frombuffer(file_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise ValueError("[Error] Image is unavailable. It couldn't be read")
    
    return image_bgr

@app.get("/")
def health_check():
    return{
        "status":"ok",
        "message":"Car Damage Estimate API is running"
    }

@app.post("/predict-estimate")
async def predict_estimate(
    image : UploadFile = File(...),
    manufacturer : str = Form("Unknown"),
    car_size : int = Form(0),
    # year : int = Form(0)
):
    try:
        file_bytes = await image.read()
        image_bgr = read_upload_image(file_bytes)

        damage_masks = damage_predictor.predict(image_bgr)
        part_mask = part_predictor.predict(image_bgr)

        overlay_filename, overlay_path = save_overlay_image(
            image_bgr=image_bgr,
            damage_masks=damage_masks,
            part_mask=part_mask,
            output_dir=str(RESULT_DIR)
        )

        overlay_url = f"/static/results/{overlay_filename}"        

        car_info = {
            "manufacturer":manufacturer,
            "car_size":car_size
            # "year":year
        }

        image_features = feature_extractor.extract(
            damage_masks=damage_masks,
            part_mask=part_mask,
            car_info=car_info
        )

        features = feature_aggregator.aggregate([image_features])
        estimate = estimate_model.predict(features)

        return {
            "success": True,
            "estimate":estimate,
            "summary": {
                "main_damage_type": features.get("main_damage_type"),
                "main_damaged_part": features.get("main_damaged_part"),
                "total_damage_area_ratio": features.get("total_damage_area_ratio_sum"),
            },
            "overlay_image_url": overlay_url,
            "features": features,
        }
    
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error":str(e)
            }
        )
    
@app.post("/predict-estimate-multi")
async def predict_estimate_multi(
    image1: UploadFile = File(...),
    image2: Optional[UploadFile] = File(None),
    image3: Optional[UploadFile] = File(None),
    image4: Optional[UploadFile] = File(None),
    manufacturer: str = Form("Unknown"),
    car_size: int = Form(0),
):
    try:
        images = [img for img in [image1, image2, image3, image4] if img is not None]

        if len(images) == 0:
            raise ValueError("At least one image is required")

        car_info = {
            "manufacturer": manufacturer,
            "car_size": car_size
        }

        image_features_list = []
        overlay_images = []

        for image in images:
            file_bytes = await image.read()
            image_bgr = read_upload_image(file_bytes)

            damage_masks = damage_predictor.predict(image_bgr)
            part_mask = part_predictor.predict(image_bgr)

            overlay_filename, overlay_path = save_overlay_image(
                image_bgr=image_bgr,
                damage_masks=damage_masks,
                part_mask=part_mask,
                output_dir=str(RESULT_DIR)
            )

            overlay_url = f"/static/results/{overlay_filename}"

            image_features = feature_extractor.extract(
                damage_masks=damage_masks,
                part_mask=part_mask,
                car_info=car_info
            )

            image_features_list.append(image_features)

            overlay_images.append({
                "filename": image.filename,
                "overlay_image_url": overlay_url,
                "main_damage_type": image_features.get("main_damage_type"),
                "main_damaged_part": image_features.get("main_damaged_part"),
                "total_damage_area_ratio": image_features.get("total_damage_area_ratio"),
            })

        features = feature_aggregator.aggregate(image_features_list)
        estimate = estimate_model.predict(features)

        return {
            "success": True,
            "num_images": len(images),
            "estimate": estimate,
            "summary": {
                "main_damage_type": features.get("main_damage_type"),
                "main_damaged_part": features.get("main_damaged_part"),
                "total_damage_area_ratio": features.get("total_damage_area_ratio_sum"),
            },
            "overlay_images": overlay_images,
            "features": features,
        }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )