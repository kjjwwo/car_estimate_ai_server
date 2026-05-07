import cv2
import numpy as np
import torch

from src.Models import Unet

DAMAGE_LABELS = ["Scratched", "Separated", "Crushed", "Breakage"]
PART_LABELS = {
    0:"Background",
    1:"Bumper",
    2:"Body Panel(Top)",
    3:"Side Panel",
    4:"Glass & Light",
    5:"Wheels"
}

class BaseSegmentationPredictor:
    def __init__(self, num_classes, weight_path, device=None, size=256):
        self.size = size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        wrapper = Unet(
            encoder="resnet34",
            pre_weight=None, # 왜 Pre-weight가 imagenet이 아니지? 학습할 때는 imagenet으로 했는데
            num_classes=num_classes
        )

        self.model = wrapper.model
        self.model.load_state_dict(
            torch.load(weight_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, image_bgr):
        if image_bgr is None:
            raise ValueError("image_bgr is None")
        
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image_rgb, (self.size, self.size))

        resized = resized.astype(np.float32) / 255.0
        resized = resized.transpose(2,0,1)    # [H,W,C] -> [C,H,W]

        tensor = torch.from_numpy(resized).unsqueeze(0).to(self.device)    # unsqueeze 하면 0번째 껍데기 생성인데 왜 함?
        return tensor

    def predict_mask(self, image_bgr):
        x = self.preprocess(image_bgr)

        with torch.inference_mode():
            logits = self.model(x)
            pred = torch.argmax(logits, dim=1)

        mask = pred.squeeze(0).cpu().numpy().astype(np.uint8) # unsqueeze 했다가 다시 squeeze 하는 이유는?
        return mask
    

class DamagePredictor:
    def __init__(self, weight_paths, device=None, size=256):
        self.predictors = {}

        for label_name, path in weight_paths.items():
            self.predictors[label_name] = BaseSegmentationPredictor(
                num_classes=2,
                weight_path=path,
                device=device,
                size=size
            )
        
    def predict(self, image_bgr):
        damage_masks = {}

        for label_name, predictor in self.predictors.items():   # items()를 사용하는 이유는? 그냥 없이 쓰면 안되나
            mask = predictor.predict_mask(image_bgr)
            damage_masks[label_name] = (mask==1).astype(np.uint8)   # 이 부분은 겹치는 것 아닌가?
        
        return damage_masks

class PartPredictor:
    def __init__(self, weight_path, device=None, size=256):
        self.predictor = BaseSegmentationPredictor(
            num_classes=6,
            weight_path=weight_path,
            device=device,
            size=size
        )

    def predict(self, image_bgr):
        part_mask = self.predictor.predict_mask(image_bgr)
        return part_mask        
