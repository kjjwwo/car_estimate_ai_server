import numpy as np

DAMAGE_LABELS = ["Scratched", "Separated", "Crushed", "Breakage"]
PART_LABELS = {
    0:"background",
    1:"bumper",
    2:"body_panel_top",
    3:"side_panel",
    4:"glass_and_light",
    5:"wheels"
}

class FeatureExtractor:
    def __init__(self): 
        pass

    def _safe_ratio(self, numerator, denominator):  
        if denominator == 0:
            return 0.0
        return float(numerator) / float(denominator)

    def extract(self, damage_masks, part_mask, car_info):
        # damage_masks = {"Scratched":np.ndarray[H,W], "Separated":np.ndarray[H,W],
        #                  "Crushed":np.ndarray[H,W], "Breakage":np.ndarray[H,W]}
        # part_mask = np.ndarray[H,W], class index mask

        features = {}
        h,w = part_mask.shape
        total_pixels = h * w

        # 1. car_info
        features["manufacturer"] = car_info.get("manufacturer", "Unknown")
        features["car_size"] = car_info.get("car_size", 0) # 0 is Unknown, (1,2,3,4) size available
        # manufacturer  : "차량 제조사"
        # car_size      : "차량 크기"

        # 2. damage mask
        total_damage_mask = np.zeros_like(part_mask, dtype=np.uint8)

        for damage_name, damage_mask in damage_masks.items():
            damage_name = damage_name.lower()
            binary_damage = (damage_mask > 0).astype(np.uint8)
            
            damage_area = binary_damage.sum()
            features[f"{damage_name}_area_ratio"] = self._safe_ratio(damage_area, total_pixels)
        
            total_damage_mask = np.maximum(total_damage_mask, binary_damage)
        # scratched_area_ratio : "전체 이미지 내 scratched 비중"
        # separated_area_ratio : "전체 이미지 내 separated 비중"
        # crushed_area_raio    : "전체 이미지 내 crushed 비중"
        # breakage_area_ratio  : "전체 이미지 내 breakage 비중"
        
        features["total_damage_area_ratio"]=self._safe_ratio(total_damage_mask.sum(), total_pixels)
        # total_damage_area_ratio   : "전체 이미지 내 damage 비중"

        # 3. part x total damage overlap
        for part_id, part_name in PART_LABELS.items():
            if part_id == 0:    # background ignore
                continue

            part_region = (part_mask == part_id)
            part_area = part_region.sum()
            damaged_part_area = np.logical_and(total_damage_mask > 0, part_region).sum()

            features[f"{part_name}_area_ratio"] = self._safe_ratio(part_area, total_pixels)
            # bumper_area_ratio     : "전체 이미지에서 부품 비율" <= leakage 방지를 위해 뺄 예정
            features[f"{part_name}_damage_ratio"] = self._safe_ratio(damaged_part_area, part_area)
            # bumper_damage_ratio   : "부품 이미지 내에서 파손 비율" 
            features[f"{part_name}_damage_image_ratio"] = self._safe_ratio(damaged_part_area, total_pixels)
            # bumper_damage_image_ratio : "전체 이미지에서 부품 내부 파손 비율" <= 이건 꼭 필요하진 않을 듯
        
        # 4. part x damage type overlap
        for damage_name, damage_mask in damage_masks.items():
            damage_name = damage_name.lower()
            binary_damage = (damage_mask > 0).astype(np.uint8)

            for part_id, part_name in PART_LABELS.items():
                if part_id == 0:
                    continue

                part_region = part_mask == part_id
                overlap = np.logical_and(binary_damage, part_region).sum()
                features[f"{damage_name}_{part_name}_overlap_ratio"] = self._safe_ratio(
                    overlap,
                    total_pixels
                )
            
        # 5. Representative Damage and Part
        damage_area_dict = {
            name: int((mask > 0).sum()) for name, mask in damage_masks.items()
        }
        if max(damage_area_dict.values()) > 0:
            features["main_damage_type"] = max(
                damage_area_dict,
                key=damage_area_dict.get
            )
        else:
            features["main_damage_type"] = "None"
        
        part_damage_dict = {}
        for part_id, part_name in PART_LABELS.items():
            if part_id == 0:
                continue
            part_region = part_mask == part_id
            damaged_area = np.logical_and(total_damage_mask > 0, part_region).sum()
            part_damage_dict[part_name] = int(damaged_area)

        if max(part_damage_dict.values()) > 0:
            features["main_damaged_part"] = max(
                part_damage_dict,
                key=part_damage_dict.get
            )
        else:
            features["main_damaged_part"] = "None"

        return features
    

class FeatureAggregator:
    def __init__(self, damage_threshold=0.01, part_threshold=0.01):
        self.damage_threshold = damage_threshold
        self.part_threshold = part_threshold
        self.manufacturer_map = {
            "현대": "KOREA",
            "기아": "KOREA",
            "한국GM": "KOREA",
            "쌍용": "KOREA",
            "르노삼성": "KOREA",

            "BMW": "GERMANY",
            "BENZ": "GERMANY",
            "AUDI": "GERMANY",
            "VOLKSWAGEN": "GERMANY",

            "CHRYSLER": "USA",
            "FORD": "USA",
            "JEEP": "USA",
            "CADILLAC": "USA",

            "LANDROVER": "UK",

            "TOYOTA": "JAPAN",
            "NISSAN": "JAPAN",

            "VOLVO": "SWEDEN",            
        }
        self.damage_area_keys = [
            "scratched_area_ratio",
            "separated_area_ratio",
            "crushed_area_ratio",
            "breakage_area_ratio",
            "total_damage_area_ratio"
        ]
        self.part_damage_keys = [
            "bumper_damage_ratio",
            "body_panel_top_damage_ratio",
            "side_panel_damage_ratio",
            "glass_and_light_damage_ratio",
            "wheels_damage_ratio"
        ]
        self.part_area_keys = [
            "bumper_area_ratio",
            "body_panel_top_area_ratio",
            "side_panel_area_ratio",
            "glass_and_light_area_ratio",
            "wheels_area_ratio",
        ]
        self.damage_type_keys = [
            "scratched_area_ratio",
            "separated_area_ratio",
            "crushed_area_ratio",
            "breakage_area_ratio"
        ]
        self.categorical_keys = {
            "case_id",
            "image_name",
            "manufacturer",
            "manufacturer_group",
            "main_damage_type",
            "main_damaged_part",
        }

    def aggregate(self, feature_list):
        # feature_list = Case 별 각 이미지들의 FeatureExtractor.extract() 결과
        if not feature_list:
            return {}
        
        aggregated = {}

        # 1. 차량 정보 (car info)
        manufacturer = feature_list[0].get("manufacturer", "Unknown")
        aggregated["manufacturer"] = manufacturer
        aggregated["manufacturer_group"] = self.manufacturer_map.get(manufacturer, "OTHER")
        aggregated["car_size"] = feature_list[0].get("car_size", 0)

        # 2. numeric feature aggregation
        numeric_keys = self._get_numeric_keys(feature_list)
        for key in numeric_keys:
            if self._should_exclude(key):
                continue
            
            values = []
            for features in feature_list:
                value = features.get(key, 0.0)
                try:
                    values.append(float(value))
                except (TypeError, ValueError):
                    values.append(0.0)

            if self._is_damage_area_feature(key):
                aggregated[f"{key}_sum"] = float(np.sum(values))
            elif self._is_part_damage_feature(key):
                aggregated[f"{key}_max"] = float(np.max(values))
            elif self._is_overlap_feature(key):
                aggregated[f"{key}_max"] = float(np.max(values))
            else:
                pass
        # scratched_area_ratio_sum
        # bumper_damage_ratio_max
        # scratched_bumper_overlap_ratio_max
        # total_damage_area_ratio_sum

        # 3. categorical representative feature
        aggregated["main_damage_type"] = self._majority_vote(
            [features.get("main_damage_type", "None") for features in feature_list]
        )
        aggregated["main_damaged_part"] = self._majority_vote(
            [features.get("main_damaged_part", "None") for features in feature_list]
        )      

        # 4. 그외 feature engineering
        self._add_damage_type_count(aggregated)
        # damage_type_count : 손상 종류가 개수
        self._add_part_damage_flags(aggregated)
        # bumper_damaged : 손상 되었는지 1/0으로 표기
        self._add_damaged_part_count(aggregated)
        # damaged_part_count : 파손된 부위 개수

        return aggregated
    
    def _get_numeric_keys(self, feature_list):
        keys = set()
        
        for features in feature_list:
            for key, value in features.items():
                if key in self.categorical_keys:
                    continue

                if isinstance(value, (int, float, np.integer, np.floating)):
                    keys.add(key)
        
        return sorted(keys)

    def _should_exclude(self, key):
        if key == "num_images":
            return True
        if key in self.part_area_keys:
            return True
        if key.endswith("_damage_image_ratio"):
            return True
        if key in ["car_size"]:
            return True
        return False
    
    def _is_damage_area_feature(self, key):
        return key in self.damage_area_keys

    def _is_part_damage_feature(self, key):
        return key in self.part_damage_keys

    def _is_overlap_feature(self, key):
        return key.endswith("_overlap_ratio")

    def _add_damage_type_count(self, aggregated):
        count = 0
        for key in self.damage_type_keys:
            feature_name = f"{key}_sum"
            if aggregated.get(feature_name, 0.0) > self.damage_threshold:
                count += 1
        aggregated["damage_type_count"] = count

    def _add_part_damage_flags(self, aggregated):
        for key in self.part_damage_keys:
            part_name = key.replace("_damage_ratio", "")
            feature_name = f"{key}_max"
            aggregated[f"{part_name}_damaged"] = int(
                aggregated.get(feature_name, 0.0) > self.part_threshold
            )
    
    def _add_damaged_part_count(self, aggregated):
        flag_keys = [
            "bumper_damaged",
            "body_panel_top_damaged",
            "side_panel_damaged",
            "glass_and_light_damaged",
            "wheels_damaged",
        ]
        aggregated["damaged_part_count"] = int(
            sum(aggregated.get(key, 0) for key in flag_keys)
        )
    
    def _majority_vote(self, values):
        counts = {}
        for value in values:
            if value is None:
                value = "None"
            counts[value] = counts.get(value, 0) + 1
        return max(counts, key=counts.get)       
