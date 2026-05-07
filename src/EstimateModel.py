import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, CatBoostClassifier

class EstimateModel:
    def __init__(
        self,
        labor_model_path=None,
        paint_classifier_path=None,
        paint_regressor_path=None,
        part_classifier_path=None,
        part_regressor_path=None,
        used_features_path=None,
        paint_threshold=0.5,
        part_threshold=0.5,
        use_rule_based_if_missing=True,
    ):
        # 도장 및 부품 필요 유무 판단 시 사용할 thresholds
        self.paint_threshold = paint_threshold
        self.part_threshold = part_threshold
        self.use_rule_based_if_missing = use_rule_based_if_missing

        self.used_features = None
        if used_features_path is not None:
            self.used_features = pd.read_csv(used_features_path)["used_feature"].tolist()

        self.labor_model = self._load_regressor(labor_model_path)               # 도장비 제외 공임비 회귀 분석
        self.paint_classifier = self._load_classifier(paint_classifier_path)    # 도장 필요 유무 판단
        self.paint_regressor = self._load_regressor(paint_regressor_path)       # 도장 필요 시 도장비 회귀 분석
        self.part_classifier = self._load_classifier(part_classifier_path)      # 부품 필요 유무 판단
        self.part_regressor = self._load_regressor(part_regressor_path)         # 부품 필요 시 부품비 회귀 분석

        # 모든 CatBoost 존재 확인 + 사용 특징 파일 유무 확인
        self.model_ready = all([
            self.labor_model is not None,
            self.paint_classifier is not None,
            self.paint_regressor is not None,
            self.part_classifier is not None,
            self.part_regressor is not None,
            self.used_features is not None,
        ])
    
        if not self.model_ready and not self.use_rule_based_if_missing:
            raise ValueError("[EstimateModel] Cost model files are missing.")

    # CatBoost 생성
    def _load_regressor(self, path):
        if path is None:
            return None
        model = CatBoostRegressor()
        model.load_model(path)
        return model

    def _load_classifier(self, path):
        if path is None:
            return None
        model = CatBoostClassifier()
        model.load_model(path)
        return model

    # 입력 데이터 처리
    def _make_input_df(self, features):
        df = pd.DataFrame([features])

        # 학습 때 사용한 feature가 없으면 0 또는 None으로 채움
        for col in self.used_features:
            if col not in df.columns:
                df[col] = 0

        # 학습 때 사용하지 않은 feature 제거 + 순서 정렬
        df = df[self.used_features].copy()

        # CatBoost categorical feature 안전 처리
        for col in ["main_damage_type", "main_damaged_part", "manufacturer_group"]:
            if col in df.columns:
                df[col] = df[col].fillna("None").astype(str)

        return df


    def predict(self, features):
        if not self.model_ready:
            return self.rule_based_estimate(features)

        X = self._make_input_df(features)

        pure_labor_cost = self._predict_log_regression(self.labor_model, X)

        paint_cost, paint_proba = self._predict_two_stage(
            classifier=self.paint_classifier,
            regressor=self.paint_regressor,
            X=X,
            threshold=self.paint_threshold,
        )

        part_cost, part_proba = self._predict_two_stage(
            classifier=self.part_classifier,
            regressor=self.part_regressor,
            X=X,
            threshold=self.part_threshold,
        )

        subtotal_cost = pure_labor_cost + paint_cost + part_cost
        vat = subtotal_cost * 0.10
        total_cost = subtotal_cost + vat

        return {
            "pure_labor_cost": int(round(pure_labor_cost)),
            "paint_cost": int(round(paint_cost)),
            "part_cost": int(round(part_cost)),
            "subtotal_cost": int(round(subtotal_cost)),
            "vat": int(round(vat)),
            "total_cost": int(round(total_cost)),
            "paint_needed_proba": float(paint_proba),
            "part_needed_proba": float(part_proba),
        }
    
    def _predict_log_regression(self, model, X):
        pred_log = model.predict(X)[0]
        pred = np.expm1(pred_log)
        pred = np.clip(pred, 0, None)
        return float(pred)

    def _predict_two_stage(self, classifier, regressor, X, threshold):
        proba = classifier.predict_proba(X)[0][1]

        if proba < threshold:
            return 0.0, float(proba)

        pred_log = regressor.predict(X)[0]
        pred = np.expm1(pred_log)
        pred = np.clip(pred, 0, None)

        return float(pred), float(proba)
    



#######################################################################3
### estimate 모델 설계 필요, 모델 없으면 임시로 이거 사용
    def rule_based_estimate(self, features):
        ### 임시 견적 계산용
        base_cost = 100000

        damage_cost = 0
        damage_cost += features.get("scratched_area_ratio_sum", 0) * 2000000
        damage_cost += features.get("separated_area_ratio_sum", 0) * 3000000
        damage_cost += features.get("crushed_area_ratio_sum", 0) * 4000000
        damage_cost += features.get("breakage_area_ratio_sum", 0) * 5000000

        part_multiplier = 1.0

        if features.get("main_damaged_part") == "glass_and_light":
            part_multiplier = 1.3
        elif features.get("main_damaged_part") == "wheels":
            part_multiplier = 1.4
        elif features.get("main_damaged_part") == "bumper":
            part_multiplier = 1.1

        car_size = features.get("car_size", 0)

        if car_size == 1:
            size_multiplier = 1.2
        elif car_size == 2:
            size_multiplier = 1.3
        elif car_size == 3:
            size_multiplier = 1.4
        elif car_size == 4:
            size_multiplier = 1.5
        else:
            size_multiplier = 1.0

        subtotal_cost = (base_cost + damage_cost) * part_multiplier * size_multiplier
        vat = subtotal_cost * 0.10
        total_cost = subtotal_cost + vat

        return {
            "pure_labor_cost": int(subtotal_cost * 0.4),
            "paint_cost": int(subtotal_cost * 0.35),
            "part_cost": int(subtotal_cost * 0.25),
            "subtotal_cost": int(subtotal_cost),
            "vat": int(vat),
            "total_cost": int(total_cost),
            "paint_needed_proba": None,
            "part_needed_proba": None,
        }