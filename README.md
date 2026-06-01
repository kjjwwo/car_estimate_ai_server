# 차량 파손 견적 AI 서버 (Car Damage Estimate AI Server)

![demo](docs/demo.png)

## 1. 프로젝트 개요 (Overview)

본 프로젝트는 차량 파손 이미지를 분석하여 자동으로 수리 견적을 예측하는 AI 추론 서버입니다.

서버는 다음 기능을 수행합니다.

- 차량 손상(Damage) Segmentation
- 차량 부위(Part) Segmentation
- Feature 추출
- 차량 수리 견적 예측
- Segmentation Overlay 시각화 결과 생성

AI 서버는 FastAPI와 PyTorch 기반으로 구현되었습니다.

---

## 2. 프로젝트 구조 (Project Structure)

```text
project/
├─ api/
│  └─ main.py
├─ src/
│  ├─ Inference.py
│  ├─ FeatureExtractor.py
│  ├─ EstimateModel.py
│  ├─ Visualization.py
│  └─ Models.py
├─ weights/
│  ├─ damage/
│  │  ├─ [DAMAGE][Scratch_0]Unet.pt
│  │  ├─ [DAMAGE][Separated_1]Unet.pt
│  │  ├─ [DAMAGE][Crushed_2]Unet.pt
│  │  └─ [DAMAGE][Breakage_3]Unet.pt
│  ├─ part/
│  │  └─ [PART]Unet.pt
│  └─ estimate/
│     ├─ pure_labor_regressor.cbm
│     ├─ paint_classifier.cbm
│     ├─ paint_regressor.cbm
│     ├─ part_classifier.cbm
│     ├─ part_regressor.cbm
│     └─ used_features_3models.csv
├─ static/
│  └─ results/
├─ requirements.txt
└─ README.md
```

---

## 3. 개발 환경 및 요구 사항 (Requirements)

* Python 3.13+
* CUDA 지원 GPU 권장
* PyTorch
* FastAPI
* CatBoost

---

## 4. 설치 방법 (Installation)

필요 패키지 설치:

```bash
pip install -r requirements.txt
```

---

## 5. 모델 Weight 설정 (Weights Setup)

아래 경로에 모델 weight 파일을 배치해야 합니다.

```text
weights/
├─ damage/
├─ part/
└─ estimate/
```

필요 파일 목록:

### Damage Segmentation

* [DAMAGE][Scratch_0]Unet.pt
* [DAMAGE][Separated_1]Unet.pt
* [DAMAGE][Crushed_2]Unet.pt
* [DAMAGE][Breakage_3]Unet.pt

### Part Segmentation

* [PART]Unet.pt

### Estimate Models

* pure_labor_regressor.cbm
* paint_classifier.cbm
* paint_regressor.cbm
* part_classifier.cbm
* part_regressor.cbm
* used_features_3models.csv

---

## 6. 서버 실행 방법 (Run Server)

FastAPI 서버 실행:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Swagger API 문서:

```text
http://127.0.0.1:8000/docs
```

---

## 7. API 사용 방법 (API Usage)

### Endpoint

```text
POST /predict-estimate
```

### Request

multipart/form-data

| Field        | Type   | 설명       |
| ------------ | ------ | -------- |
| image        | file   | 차량 이미지   |
| manufacturer | string | 차량 제조사   |
| car_size     | int    | 차량 크기 분류 |

---

## 8. 응답 예시 (Response Example)

```json
{
  "success": true,
  "estimate": {
    "pure_labor_cost": 120000,
    "paint_cost": 80000,
    "part_cost": 50000,
    "subtotal_cost": 250000,
    "vat": 25000,
    "total_cost": 275000
  },
  "summary": {
    "main_damage_type": "Scratched",
    "main_damaged_part": "bumper",
    "total_damage_area_ratio": 0.032
  },
  "overlay_image_url": "/static/results/example.jpg"
}
```

---

## 9. Overlay 결과 이미지 (Overlay Result)

서버는 segmentation 결과를 원본 이미지에 overlay한 결과 이미지를 생성합니다.

결과 이미지는 아래 경로를 통해 접근할 수 있습니다.

```text
/static/results/{filename}.jpg
```

---

## 10. 현재 한계점 (Current Limitations)

* Glass & Light 클래스 및 전반적인 segmentation 성능이 상대적으로 낮습니다.
* 견적 결과는 AI 예측값이며 실제 정비 견적과 차이가 발생할 수 있습니다.
* 현재 단일 이미지 추론만 지원합니다.
* 여러 이미지 기반 aggregation 기능은 추후 확장 예정입니다.

---
## 아래는 영문 버전 README.md 입니다.
---


# Car Damage Estimate AI Server

## 1. Overview

This project is an AI inference server for vehicle damage analysis and repair cost estimation.

The server performs:

- Damage segmentation
- Vehicle part segmentation
- Feature extraction
- Repair cost estimation
- Segmentation overlay visualization

The AI server is built with FastAPI and PyTorch.

---

## 2. Project Structure

```text
project/
├─ api/
│  └─ main.py
├─ src/
│  ├─ Inference.py
│  ├─ FeatureExtractor.py
│  ├─ EstimateModel.py
│  ├─ Visualization.py
│  └─ Models.py
├─ weights/
│  ├─ damage/
│  │  ├─ [DAMAGE][Scratch_0]Unet.pt
│  │  ├─ [DAMAGE][Separated_1]Unet.pt
│  │  ├─ [DAMAGE][Crushed_2]Unet.pt
│  │  └─ [DAMAGE][Breakage_3]Unet.pt
│  ├─ part/
│  │  └─ [PART]Unet.pt
│  └─ estimate/
│     ├─ pure_labor_regressor.cbm
│     ├─ paint_classifier.cbm
│     ├─ paint_regressor.cbm
│     ├─ part_classifier.cbm
│     ├─ part_regressor.cbm
│     └─ used_features_3models.csv
├─ static/
│  └─ results/
├─ requirements.txt
└─ README.md
```

---

## 3. Requirements

* Python 3.11+
* CUDA capable GPU recommended
* PyTorch
* FastAPI
* CatBoost

---

## 4. Installation

Install required packages:

```bash
pip install -r requirements.txt
```

---

## 5. Weights Setup

Place model weights in the following directories:

```text
weights/
├─ damage/
├─ part/
└─ estimate/
```

Required files:

### Damage Segmentation

* [DAMAGE][Scratch_0]Unet.pt
* [DAMAGE][Separated_1]Unet.pt
* [DAMAGE][Crushed_2]Unet.pt
* [DAMAGE][Breakage_3]Unet.pt

### Part Segmentation

* [PART]Unet.pt

### Estimate Models

* pure_labor_regressor.cbm
* paint_classifier.cbm
* paint_regressor.cbm
* part_classifier.cbm
* part_regressor.cbm
* used_features_3models.csv

---

## 6. Run Server

Run FastAPI server:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Swagger API docs:

```text
http://127.0.0.1:8000/docs
```

---

## 7. API Usage

### Endpoint

```text
POST /predict-estimate
```

### Request

multipart/form-data

| Field        | Type   | Description           |
| ------------ | ------ | --------------------- |
| image        | file   | Vehicle image         |
| manufacturer | string | Vehicle manufacturer  |
| car_size     | int    | Vehicle size category |

---

## 8. Response Example

```json
{
  "success": true,
  "estimate": {
    "pure_labor_cost": 120000,
    "paint_cost": 80000,
    "part_cost": 50000,
    "subtotal_cost": 250000,
    "vat": 25000,
    "total_cost": 275000
  },
  "summary": {
    "main_damage_type": "Scratched",
    "main_damaged_part": "bumper",
    "total_damage_area_ratio": 0.032
  },
  "overlay_image_url": "/static/results/example.jpg"
}
```

---

## 9. Overlay Result

The server generates segmentation overlay images.

Overlay images can be accessed via:

```text
/static/results/{filename}.jpg
```

---

## 10. Current Limitations

* Glass & light segmentation performance is relatively low.
* Estimate values are AI predictions and may differ from actual repair costs.
* Single image inference is currently supported.
* Multiple image aggregation support is under development.

---
