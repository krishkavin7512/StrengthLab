"""Paths, constants, mix-ingredient metadata and Indian Standard rules for concrete grades."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "concrete_compressive_strength.csv"
ARTIFACTS = ROOT / "artifacts"
FRONTEND = ROOT / "frontend"
SEED = 7
SPLIT = (0.70, 0.15, 0.15)

TARGET = "strength"
RAW = ["cement", "slag", "flyash", "water", "sp", "coarse", "fine", "age"]
COLUMN_NAMES = RAW + [TARGET]  # UCI column order

INGREDIENTS = [
    {"key": "cement", "label": "Cement", "unit": "kg/m³", "min": 100, "max": 550, "step": 5, "default": 320},
    {"key": "slag", "label": "Blast-furnace slag", "unit": "kg/m³", "min": 0, "max": 360, "step": 5, "default": 0},
    {"key": "flyash", "label": "Fly ash", "unit": "kg/m³", "min": 0, "max": 200, "step": 5, "default": 0},
    {"key": "water", "label": "Water", "unit": "kg/m³", "min": 120, "max": 250, "step": 1, "default": 185},
    {"key": "sp", "label": "Superplasticizer", "unit": "kg/m³", "min": 0, "max": 32, "step": 0.5, "default": 4},
    {"key": "coarse", "label": "Coarse aggregate", "unit": "kg/m³", "min": 800, "max": 1150, "step": 5, "default": 1000},
    {"key": "fine", "label": "Fine aggregate", "unit": "kg/m³", "min": 590, "max": 1000, "step": 5, "default": 760},
    {"key": "age", "label": "Curing age", "unit": "days", "min": 1, "max": 365, "step": 1, "default": 28},
]
ING_BY_KEY = {i["key"]: i for i in INGREDIENTS}
DEFAULT_MIX = {i["key"]: i["default"] for i in INGREDIENTS}
LABEL = {i["key"]: i["label"] for i in INGREDIENTS} | {
    TARGET: "Compressive strength", "log_age": "ln(age)", "wc": "Water / cement", "wb": "Water / binder"}

# IS 456:2000 grades and IS 10262:2019 target mean strength f'ck = fck + 1.65 s,
# with the assumed standard deviation s from IS 10262:2019 Table 2.
GRADES = [
    {"grade": "M20", "fck": 20, "s": 4.0},
    {"grade": "M25", "fck": 25, "s": 4.0},
    {"grade": "M30", "fck": 30, "s": 5.0},
    {"grade": "M35", "fck": 35, "s": 5.0},
    {"grade": "M40", "fck": 40, "s": 5.0},
]
for g in GRADES:
    g["target"] = round(g["fck"] + 1.65 * g["s"], 2)
GRADE_BY_NAME = {g["grade"]: g for g in GRADES}
MAIN_GRADE = "M30"

# IS 456:2000 Table 5 (reinforced concrete): durability limits by exposure condition.
# Cement content includes mineral admixtures (fly ash, slag).
EXPOSURE = [
    {"exposure": "Mild", "min_cement": 300, "max_wc": 0.55, "min_grade": "M20"},
    {"exposure": "Moderate", "min_cement": 300, "max_wc": 0.50, "min_grade": "M25"},
    {"exposure": "Severe", "min_cement": 320, "max_wc": 0.45, "min_grade": "M30"},
    {"exposure": "Very severe", "min_cement": 340, "max_wc": 0.45, "min_grade": "M35"},
    {"exposure": "Extreme", "min_cement": 360, "max_wc": 0.40, "min_grade": "M40"},
]

# Rough embodied-CO2 factors (kg CO2 per kg of material) for an order-of-magnitude readout only.
CO2_FACTORS = {"cement": 0.87, "slag": 0.07, "flyash": 0.01, "sp": 1.9, "coarse": 0.005, "fine": 0.005, "water": 0.0003}

PRESETS = {
    "site_m20": {"label": "Ordinary site mix", "mix": {"cement": 300, "slag": 0, "flyash": 0, "water": 192, "sp": 0,
                                                        "coarse": 1040, "fine": 780, "age": 28}},
    "hp_m40": {"label": "High-performance + superplasticizer", "mix": {"cement": 450, "slag": 0, "flyash": 0, "water": 160,
                                                                     "sp": 10, "coarse": 1000, "fine": 760, "age": 28}},
    "green": {"label": "Green mix (slag + fly ash)", "mix": {"cement": 220, "slag": 140, "flyash": 90, "water": 175,
                                                             "sp": 8, "coarse": 950, "fine": 760, "age": 28}},
}
