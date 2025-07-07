# Nexus 2.0

Nexus 2.0 is a reference trading agent focused on XAU/USD 5-minute data.

## Installation
```bash
pip install -r requirements.txt
```

## Data
Place `XAUUSD-5m-2022-Present.parquet` in `data/raw/` and run:
```bash
python preprocess.py
```

## Train Forecast Encoder
```bash
python scripts/train_encoder.py --epochs 20 --batch 64 --seq_len 60 --horizon 12
```

## Reinforcement Learning
Fast debug mode:
```bash
python train_nexus.py --epochs 1 --fast-debug
```
Full training:
```bash
python train_nexus.py --epochs 3
```

## Testing
```bash
pytest -q
```
