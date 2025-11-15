# AI-Based RF Signal Classification

**By:** Derrick Mucheru, Valentine Ndichu, Achieng Mary  
**Supervised by:** Prof. Ciira Maina

This repository contains all code for our project **AI-Based RF Signal Classification**, including dataset generation, model training, evaluation, and real-time inference using RTL-SDR hardware.

---

## Project Overview

We develop a multi-label RF signal classifier capable of detecting:

- AM
- FM
- Bluetooth
- WiFi
- Zigbee
- Radar
- LTE
- Noise

The model uses spectrograms generated via STFT and a modified ResNet-18 optimized for multi-label classification using sigmoid activation and regression-style loss.

---

## Repository Structure

```
dataset/              # MATLAB dataset generator + sample data
model/                # Training code, inference scripts, saved models
rtlsdr_realtime/      # Real-time RTL-SDR acquisition + classification
gui_demo/             # Streamlit GUI for live demo visualization
```

---

## Model Summary

- **Base Model:** ResNet-18 (pretrained)
- **Input:** 257×39 STFT spectrogram → 224×224 RGB
- **Output:** 8 multi-label probabilities
- **Training Epochs:** 20
- **Optimizer:** Adam (1e-4)
- **Validation Split:** 70/15/15

### Independent Test Performance

| Metric | Value |
|--------|-------|
| Exact Match Accuracy | 91.0% |
| Avg F1 Score | 0.98 |
| Hamming Loss | 0.0112 |
| Inference Speed | 62.5 ms/sample |

---

## 📡 Real-Time RTL-SDR Pipeline

1. Capture IQ data with RTL-SDR
2. Apply filtering + STFT
3. Convert to spectrogram (224×224)
4. Run model inference
5. Display probabilities + detected signals

All code is inside: `rtlsdr_realtime/`

---

## Running Real-Time Demo

### Install dependencies:

```bash
pip install -r requirements.txt
```

### Run live classification:

```bash
python rtlsdr_realtime/realtime_demo.py
```

### Run graphical dashboard:

```bash
streamlit run gui_demo/gui.py
```

---

## References

*(List the same references provided in your project summary)*

---

## License

MIT License © 2025

---

## Demo Video

*(Video link to be added)*
