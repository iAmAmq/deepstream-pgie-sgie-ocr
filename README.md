# ds_ts_ocr — Truck Inspection Detection with Plate OCR

A config-driven DeepStream 8.0 Python pipeline that detects 21 truck component
classes and runs text recognition **only** on detected license plates.

```
source → streammux → pgie (YOLO) → [tracker] → [sgie / PaddleOCR] → osd → sink
```

| Stage | Model | Runs on |
|---|---|---|
| Primary | `truck_best.onnx` — 21 classes | Every frame |
| Secondary | PaddleOCR recognition | **License_Plate crops only** |

---

## Status

| Component | State |
|---|---|
| Pipeline, config parsing, sinks | Working, verified end to end |
| Custom bbox parser | Working — 2,373 detections on the sample clip |
| License-plate gating | Configured, not yet exercised (see *Verification gaps*) |
| OCR recognition | **Wired but inactive — model not supplied** |

---

## How the license-plate gate works

This is the core requirement, and it needs **no application code**. nvinfer does
it natively via three keys in `configs/config_infer_secondary_ocr.txt`:

```ini
process-mode=2            # operate on cropped objects, not whole frames
operate-on-gie-id=1       # only objects from the primary detector
operate-on-class-ids=5    # only class 5 = License_Plate
```

nvinfer crops and infers on class-5 detections alone. No other class ever reaches
the recognition model — there is no per-frame Python filtering, and no wasted GPU
work on the other 20 classes.

Two extra keys skip plates too small to read:

```ini
input-object-min-width=32
input-object-min-height=16
```

---

## Enabling OCR

The recognition stage needs two files that are **not currently in `model_wts/`**:

1. **PaddleOCR recognition model as ONNX** — e.g. `en_PP-OCRv4_rec_infer` exported
   with `paddle2onnx`. Save as `model_wts/ppocr_rec.onnx`.
2. **Character dictionary** — one character per line. Save as
   `model_wts/ocr_dict.txt`.

Then:

```ini
# configs/ds_ts_ocr_config.txt
[secondary-gie0]
enable=1
```

Check `infer-dims` in `config_infer_secondary_ocr.txt` matches your model —
PP-OCRv4 rec is usually `3;48;320`, older v3 models `3;32;320`. Confirm the
output tensor name too:

```bash
trtexec --onnx=model_wts/ppocr_rec.onnx --skipInference 2>&1 | grep -i output
```

### Why decoding happens in Python

PaddleOCR recognition is CTC-based: the output is `[seq_len, num_chars]`, which
nvinfer cannot turn into a label. The SGIE therefore runs with
`network-type=100` and `output-tensor-meta=1`, and the raw tensor is
greedy-decoded in `sgie_probe()` — collapsing repeats, dropping blanks, and
averaging per-character confidence. Recognised text replaces the object's OSD
label.

---

## Running

```bash
# 1. Build the custom bbox parser (once)
cd nvdsinfer_custom_bbox && make && cd ..

# 2. Run
python3 ds_ts_ocr.py -c configs/ds_ts_ocr_config.txt
```

The first run builds a TensorRT engine from the ONNX (~2 minutes) and caches it
as `model_wts/truck_best.onnx_b1_gpu0_fp16.engine`. Delete that file after
changing `network-mode` or `batch-size`.

```
python3 ds_ts_ocr.py -c FILE [-v]
  -c, --cfg-file FILE   Application config          [required]
  -v, --verbose         Print each recognised plate
```

---

## The custom bbox parser

`truck_best.onnx` is a YOLOv5/v7-style export whose head is split into three
outputs with **no NMS applied**:

| Tensor | Shape | Contents |
|---|---|---|
| `boxes` | `[batch, 25200, 4]` | `cx, cy, w, h` |
| `scores` | `[batch, 25200, 1]` | Confidence |
| `classes` | `[batch, 25200, 1]` | Class index as float |

`25200 = 3 × (80² + 40² + 20²)` — all raw anchors. nvinfer parses no such
layout natively, hence `nvdsinfer_custom_bbox/`. The parser only thresholds and
converts; clustering is left to nvinfer, so **`cluster-mode=2` is required**.

Box layout is overridable if you re-export the model differently:

```bash
TRUCK_BBOX_FORMAT=xyxy python3 ds_ts_ocr.py -c configs/ds_ts_ocr_config.txt
TRUCK_BBOX_DEBUG=1     python3 ds_ts_ocr.py -c configs/ds_ts_ocr_config.txt
```

`TRUCK_BBOX_DEBUG=1` prints the first few raw candidate rows once — the fastest
way to identify an unfamiliar export's layout.

---

## Configuration

`configs/ds_ts_ocr_config.txt`, deepstream-app style. Paths resolve relative to
the config file; `output-file` resolves relative to the working directory.

| Group | Notable keys |
|---|---|
| `[source0]` | `uri` — file or `rtsp://`; `file-loop` |
| `[streammux]` | `width`, `height`, `batch-size`, `live-source` (1 for RTSP) |
| `[primary-gie]` | `config-file` |
| `[tracker]` | `enable` — off by default; see below |
| `[secondary-gie0]` | `enable`, `ocr-dict-file`, `ocr-min-confidence` |
| `[sink0]` | `type` 1=fake 2=display 3=file; `enc-type` |

### Encoder selection

`enc-type=0` (default) auto-detects: hardware NVENC when `/dev/v4l2-nvenc`
exists, otherwise `x264enc`. Force with `1` (hardware) or `2` (software).

This matters — many GPU containers expose `/dev/nvidia*` but **not**
`/dev/v4l2-nvenc`, and `nvv4l2h264enc` then fails at S_FMT with an opaque
`Unknown error -1`. This container is one of them, so it encodes in software.

### Tracker

Off by default. Turn it on once OCR is live: it gives each plate a stable
`object_id`, so you can recognise a plate once rather than every frame.

---

## Class list

`model_wts/labels.txt`, 21 classes. **`License_Plate` is class id 5.**

```
0 Headlight    5 License_Plate    10 Top_Hatch_Open   15 Person
1 Bumper       6 Top_Hatch_Closed 11 Ladder           16 Spare_Tire
2 Truck        7 Tanker_Shell     12 Fuel_Tank        17 Brake_Lights
3 Reflector    8 MVPI             13 Fire_Extinguisher 18 Tanker_Id
4 Cabin_Hood   9 Helmet           14 (see file)       19 External_Battery_Secured
                                                      20 Cabin_Door
```

`Tanker_Id` (17) and `MVPI` (8) are also text-bearing — if you want those read
too, add their ids: `operate-on-class-ids=5;8;17`.

---

## Verification gaps

**License_Plate was never detected on the test clip.** The pipeline ran 1443
frames of DeepStream's `sample_1080p_h264.mp4` and produced 2,373 detections,
but zero plates. That footage is generic road traffic; this model is trained for
close-up tanker-truck inspection (`Tanker_Shell`, `Top_Hatch`, `MVPI`,
`Fire_Extinguisher`). Detections on it are largely spurious — `Truck` in 96% of
frames, `Fire_Extinguisher` 481 times.

**Test against your own inspection footage** before trusting any of the tuning
here. Specifically unverified:

- that class-5 detections fire at the configured `pre-cluster-threshold=0.35`
- the license-plate → OCR gating path, end to end
- whether `maintain-aspect-ratio=1` matches how the model was trained

The parser, pipeline, engine build, encoder fallback and file output **are**
verified.

---

## Layout

```
ds_ts_ocr/
├── ds_ts_ocr.py                       Application
├── configs/
│   ├── ds_ts_ocr_config.txt           Pipeline config
│   ├── config_infer_primary_truck.txt PGIE
│   └── config_infer_secondary_ocr.txt SGIE / OCR (inactive)
├── nvdsinfer_custom_bbox/
│   ├── nvdsparsebbox_truck.cpp        Custom parser
│   └── Makefile
└── model_wts/
    ├── truck_best.onnx                Detector
    ├── labels.txt                     21 classes
    ├── best.pt                        PyTorch source
    ├── ppocr_rec.onnx                 ← you supply
    └── ocr_dict.txt                   ← you supply
```
