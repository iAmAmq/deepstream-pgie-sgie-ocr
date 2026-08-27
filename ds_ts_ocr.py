#!/usr/bin/env python3
"""ds_ts_ocr -- truck component detection with license-plate recognition.

A configuration-file driven DeepStream pipeline:

    source -> streammux -> pgie -> [tracker] -> [sgie/OCR] -> osd -> sink

The primary GIE is a YOLO detector over 21 truck component classes. The
secondary GIE is a PaddleOCR text-recognition model that runs *only* on
License_Plate detections -- that gating is done by nvinfer itself through
``process-mode=2`` / ``operate-on-gie-id`` / ``operate-on-class-ids`` in
config_infer_secondary_ocr.txt, not by application code.

The recognition head is CTC based, so its output cannot be turned into a label
by nvinfer. The SGIE therefore runs with ``output-tensor-meta=1`` and the raw
tensor is decoded here, in :func:`sgie_probe`.

Usage:
    $ python3 ds_ts_ocr.py -c configs/ds_ts_ocr_config.txt
"""

import argparse
import configparser
import ctypes
import os
import sys

import numpy as np

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst          # noqa: E402

import pyds                                   # noqa: E402


# Class id of License_Plate in model_wts/labels.txt. Kept here only for
# logging and the OCR-text association; the actual gating lives in the SGIE
# config so that nvinfer never even crops the other classes.
LICENSE_PLATE_CLASS_ID = 5

PGIE_UNIQUE_ID = 1
SGIE_UNIQUE_ID = 2


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

class DsConfig:
    """deepstream-app style .txt config, with paths relative to the file."""

    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.dirname = os.path.dirname(self.path)
        self._cp = configparser.ConfigParser(strict=False)
        # Keys are case sensitive in deepstream-app configs.
        self._cp.optionxform = str
        if not self._cp.read(self.path):
            raise IOError("could not read config %s" % self.path)

    def group(self, name):
        return self._cp[name] if self._cp.has_section(name) else {}

    def has(self, name):
        return self._cp.has_section(name)

    def resolve(self, value):
        """Turn a config-relative path into an absolute one."""
        if not value:
            return value
        if os.path.isabs(value):
            return value
        return os.path.normpath(os.path.join(self.dirname, value))


def get_str(group, key, default=None):
    value = group.get(key, default)
    return value.strip() if isinstance(value, str) else value


def get_int(group, key, default=0):
    try:
        return int(str(group.get(key, default)).strip())
    except (TypeError, ValueError):
        return default


def get_float(group, key, default=0.0):
    try:
        return float(str(group.get(key, default)).strip())
    except (TypeError, ValueError):
        return default


def get_bool(group, key, default=False):
    raw = group.get(key)
    if raw is None:
        return default
    return str(raw).strip() not in ("0", "false", "False", "")


def make_element(factory, name):
    elem = Gst.ElementFactory.make(factory, name)
    if not elem:
        sys.stderr.write("Unable to create element %s (%s)\n" % (name, factory))
        sys.exit(1)
    return elem


def set_prop(elem, name, value):
    if value is not None:
        elem.set_property(name, value)


# ---------------------------------------------------------------------------
# CTC decoding for the PaddleOCR recognition head
# ---------------------------------------------------------------------------

def load_char_dict(path):
    """Load a PaddleOCR character dictionary.

    The file holds one character per line. PaddleOCR reserves index 0 for the
    CTC blank and appends a space at the end, so the decode table is
    ``["blank"] + lines + [" "]``.
    """
    if not path or not os.path.isfile(path):
        return None
    with open(path, 'r', encoding='utf-8') as handle:
        chars = [line.rstrip('\n\r') for line in handle]
    return ['<blank>'] + chars + [' ']


def ctc_greedy_decode(probs, charset):
    """Best-path CTC decode of one recognition output.

    ``probs`` is a [num_steps, num_classes] array of softmax scores. Returns
    ``(text, mean_confidence)``; repeated symbols are collapsed and blanks
    dropped, which is the standard PaddleOCR post-processing.

    The per-step argmax runs in numpy rather than Python: PP-OCRv6 has 18710
    classes, so a 40-step output is ~750k comparisons per crop, which is far
    too slow to do element-wise inside a streaming pad probe.
    """
    indices = probs.argmax(axis=1)
    scores = probs[np.arange(probs.shape[0]), indices]

    pieces = []
    confidences = []
    previous = -1

    for step in range(indices.shape[0]):
        best_idx = int(indices[step])
        # 0 is the CTC blank; a repeat of the previous symbol is a duplicate.
        if best_idx != 0 and best_idx != previous:
            pieces.append(charset[best_idx]
                          if best_idx < len(charset) else '?')
            confidences.append(float(scores[step]))
        previous = best_idx

    if not pieces:
        return "", 0.0
    return "".join(pieces), sum(confidences) / len(confidences)


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def plate_row_probe(pad, info, app):
    """Shrink License_Plate boxes to the text row the recogniser should read.

    Gulf plates are two-row: Arabic script above, Latin below. nvinfer crops
    the whole object box and squeezes it into the recogniser's 48px input
    height, so each row is left with roughly 24px -- and the single-line CTC
    head then has to read two rows at once. Trimming the box to the Latin row
    before the SGIE gives that row the full input height.

    Runs on the primary detector's src pad, so the change is in place before
    the secondary sees the batch.
    """
    buf = info.get_buffer()
    if not buf:
        return Gst.PadProbeReturn.OK
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    keep = app.plate_row_fraction
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            if obj_meta.class_id == LICENSE_PLATE_CLASS_ID:
                r = obj_meta.rect_params
                trimmed = r.height * keep
                r.top = r.top + (r.height - trimmed)
                r.height = trimmed
            try:
                l_obj = l_obj.next
            except StopIteration:
                break
        try:
            l_frame = l_frame.next
        except StopIteration:
            break
    return Gst.PadProbeReturn.OK


def sgie_probe(pad, info, app):
    """Decode the OCR tensor attached to each License_Plate object.

    Runs on the SGIE src pad. Only plate objects carry tensor metadata, since
    the SGIE is configured to operate on class 5 alone, so no class filtering
    is needed here -- but it is asserted anyway to keep the association honest
    if the config is ever loosened.
    """
    buf = info.get_buffer()
    if not buf:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            if obj_meta.class_id == LICENSE_PLATE_CLASS_ID:
                text, confidence = _decode_object(obj_meta, app)
                if app.verbose:
                    # Report the raw result before thresholding, so a silent
                    # zero can be told apart from a rejected low-confidence read.
                    if text:
                        print("  ocr raw: %r conf=%.3f%s"
                              % (text, confidence,
                                 "" if confidence >= app.ocr_min_confidence
                                 else "  [below ocr-min-confidence]"))
                    else:
                        print("  ocr raw: <empty>  (no tensor meta, or decode "
                              "produced only blanks)")
                if text and confidence >= app.ocr_min_confidence:
                    _label_object(obj_meta, text, confidence)
                    app.plate_reads += 1
                    if app.verbose:
                        print("frame %d: plate '%s' (%.2f)"
                              % (frame_meta.frame_num, text, confidence))

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def _decode_object(obj_meta, app):
    """Find this object's OCR tensor meta and CTC-decode it."""
    l_user = obj_meta.obj_user_meta_list
    while l_user is not None:
        try:
            user_meta = pyds.NvDsUserMeta.cast(l_user.data)
        except StopIteration:
            break

        if (user_meta.base_meta.meta_type
                == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META):
            tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
            result = _decode_tensor(tensor_meta, app)
            if result is not None:
                return result

        try:
            l_user = l_user.next
        except StopIteration:
            break

    return "", 0.0


def _decode_tensor(tensor_meta, app):
    """Read layer 0 of a tensor meta as [num_steps, num_classes] and decode."""
    if tensor_meta.num_output_layers < 1:
        return None

    layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
    dims = layer.inferDims

    # Expect [seq_len, num_classes]; a leading 1 is tolerated.
    shape = [dims.d[i] for i in range(dims.numDims)]
    shape = [d for d in shape if d != 1] or shape
    if len(shape) < 2:
        if app.verbose:
            sys.stderr.write("OCR tensor has unexpected shape %s\n" % shape)
        return None
    num_steps, num_classes = shape[-2], shape[-1]

    if app.charset is None:
        # Without a dictionary the indices cannot be turned into characters.
        if not app.warned_no_dict:
            app.warned_no_dict = True
            sys.stderr.write(
                "OCR: no character dictionary loaded (ocr-dict-file); "
                "recognition output cannot be decoded\n")
        return None

    ptr = ctypes.cast(pyds.get_ptr(layer.buffer),
                      ctypes.POINTER(ctypes.c_float))
    # Wrap the inference buffer without copying; it stays valid for the life
    # of this probe call, which is all the decode needs.
    probs = np.ctypeslib.as_array(ptr, shape=(num_steps, num_classes))

    return ctc_greedy_decode(probs, app.charset)


def _label_object(obj_meta, text, confidence):
    """Replace the object's OSD label with the recognised plate text."""
    obj_meta.text_params.display_text = "%s (%.2f)" % (text, confidence)
    obj_meta.text_params.set_bg_clr = 1
    obj_meta.text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.8)
    obj_meta.text_params.font_params.font_color.set(1.0, 1.0, 0.0, 1.0)


def write_ppm(path, rgb):
    """Write an HxWx3 uint8 array as a binary PPM.

    Deliberately dependency-free: neither OpenCV nor Pillow is present in the
    DeepStream container, and PPM is enough to eyeball a crop.
    """
    height, width = rgb.shape[0], rgb.shape[1]
    with open(path, 'wb') as handle:
        handle.write(b"P6\n%d %d\n255\n" % (width, height))
        handle.write(rgb.tobytes())


def dump_plate_crops(gst_buffer, frame_meta, app):
    """Save each License_Plate crop of this frame for inspection."""
    try:
        surface = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
    except Exception as err:                                  # noqa: BLE001
        if not app.warned_no_surface:
            app.warned_no_surface = True
            sys.stderr.write("Cannot map frame surface for crop dump (%s); "
                             "is nvbuf-memory-type unified?\n" % err)
        return

    frame = np.array(surface, copy=True, order='C')
    pyds.unmap_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)

    l_obj = frame_meta.obj_meta_list
    while l_obj is not None:
        try:
            obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
        except StopIteration:
            break
        if obj_meta.class_id == LICENSE_PLATE_CLASS_ID:
            r = obj_meta.rect_params
            x0, y0 = max(int(r.left), 0), max(int(r.top), 0)
            x1 = min(int(r.left + r.width), frame.shape[1])
            y1 = min(int(r.top + r.height), frame.shape[0])
            if x1 > x0 and y1 > y0:
                crop = frame[y0:y1, x0:x1, :3]
                name = os.path.join(app.dump_dir, "plate_f%05d_%02d.ppm"
                                    % (frame_meta.frame_num, app.dumped))
                write_ppm(name, np.ascontiguousarray(crop))
                app.dumped += 1
        try:
            l_obj = l_obj.next
        except StopIteration:
            break


def osd_probe(pad, info, app):
    """Count detections per class and drive the periodic report."""
    buf = info.get_buffer()
    if not buf:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        app.frame_count += 1
        if app.dump_dir and app.dumped < app.dump_limit:
            dump_plate_crops(buf, frame_meta, app)
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            label = obj_meta.obj_label or str(obj_meta.class_id)
            app.class_counts[label] = app.class_counts.get(label, 0) + 1
            # Confidence range per class, so thresholds can be tuned against
            # what the detector actually produces on real footage.
            conf = obj_meta.confidence
            lo, hi, total = app.class_conf.get(label, (1e9, -1e9, 0.0))
            app.class_conf[label] = (min(lo, conf), max(hi, conf), total + conf)
            if obj_meta.class_id == LICENSE_PLATE_CLASS_ID:
                app.plate_detections += 1
                r = obj_meta.rect_params
                app.plate_sizes.append((r.width, r.height))
                if app.verbose:
                    print("  plate box: %.0fx%.0f conf=%.3f"
                          % (r.width, r.height, obj_meta.confidence))
            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class App:

    def __init__(self, cfg, verbose=False):
        self.cfg = cfg
        self.verbose = verbose
        self.pipeline = None
        self.loop = None

        self.frame_count = 0
        self.plate_detections = 0
        self.plate_reads = 0
        self.class_counts = {}
        self.class_conf = {}
        self.plate_sizes = []
        self.dump_dir = None
        self.dumped = 0
        self.dump_limit = 20
        self.warned_no_surface = False

        self.charset = None
        self.ocr_min_confidence = 0.0
        self.plate_row_fraction = 0.0
        self.warned_no_dict = False

    # -- build ---------------------------------------------------------------

    def build(self):
        self.pipeline = Gst.Pipeline()

        source_cfg = self.cfg.group('source0')
        mux_cfg = self.cfg.group('streammux')
        pgie_cfg = self.cfg.group('primary-gie')
        tracker_cfg = self.cfg.group('tracker')
        sgie_cfg = self.cfg.group('secondary-gie0')
        osd_cfg = self.cfg.group('osd')
        sink_cfg = self.cfg.group('sink0')

        # -- source + streammux ---------------------------------------------
        streammux = make_element("nvstreammux", "streammux")
        set_prop(streammux, 'batch-size', get_int(mux_cfg, 'batch-size', 1))
        set_prop(streammux, 'width', get_int(mux_cfg, 'width', 1920))
        set_prop(streammux, 'height', get_int(mux_cfg, 'height', 1080))
        set_prop(streammux, 'batched-push-timeout',
                 get_int(mux_cfg, 'batched-push-timeout', 40000))
        set_prop(streammux, 'live-source', get_bool(mux_cfg, 'live-source'))
        set_prop(streammux, 'gpu-id', get_int(mux_cfg, 'gpu-id', 0))
        self.pipeline.add(streammux)

        uri = get_str(source_cfg, 'uri')
        if not uri:
            sys.stderr.write("[source0] uri is required\n")
            sys.exit(1)
        source_bin = make_element("nvurisrcbin", "source-bin-0")
        set_prop(source_bin, 'uri', uri)
        set_prop(source_bin, 'gpu-id', get_int(source_cfg, 'gpu-id', 0))
        set_prop(source_bin, 'file-loop', get_bool(source_cfg, 'file-loop'))
        set_prop(source_bin, 'cudadec-memtype',
                 get_int(source_cfg, 'cudadec-memtype', 0))
        self.pipeline.add(source_bin)
        source_bin.connect("pad-added", self._on_source_pad, streammux)

        chain = [streammux]

        # -- primary detector ------------------------------------------------
        if get_bool(pgie_cfg, 'enable', True):
            pgie = make_element("nvinfer", "primary-gie")
            set_prop(pgie, 'config-file-path',
                     self.cfg.resolve(get_str(pgie_cfg, 'config-file')))
            set_prop(pgie, 'unique-id', PGIE_UNIQUE_ID)
            set_prop(pgie, 'gpu-id', get_int(pgie_cfg, 'gpu-id', 0))
            # The muxer batch governs how many frames reach the detector.
            set_prop(pgie, 'batch-size', get_int(mux_cfg, 'batch-size', 1))
            self.pipeline.add(pgie)
            chain += [make_element("queue", "queue-pgie"), pgie]

        # -- tracker ----------------------------------------------------------
        if get_bool(tracker_cfg, 'enable', False):
            tracker = make_element("nvtracker", "tracker")
            set_prop(tracker, 'tracker-width',
                     get_int(tracker_cfg, 'tracker-width', 640))
            set_prop(tracker, 'tracker-height',
                     get_int(tracker_cfg, 'tracker-height', 384))
            set_prop(tracker, 'gpu_id', get_int(tracker_cfg, 'gpu-id', 0))
            set_prop(tracker, 'll-lib-file',
                     self.cfg.resolve(get_str(tracker_cfg, 'll-lib-file')))
            ll_config = get_str(tracker_cfg, 'll-config-file')
            if ll_config:
                set_prop(tracker, 'll-config-file', self.cfg.resolve(ll_config))
            self.pipeline.add(tracker)
            chain += [make_element("queue", "queue-tracker"), tracker]

        # -- secondary OCR ----------------------------------------------------
        self.sgie = None
        if get_bool(sgie_cfg, 'enable', False):
            sgie_config = self.cfg.resolve(get_str(sgie_cfg, 'config-file'))
            if not os.path.isfile(sgie_config):
                sys.stderr.write(
                    "[secondary-gie0] enabled but config-file %s is missing\n"
                    % sgie_config)
                sys.exit(1)
            sgie = make_element("nvinfer", "secondary-gie-ocr")
            set_prop(sgie, 'config-file-path', sgie_config)
            set_prop(sgie, 'unique-id', SGIE_UNIQUE_ID)
            set_prop(sgie, 'gpu-id', get_int(sgie_cfg, 'gpu-id', 0))
            # process-mode / operate-on-* come from the config file: those keys
            # are what restrict recognition to License_Plate.
            self.pipeline.add(sgie)
            chain += [make_element("queue", "queue-sgie"), sgie]
            self.sgie = sgie

            self.ocr_min_confidence = get_float(sgie_cfg, 'ocr-min-confidence', 0.5)
            # Fraction of the plate box height to keep, measured from the
            # bottom. 0 disables the trim and the whole box is recognised.
            self.plate_row_fraction = get_float(sgie_cfg, 'ocr-plate-row-fraction', 0.0)
            if self.plate_row_fraction:
                print("Plate crop trimmed to the bottom %.0f%% (Latin row)"
                      % (100 * self.plate_row_fraction))
            dict_path = self.cfg.resolve(get_str(sgie_cfg, 'ocr-dict-file'))
            self.charset = load_char_dict(dict_path)
            if self.charset is None:
                sys.stderr.write(
                    "Warning: ocr-dict-file %s not found; recognised text "
                    "cannot be decoded\n" % dict_path)
            else:
                print("OCR enabled: %d symbols, min confidence %.2f"
                      % (len(self.charset), self.ocr_min_confidence))
        else:
            print("OCR stage disabled ([secondary-gie0] enable=0) -- "
                  "running detection only")

        # -- converter / osd / sink -------------------------------------------
        converter = make_element("nvvideoconvert", "converter")
        if self.dump_dir:
            # CUDA unified memory is required for get_nvds_buf_surface().
            set_prop(converter, 'nvbuf-memory-type', 3)
        chain += [make_element("queue", "queue-convert"), converter]

        if get_bool(osd_cfg, 'enable', True):
            osd = make_element("nvdsosd", "osd")
            set_prop(osd, 'display-bbox', get_bool(osd_cfg, 'display-bbox', True))
            set_prop(osd, 'display-text', get_bool(osd_cfg, 'display-text', True))
            set_prop(osd, 'process-mode', get_int(osd_cfg, 'process-mode', 0))
            chain += [make_element("queue", "queue-osd"), osd]

        chain += self._build_sink(sink_cfg)

        for elem in chain[1:]:
            if elem.get_parent() is None:
                self.pipeline.add(elem)
        for a, b in zip(chain, chain[1:]):
            if not a.link(b):
                sys.stderr.write("Failed to link %s -> %s\n"
                                 % (a.get_name(), b.get_name()))
                sys.exit(1)

        self._attach_probes()

    def _build_sink(self, sink_cfg):
        sink_type = get_int(sink_cfg, 'type', 3)
        sync = get_bool(sink_cfg, 'sync', False)

        if sink_type == 1:
            sink = make_element("fakesink", "sink")
            set_prop(sink, 'sync', sync)
            return [make_element("queue", "queue-sink"), sink]

        if sink_type == 2:
            sink = make_element("nveglglessink", "sink")
            set_prop(sink, 'sync', sync)
            return [make_element("queue", "queue-sink"),
                    make_element("nvvideoconvert", "sink-convert"), sink]

        # type 3: encode to a file.
        codec = get_int(sink_cfg, 'codec', 1)
        bitrate = get_int(sink_cfg, 'bitrate', 4000000)
        elements = [make_element("queue", "queue-sink"),
                    make_element("nvvideoconvert", "sink-convert")]

        if self._use_hw_encoder(sink_cfg):
            encoder = make_element(
                "nvv4l2h265enc" if codec == 2 else "nvv4l2h264enc", "encoder")
            set_prop(encoder, 'bitrate', bitrate)
            elements.append(encoder)
        else:
            # x264enc needs system memory in a plain raw format, which the
            # NVMM output of nvvideoconvert is not, so cap it explicitly.
            caps = make_element("capsfilter", "sink-caps")
            caps.set_property("caps",
                              Gst.Caps.from_string("video/x-raw, format=I420"))
            encoder = make_element("x264enc", "encoder")
            # x264enc takes kbit/s, and zerolatency keeps it from buffering
            # frames the pipeline will never flush on EOS.
            set_prop(encoder, 'bitrate', max(1, bitrate // 1000))
            set_prop(encoder, 'speed-preset', 1)
            set_prop(encoder, 'tune', 0x00000004)
            elements += [caps, encoder]
            if codec == 2:
                print("Software encoding falls back to H264; codec=2 ignored")
                codec = 1

        elements.append(make_element(
            "h265parse" if codec == 2 else "h264parse", "parser"))
        container = get_int(sink_cfg, 'container', 1)
        elements.append(make_element(
            "matroskamux" if container == 2 else "qtmux", "muxer"))

        sink = make_element("filesink", "sink")
        # Output paths are relative to the working directory, not the config.
        output = os.path.abspath(get_str(sink_cfg, 'output-file', 'out.mp4'))
        set_prop(sink, 'location', output)
        set_prop(sink, 'sync', sync)
        set_prop(sink, 'async', False)
        elements.append(sink)
        print("Writing output to %s" % output)
        return elements

    @staticmethod
    def _use_hw_encoder(sink_cfg):
        """Pick NVENC or x264enc.

        The nvv4l2 encoders need /dev/v4l2-nvenc, which many containers do not
        expose even when a GPU is visible; without it the element fails at
        S_FMT time with an unhelpful error, so probe for the node instead.
        """
        mode = get_int(sink_cfg, 'enc-type', 0)
        if mode == 1:
            return True
        if mode == 2:
            return False
        available = os.path.exists("/dev/v4l2-nvenc")
        if not available:
            print("No /dev/v4l2-nvenc; using the software encoder (x264enc)")
        return available

    def _on_source_pad(self, bin_, pad, streammux):
        """Link a source bin's dynamically added pad to the muxer."""
        caps = pad.get_current_caps() or pad.query_caps()
        name = caps.to_string()
        if not name.startswith("video/"):
            return
        sinkpad = streammux.request_pad_simple("sink_0") \
            if hasattr(streammux, "request_pad_simple") \
            else streammux.get_request_pad("sink_0")
        if not sinkpad:
            sys.stderr.write("Could not obtain a streammux sink pad\n")
            return
        if pad.link(sinkpad) != Gst.PadLinkReturn.OK:
            sys.stderr.write("Failed to link source to streammux\n")

    def _attach_probes(self):
        if self.sgie is not None and self.plate_row_fraction:
            pgie = self.pipeline.get_by_name("primary-gie")
            if pgie:
                pad = pgie.get_static_pad("src")
                if pad:
                    pad.add_probe(Gst.PadProbeType.BUFFER, plate_row_probe, self)

        if self.sgie is not None:
            pad = self.sgie.get_static_pad("src")
            if pad:
                pad.add_probe(Gst.PadProbeType.BUFFER, sgie_probe, self)

        osd = self.pipeline.get_by_name("osd")
        target = osd or self.pipeline.get_by_name("converter")
        if target:
            pad = target.get_static_pad("sink")
            if pad:
                pad.add_probe(Gst.PadProbeType.BUFFER, osd_probe, self)

    # -- run -----------------------------------------------------------------

    def run(self):
        self.loop = GLib.MainLoop()
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        interval = get_int(self.cfg.group('application'),
                           'perf-measurement-interval-sec', 5)
        if interval > 0:
            GLib.timeout_add_seconds(interval, self._report)

        print("Starting pipeline")
        self.pipeline.set_state(Gst.State.PLAYING)
        try:
            self.loop.run()
        except KeyboardInterrupt:
            print("\nInterrupted")
        finally:
            self.pipeline.set_state(Gst.State.NULL)

        self._summary()
        return 0

    def _on_bus_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.EOS:
            print("End of stream")
            self.loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            sys.stderr.write("Error: %s: %s\n" % (err, debug or ""))
            self.loop.quit()
        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            sys.stderr.write("Warning: %s\n" % err)
        return True

    def _report(self):
        top = sorted(self.class_counts.items(), key=lambda kv: -kv[1])[:6]
        summary = ", ".join("%s=%d" % (k, v) for k, v in top) or "none"
        print("frames=%d  detections: %s" % (self.frame_count, summary))
        return True

    def _summary(self):
        print("\n--- summary ---")
        print("frames processed      : %d" % self.frame_count)
        print("License_Plate detected: %d" % self.plate_detections)
        if self.plate_sizes:
            ws = [w for w, _ in self.plate_sizes]
            hs = [h for _, h in self.plate_sizes]
            print("  plate box w: %.0f-%.0f (mean %.0f)  h: %.0f-%.0f (mean %.0f)"
                  % (min(ws), max(ws), sum(ws) / len(ws),
                     min(hs), max(hs), sum(hs) / len(hs)))
        if self.sgie is not None:
            print("plates recognised     : %d" % self.plate_reads)
        print("  %-24s %6s  %s" % ("class", "count", "confidence min/mean/max"))
        for label, count in sorted(self.class_counts.items(),
                                   key=lambda kv: -kv[1]):
            lo, hi, total = self.class_conf.get(label, (0.0, 0.0, 0.0))
            print("  %-24s %6d  %.3f / %.3f / %.3f"
                  % (label, count, lo, total / max(count, 1), hi))


def parse_args():
    parser = argparse.ArgumentParser(
        prog="ds_ts_ocr",
        description="Truck component detection with license-plate OCR")
    parser.add_argument("-c", "--cfg-file", required=True, metavar="FILE",
                        help="Application config file")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print each recognised plate")
    parser.add_argument("--dump-plates", metavar="DIR",
                        help="Save License_Plate crops as PPM into DIR "
                             "(first 20), to inspect what the OCR is fed")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.cfg_file):
        sys.stderr.write("Config file %s not found\n" % args.cfg_file)
        return 1

    Gst.init(None)
    cfg = DsConfig(args.cfg_file)
    app = App(cfg, verbose=args.verbose)
    if args.dump_plates:
        os.makedirs(args.dump_plates, exist_ok=True)
        app.dump_dir = args.dump_plates
    app.build()
    return app.run()


if __name__ == '__main__':
    sys.exit(main())
