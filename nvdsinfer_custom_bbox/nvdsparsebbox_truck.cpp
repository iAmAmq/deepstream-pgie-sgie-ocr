/*
 * Custom bounding-box parser for truck_best.onnx.
 *
 * The model is a YOLOv5/v7-style export whose decode head has been split into
 * three output tensors, with NMS left to the caller:
 *
 *   boxes    [batch, 25200, 4]   candidate boxes
 *   scores   [batch, 25200, 1]   objectness * class confidence
 *   classes  [batch, 25200, 1]   class index as a float
 *
 * 25200 = 3 * (80^2 + 40^2 + 20^2), the usual anchor count for a 640x640
 * input, so every candidate is emitted raw. This parser only thresholds and
 * converts; clustering is left to nvinfer, which must therefore run with
 * cluster-mode=2 (NMS) and a sensible nms-iou-threshold.
 *
 * Box layout is selected with the TRUCK_BBOX_FORMAT environment variable:
 *   cxcywh (default)  boxes are center-x, center-y, width, height
 *   xyxy              boxes are x1,y1,x2,y2
 * cxcywh is the default because that is what this export emits, confirmed by
 * dumping raw rows: e.g. [275.25 280.75 19.53 32.50], where fields 3-4 are far
 * smaller than fields 1-2 and so cannot be a bottom-right corner.
 * Set TRUCK_BBOX_DEBUG=1 to print the first few raw rows once, which is the
 * quickest way to confirm which layout an export actually uses.
 *
 * Build:  make
 * Use:    parse-bbox-func-name=NvDsInferParseCustomTruck
 *         custom-lib-path=<path>/libnvdsinfer_custom_bbox_truck.so
 *         cluster-mode=2
 */

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>

#include "nvdsinfer_custom_impl.h"

#define TRUCK_NUM_CANDIDATES_MAX 1000000

extern "C" bool NvDsInferParseCustomTruck (
    std::vector < NvDsInferLayerInfo > const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector < NvDsInferObjectDetectionInfo > &objectList);

namespace {

enum BoxFormat { FMT_XYXY, FMT_CXCYWH };

/* Read once: the layout, and whether to dump a sample of raw rows. */
BoxFormat
box_format ()
{
  static BoxFormat fmt = [] () {
    const char *e = std::getenv ("TRUCK_BBOX_FORMAT");
    if (e && std::strcmp (e, "xyxy") == 0)
      return FMT_XYXY;
    return FMT_CXCYWH;
  }();
  return fmt;
}

bool
debug_enabled ()
{
  static bool on = [] () {
    const char *e = std::getenv ("TRUCK_BBOX_DEBUG");
    return e && e[0] == '1';
  }();
  return on;
}

/* Locate an output layer by name. Returns nullptr when absent. */
const NvDsInferLayerInfo *
find_layer (std::vector < NvDsInferLayerInfo > const &layers, const char *name)
{
  for (auto const &l : layers) {
    if (l.layerName && std::strcmp (l.layerName, name) == 0)
      return &l;
  }
  return nullptr;
}

}  /* namespace */

extern "C" bool
NvDsInferParseCustomTruck (
    std::vector < NvDsInferLayerInfo > const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector < NvDsInferObjectDetectionInfo > &objectList)
{
  const NvDsInferLayerInfo *boxes = find_layer (outputLayersInfo, "boxes");
  const NvDsInferLayerInfo *scores = find_layer (outputLayersInfo, "scores");
  const NvDsInferLayerInfo *classes = find_layer (outputLayersInfo, "classes");

  if (!boxes || !scores || !classes) {
    static bool warned = false;
    if (!warned) {
      warned = true;
      fprintf (stderr, "NvDsInferParseCustomTruck: expected output layers "
          "'boxes', 'scores' and 'classes'; got %zu layers:",
          outputLayersInfo.size ());
      for (auto const &l : outputLayersInfo)
        fprintf (stderr, " %s", l.layerName ? l.layerName : "(unnamed)");
      fprintf (stderr, "\n");
    }
    return false;
  }

  /* Candidate count is dimension 0 of the per-image slice (the batch
   * dimension is already stripped by nvinfer). */
  const NvDsInferDims &dims = boxes->inferDims;
  unsigned int num = dims.numDims > 0 ? dims.d[0] : 0;

  if (num == 0 || num > TRUCK_NUM_CANDIDATES_MAX) {
    static bool warned = false;
    if (!warned) {
      warned = true;
      fprintf (stderr, "NvDsInferParseCustomTruck: implausible candidate "
          "count %u from 'boxes' dims (numDims=%u)\n", num, dims.numDims);
    }
    return false;
  }

  const float *pb = (const float *) boxes->buffer;
  const float *ps = (const float *) scores->buffer;
  const float *pc = (const float *) classes->buffer;

  if (!pb || !ps || !pc)
    return false;

  /* One-shot dump to identify the box layout of an unfamiliar export. */
  if (debug_enabled ()) {
    static bool dumped = false;
    if (!dumped) {
      dumped = true;
      fprintf (stderr, "NvDsInferParseCustomTruck: network %ux%u, %u candidates\n",
          networkInfo.width, networkInfo.height, num);
      unsigned int shown = 0;
      for (unsigned int i = 0; i < num && shown < 5; i++) {
        if (ps[i] < 0.10f)
          continue;
        fprintf (stderr, "  cand[%u] score=%.3f class=%.0f box=[%.2f %.2f %.2f %.2f]\n",
            i, ps[i], pc[i], pb[i * 4 + 0], pb[i * 4 + 1],
            pb[i * 4 + 2], pb[i * 4 + 3]);
        shown++;
      }
      if (shown == 0)
        fprintf (stderr, "  (no candidate scored above 0.10)\n");
    }
  }

  const BoxFormat fmt = box_format ();
  const unsigned int numClasses = detectionParams.numClassesConfigured;

  objectList.clear ();
  objectList.reserve (256);

  for (unsigned int i = 0; i < num; i++) {
    const float score = ps[i];
    const int cls = (int) (pc[i] + 0.5f);

    if (cls < 0 || (unsigned int) cls >= numClasses)
      continue;

    /* Per-class threshold when configured, else the global pre-cluster one. */
    float thresh = detectionParams.perClassPreclusterThreshold.size () > (size_t) cls
        ? detectionParams.perClassPreclusterThreshold[cls]
        : 0.0f;
    if (score < thresh)
      continue;

    const float *b = &pb[i * 4];
    float left, top, width, height;

    if (fmt == FMT_CXCYWH) {
      width = b[2];
      height = b[3];
      left = b[0] - width * 0.5f;
      top = b[1] - height * 0.5f;
    } else {
      left = b[0];
      top = b[1];
      width = b[2] - b[0];
      height = b[3] - b[1];
    }

    if (width <= 0.0f || height <= 0.0f)
      continue;

    /* Clamp to the network input; nvinfer rescales to source resolution. */
    const float maxW = (float) networkInfo.width;
    const float maxH = (float) networkInfo.height;

    if (left < 0.0f) { width += left; left = 0.0f; }
    if (top < 0.0f) { height += top; top = 0.0f; }
    if (left + width > maxW) width = maxW - left;
    if (top + height > maxH) height = maxH - top;
    if (width <= 0.0f || height <= 0.0f)
      continue;

    NvDsInferObjectDetectionInfo obj;
    obj.classId = (unsigned int) cls;
    obj.left = left;
    obj.top = top;
    obj.width = width;
    obj.height = height;
    obj.detectionConfidence = score;
    objectList.push_back (obj);
  }

  return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE (NvDsInferParseCustomTruck);
