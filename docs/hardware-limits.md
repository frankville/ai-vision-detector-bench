# GPU video-session limits

GPU inference, video decoding, and video encoding use different resources. An encoding-session limit is not a camera-count limit.

| Stage | Relevant constraints |
| --- | --- |
| CUDA inference | Model, input size, precision, VRAM, compute, transfers, and scheduling |
| NVDEC decoding | GPU/codec support, resolution, bit depth, decode throughput, and memory |
| NVENC encoding | GPU/codec support, encode throughput, memory, and driver-enforced session policy |

## NVIDIA encoding policy

Checked 2026-11-01: NVIDIA Video Codec SDK 13.1 documents a limit of **12 concurrent NVENC encode sessions per system** across all non-qualified GPUs combined. Qualified GPUs are limited by available resources. Qualification is model-specific; use NVIDIA's support matrix rather than inferring it from a consumer or enterprise label.

This is an encoding policy, not a CUDA inference or NVDEC decoding limit. Older SDK/driver combinations can have lower encoding limits; verify the GPU, OS, installed driver, and current policy before planning deployment. An unrestricted session policy does not imply unlimited throughput.

## VisionBench scope

VisionBench opens inputs through OpenCV `VideoCapture`; it does not explicitly select or verify NVDEC decoding and has no NVENC output path. Selecting `--device cuda` selects inference placement, not hardware video decoding. The current `capacity_streams` value is a throughput estimate, not a validated multicamera result or encoder-session allowance.

Capacity work must record the effective decode/encode backends and distinguish unsupported hardware, documented session policies, and measured resource saturation. Apply an encoding policy only when NVENC is actually used. Unknown capabilities remain unknown; do not silently clamp inference capacity to an encoder-session limit.

## Sources

- [NVIDIA NVENC application note, SDK 13.1](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.1/nvenc-application-note/index.html#nvenc-licensing-policy)
- [NVIDIA video encode/decode support matrix](https://developer.nvidia.com/video-encode-decode-support-matrix)
- [NVIDIA NVDEC application note](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvdec-application-note/index.html#nvdec-performance)
