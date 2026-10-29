# PULP-DroNet v3 float inference

`pulp_dronet_v3.py` is derived from ETH Zurich / University of Bologna's
Apache-2.0 licensed source. The original copyright notice is retained; the
license is in `PULP_DRONET_LICENSE`.

- Upstream: https://github.com/pulp-platform/pulp-dronet
- Revision: `98f15cfe2e9c34c3a546358e22e45b3b09f11a79`
- Original path: `tiny-pulp-dronet-v3/model/dronet_v3.py`
- Original source SHA-256: `410c7a3ca9437e670b2533086847789e2e11f46081fb39a75a12ec386ea35854`

Local changes: removed the NeMO import, console output and relative `sys.path`
mutation; replaced its parameter-free, non-quantized `PACT_IntegerAdd` with
ordinary tensor addition. This preserves the float architecture and checkpoint
keys. Quantized NeMO models are not supported by this adapter.

The regular checkpoint uses `ResBlock`, depth multiplier 1.0, bypass enabled.
The Tiny checkpoint uses `Depthwise_Separable`, multiplier 0.125, bypass disabled.
The official testing transform is `CenterCrop(200), ToTensor()` on grayscale
images. The adapter applies that transform to raw BGR camera frames, without
ImageNet normalization. Positive upstream yaw is leftward, matching the robot
coordinates used here; the Tello RC mapper performs the clockwise sign change.
