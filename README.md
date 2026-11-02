# Tello Autonomous Navigation

This project contains a set of autonomous navigation experiments for the DJI Tello drone. The repository includes a real-flight control script and several navigation modes for indoor flight and person-following behavior.

## Included modes

1. MonoNav Metric Ultimate
   - Wall-corner navigation using recovered yaw kinematics and metric depth estimation.
2. HKUST EGO-Planner v2
   - Smooth corridor navigation using B-spline trajectory optimization.
3. MonoNav MediaPipe Person Follower
   - Autonomous tracking and following of a person in the camera frame.

## Project structure

```text
Tello_Autonomous_Navigation/
├── tello_live_autopilot.py              # Main real-flight entry point (--mode ultimate / ego)
├── mononav_metric_ultimate.py           # Module 1: MonoNav Metric Ultimate
├── ego_planner_v2.py                    # Module 2: HKUST EGO-Planner v2
├── mononav_person_follower.py           # Module 3: person tracking and following
│
├── weights/
│   ├── depth_anything_v2_metric_hypersim_vits.pth
│   ├── depth_anything_v2_vits.pth
│   ├── dronet_indoor_finetuned.pth
│   └── pulp-dronet-v3-resblock-1.0.pth
│
├── efficientdet_lite0.tflite
├── pose_landmarker_lite.task
│
├── real_flights_signed/
│   ├── Flight_1_MonoNav_Ultimate_Test_12-31-02.mp4
│   ├── Flight_2_MonoNav_Ultimate_Calibration_12-33-17.mp4
│   ├── Flight_3_MonoNav_Ultimate_Full_Flight_12-35-29.mp4
│   ├── Flight_4_HKUST_EGO_Planner_v2_Test_12-50-55.mp4
│   ├── Flight_5_HKUST_EGO_Planner_v2_Full_Flight_12-54-54.mp4
│   ├── Flight_11_MediaPipe_Follow_Success_15-16-39.mp4
│   └── Flight_Comparison_3_Autopilots_Split.mp4
│
├── docs/
│   ├── RESEARCH_AND_BENCHMARK_REPORT_UA.md
│   └── screenshots/
│
├── requirements.txt
├── start_real_drone.bat
├── fix_tello_firewall.bat
├── README.md
└── .gitignore
```

## Requirements

- Python 3.9+
- DJI Tello drone
- Tello Wi-Fi connection
- Dependencies from `requirements.txt`
- CUDA-capable GPU recommended for perception and depth models

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running on a real drone

1. Power on the DJI Tello and connect to its Wi-Fi network (`TELLO-XXXXXX`).
2. Start the required mode:

```bash
# Mode 1: MonoNav Metric Ultimate
python tello_live_autopilot.py --mode ultimate

# Mode 2: HKUST EGO-Planner v2
python tello_live_autopilot.py --mode ego

# Mode 3: Person follower
python mononav_person_follower.py
```

## Controls during flight

- `[t]` — Take off
- `[n]` — Enable / disable autonomous autopilot
- `[SPACE]` — Emergency hover (`v = 0`, `w = 0`)
- `[l]` / `[q]` — Safe landing

## Notes

- This project is intended for research and experimentation with autonomous drone navigation.
- Real-flight behavior depends on the environment, lighting, drone battery level, and Wi-Fi stability.
- Some model files are included in the repository, but additional setup may still be required depending on the environment.

## Documentation

Additional material and benchmark notes are available in the `docs/` folder.

## License

This repository does not currently declare a license. If you plan to reuse or redistribute the code, confirm the licensing status with the repository owner before publishing or distributing it.
