# 🚁 Tello Autonomous Navigation (Final Clean Release)

Фінальний, очищений від зайвого коду репозиторій автономного польоту для квадрокоптера **DJI Tello**, заснований на **трьох реально підтверджених компонентах**:

1. 🏆 **MonoNav Metric Ultimate** (обхід кутів кімнат на базі відновленої кінематики $\Delta\text{yaw} \to (w_x, w_y)$ та метричного LiDAR);
2. 🏆 **HKUST EGO-Planner v2** (B-сплайн оптимізація траєкторій без зіткнень і без ривків у коридорах);
3. 👤 **MonoNav MediaPipe Person Follower** (автономний трекінг та супровід людини в кадрі на дистанції ~1.6м зі швидкістю 35+ FPS).

---

## 📁 Структура проекту

```text
Tello_Autonomous_Navigation/
├── tello_live_autopilot.py             # ГОЛОВНИЙ пульт реального польоту дрона (--mode ultimate / ego)
│
├── mononav_metric_ultimate.py          # Модуль 1: MonoNav Metric Ultimate (обхід кутів стін)
├── ego_planner_v2.py                   # Модуль 2: HKUST EGO-Planner v2 (B-сплайни в коридорах)
├── mononav_person_follower.py          # Модуль 3: Автономний трекінг та слідування за людиною
│
├── weights/                            # Навчені ваги моделей:
│   ├── depth_anything_v2_metric_hypersim_vits.pth  # Метричний сканер Hypersim (до 20м)
│   ├── depth_anything_v2_vits.pth                  # Базовий енкодер Depth Anything V2
│   ├── dronet_indoor_finetuned.pth                 # Дотренований рефлекс-щит екстреного гальмування
│   └── pulp-dronet-v3-resblock-1.0.pth             # Ваги DroNet v3
│
├── efficientdet_lite0.tflite           # Ваги детектора людини MediaPipe
├── pose_landmarker_lite.task           # Модель трекінгу пози MediaPipe
│
├── real_flights_signed/                # Всі підтверджені відео польотів 29 жовтня:
│   ├── Flight_1_MonoNav_Ultimate_Test_12-31-02.mp4
│   ├── Flight_2_MonoNav_Ultimate_Calibration_12-33-17.mp4
│   ├── Flight_3_MonoNav_Ultimate_Full_Flight_12-35-29.mp4  (12.4 МБ, успішний обхід кутів)
│   ├── Flight_4_HKUST_EGO_Planner_v2_Test_12-50-55.mp4
│   ├── Flight_5_HKUST_EGO_Planner_v2_Full_Flight_12-54-54.mp4 (11.7 МБ, плавний політ коридором)
│   ├── Flight_11_MediaPipe_Follow_Success_15-16-39.mp4    (6.1 МБ, успішне слідування за людиною)
│   └── Flight_Comparison_3_Autopilots_Split.mp4           (14.2 МБ, 3-віконний спліт)
│
├── docs/                               # Повна документація та скриншоти
│   ├── RESEARCH_AND_BENCHMARK_REPORT_UA.md # Головний науковий звіт з описом усіх етапів
│   └── screenshots/                    # 68 скриншотів усього шляху досліджень
│
├── requirements.txt                    # Залежності Python
├── start_real_drone.bat                # Скрипт швидкого запуску польоту
└── fix_tello_firewall.bat              # Налаштування мережі Windows Firewall для Tello
```

---

## 🚀 Як запускати на реальному дроні

1. Увімкніть квадрокоптер **DJI Tello** і підключіться до його Wi-Fi (`TELLO-XXXXXX`).
2. Запустіть потрібний режим:

```bash
# Режим 1: MonoNav Metric Ultimate (автономний обхід кутів за кінематикою Delta-yaw):
python tello_live_autopilot.py --mode ultimate

# Режим 2: HKUST EGO-Planner v2 (гладкий сплайновий політ коридорами):
python tello_live_autopilot.py --mode ego

# Режим 3: Автономний трекінг та слідування за людиною (Person Follower):
python mononav_person_follower.py
```

### Керування під час польоту:
* `[ t ]` — Зліт (Takeoff);
* `[ n ]` — Увімкнути / Вимкнути автономний автопілот;
* `[ SPACE ]` — Екстрене зависання (Hover, $v=0, w=0$);
* `[ l ] / [ q ]` — Безпечна посадка (Land).
