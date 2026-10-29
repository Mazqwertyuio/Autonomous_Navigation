# PULP-DroNet v3 для DJI Tello

Інтегровано офіційні float-моделі PULP-DroNet v3 та Tiny v3, доступні через
`--model dronet-v3` та `--model dronet-v3-tiny`. Поточний типовий CLI-режим —
[чистий NoMaD](PURE_NOMAD_UA.md). DroNet не є самостійною системою дослідження.

## Підготовка

```powershell
.\venv\Scripts\python.exe download_dronet_v3.py --variant all
```

Завантажуються лише два файли ваг з офіційного репозиторію, закріплені за
commit `98f15cfe2e9c34c3a546358e22e45b3b09f11a79`. SHA-256 перевіряється під час
завантаження і кожного створення моделі. Відсутні, пошкоджені або несумісні ваги
зупиняють запуск. Випадково ініціалізованої запасної моделі немає.
`--dronet-weights PATH` дозволяє інше розташування того самого офіційного файла;
довільні донавчені ваги цей адаптер не приймає.

У поточному середовищі встановлений CPU-only PyTorch. `--device auto` обере CPU.
Наявність MX250 сама по собі не вмикає CUDA. `--device cuda` потребує сумісної
CUDA-збірки PyTorch і за її відсутності завершиться помилкою.

## Офлайн-запуск

```powershell
# Повна модель v3
.\venv\Scripts\python.exe tello_nomad_autopilot.py --model dronet-v3 --video flight_live_last.mp4 --no-show --max-frames 120

# Tiny v3
.\venv\Scripts\python.exe tello_nomad_autopilot.py --model dronet-v3-tiny --video flight_live_last.mp4 --no-show --max-frames 120

# NoMaD + окремий щит DroNet v3
.\venv\Scripts\python.exe tello_nomad_autopilot.py --model nomad --dronet --video flight_live_last.mp4 --no-show --max-frames 120

# Порівняння тих самих кадрів, JSON із затримками та CSV із прогнозами
.\venv\Scripts\python.exe benchmark_dronet_v3.py --video flight_live_last.mp4 --device cpu --max-frames 120
```

Для щита з NoMaD/ViNT доступне `--dronet-variant tiny-v3`. Самостійну модель
обирає `--model`; ці два параметри мають різне призначення.

## Живий режим

```powershell
.\venv\Scripts\python.exe tello_nomad_autopilot.py --live --model dronet-v3
```

Підключення та зліт залишаються окремими діями: `t` — зліт, `n` — автопілот,
пробіл — зависання, `l`/`q` — посадка. Цю інтеграцію перевірено офлайн; реальний
політ не виконувався. DroNet працює реактивно: фотографія-ціль, топологічна
пам'ять і VLFly-підцілі для нього не підтримуються. Для цих задач потрібні
`--model nomad` або `--model vint`.

## Як формується керування

- Сирий BGR-кадр переводиться у grayscale, вирізається центральна область
  200×200, значення діляться на 255. Це відповідає офіційному testing transform;
  кадр не проходить ImageNet-нормалізацію або resize буфера NoMaD.
- Позитивний steer задає лівий поворот; RC-мапінг Tello змінює знак на його
  clockwise-конвенцію. Боковий рух не додається.
- Модель повертає ймовірність зіткнення, а не метричну відстань. У HUD та CSV
  показується `Pcoll`, вигаданих метрів немає.
- Швидкість зменшується зі зростанням ризику. При `Pcoll >= 0.80` forward
  обнуляється після EMA. Пороги `--dronet-caution` і `--dronet-stop` можна
  задавати явно; вони не є відкаліброваною гарантією відсутності зіткнень.
- Червоний стан евристичного щита та `TRAPPED` також блокують forward.
  У live-циклі блокування перевіряється після вибору recovery-команди.
- Для NoMaD + DroNet це окремий захисний модуль у тому самому циклі pipeline,
  а не незалежний високочастотний потік. Контроль свіжості відео й окремий
  керуючий цикл залишаються подальшими покращеннями.

У live CSV додані `nav_model`, `dronet_variant`, `coll_prob`, `dronet_steer`,
`dronet_state`. Офлайн benchmark не підключається до Tello. Його FPS описує
обчислення, а не якість польоту; відео з накладеним HUD непридатне для чистої
оцінки якості розпізнавання без додаткових застережень.

## Джерела та перевірки

- [Офіційний репозиторій і навчені ваги](https://github.com/pulp-platform/pulp-dronet/tree/98f15cfe2e9c34c3a546358e22e45b3b09f11a79/tiny-pulp-dronet-v3)
- [Атрибуція та зміни локального inference-коду](../nomad_pipeline/models/vendor/README.md)
- [Ліцензія Apache-2.0](../nomad_pipeline/models/vendor/PULP_DRONET_LICENSE)

```powershell
.\venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
```

Інтеграційні тести офіційних ваг пропускаються, якщо їх ще не завантажено;
решта перевірок використовує контрольовані прогнози без мережі та дрона.

### Результат локальної перевірки 27.10.2026

36 тестів пройшли, включно з обома завантаженими офіційними checkpoint.
DroNet v3 також обробив усі 578 кадрів останнього запису без помилок.

Окремий послідовний benchmark перших 120 кадрів на CPU, 4 потоки, без запису
відео, але з HUD та перевірками pipeline:

| Модель | Середній час кроку | Обчислювальний FPS |
|---|---:|---:|
| DroNet v3 | 24,4 мс | 41,0 |
| Tiny v3 | 10,0 мс | 99,7 |
| ViNT | 121,7 мс | 8,2 |
| NoMaD, 1 семпл / 2 ітерації | 125,1 мс | 8,0 |

Це не частота камери або перевірка успішного польоту. Tiny сформував аварійну
зупинку на 117 із 120 кадрів, повна v3 — на 62. Без розмітки перешкод ці числа
не визначають точність; запис уже містить HUD, що впливає на вхід моделі.
Тому типовою обрано повну v3, а Tiny залишено варіантом для порівняння.

Деталі цього запуску: `analysis_outputs/dronet_benchmark/summary.json` і
`analysis_outputs/dronet_benchmark/predictions.csv`.
