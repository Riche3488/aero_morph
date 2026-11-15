# test_ioniq5 — IONIQ 5 실제 수행 결과 모음

Hyundai IONIQ 5 한 차종에 대해 이 프로젝트 파이프라인을 실제로 돌린 결과를 단계별로 모아 둔 폴더입니다.
원본은 각 파이프라인 폴더에 그대로 있고, 여기에는 **복사본**만 있습니다.
`python3 test_ioniq5/collect_results.py`로 언제든 다시 갱신할 수 있습니다.

```
사진 2장 ──► 01 gen3d 복원 ──► 02 기준 형상 수리 ──► 03 치수 morphing (11 designs) ──► 04 OpenFOAM CFD
(Wikimedia)    TRELLIS           ioniq5_fixed.stl        FOH/ROH/길이/폭/스포일러          c000 해석 완료
                                                                                       c001–c010 mesh 완료
```

## 요약

| 항목 | 값 |
|---|---|
| 기준 형상 | `02_base_geometry/ioniq5_fixed.stl`: 1.0 M 삼각형, watertight, 좌우 대칭, 4635 × 1890(차체) × 1623 mm |
| CFD 조건 | OpenFOAM 12, steady RANS k-ω SST, 100 km/h, moving belt + 회전 바퀴, 대칭면 기준 half model, 8 MPI |
| 메시 | snappyHexMesh 1.59 M cells, 경계층 커버리지 98.9 %, checkMesh 1개 항목만 경고(highly skew face 46개) |
| **c000 결과** | **Cd = 0.308 ± 0.005**, CdA = 0.793 m², Cl = −0.074 (앞 −0.064 / 뒤 −0.009), 항력 368 N, 100 km/h 공력 손실 10.2 kW |
| 참고 | Hyundai 공식 Cd는 0.288입니다. 해석값이 약 7 % 높은데, 사진으로 복원한 형상이고 휠 디테일·냉각 흐름·하부 미세 형상을 단순화했으므로 이 정도 차이는 예상 범위입니다. |
| 소요 시간 | mesh 84 s, potentialFoam cold start 후 1500 it 해석 30.5 min |

## 폴더 구조

| 경로 | 내용 |
|---|---|
| `01_gen3d_reconstruction/` | 사진 → 3D 복원 단계 |
| ├ `inputs/`, `inputs_sheet.png`, `ATTRIBUTION.md` | 입력 사진 2장(앞-좌 3/4, 뒤-우 3/4)과 라이선스 표기. **fl 사진은 CC BY-SA 4.0이라 파생 모델도 출처 표기가 필요합니다.** |
| ├ `candidates/`, `candidates_sheet.png` | TRELLIS 후보 4개(seed별)의 점수(`info.json`)와 렌더. `orig_fl+rr_s1`이 선택됐습니다. |
| ├ `ioniq5_rough.*`, `ioniq5_final.glb` (+ `_views` / `_inspect`) | 선택한 후보의 rough mesh와 최종 mesh |
| └ `report.json` | 제원(spec_mm), 후보 점수, 단계별 소요 시간 |
| `02_base_geometry/` | CFD에 쓰는 기준 형상 |
| ├ `ioniq5_fixed.stl` | 수리 완료한 기준 형상(50 MB, binary STL) |
| └ `ioniq5_fixed_report.json`, `ioniq5_fixed_views.png` | 대칭화, 축 위치 보정, 타이어 접지면, 하부·번호판 평탄화, 최종 스케일 맞춤 기록 |
| `03_shape_morphing/` | 치수 morphing 스터디 |
| ├ `study.json` | 기준 형상과 바퀴 위치 정의 |
| ├ `designs.csv` | DOE 표. c000–c010은 OAT(한 번에 한 변수) 설계입니다. **c011–c015는 이전 GUI 테스트에서 들어간 행이라 export되지 않았습니다.** |
| ├ `export_index.json`, `designs/cXXX/meta.json` | export된 11개 설계의 정면 면적, bbox, 바퀴 위치, 품질 검사 기록 |
| └ `previews/dim_variables_profiles.png` | 변수별 변형 profile 비교 |
| `04_cfd/` | OpenFOAM 해석 |
| ├ `STATUS.md` | 케이스별 상태, 메시, Cd 표(자동 생성) |
| ├ `results.csv`, `report/` | 단계 4 학습용 표와 비교 리포트(ranking, profile). 변형 형상 해석이 끝나면 sensitivity도 추가됩니다. |
| ├ `run_*.log` | 실행 로그 |
| └ `cases/cXXX/` | `result.json`, `status.json`, `case.json`, `setup/`(0 · constant · system dict), `logs/`, `postProcessing/`(force 이력), `plots/` |

## c000(기준 형상) 시각화: `04_cfd/cases/c000/plots/`

기본 post 출력(`ioniq5_wind_tunnel_summary.png`, `convergence.png` 등)에 더해, `3_cfd/visualize_case.py`로 만든 상세 그림이 `viz/`에 있습니다.

| 파일 | 보이는 것 | c000 해석에서 읽히는 점 |
|---|---|---|
| `01_cp_views.png` | 표면 Cp를 측면 / 윗면 / 바닥 / 정면 / 후면에서 본 그림 | 범퍼 정면에 stagnation(Cp ≈ 1), 보닛 앞 모서리와 A-필러·미러에 강한 suction, 후면은 Cp ≈ −0.2로 고르게 낮은 base pressure |
| `02_cp_centreline.png` | 중심선(y = 0)의 윗면·하부 Cp 분포 | 앞 모서리 suction peak, 윈드실드 하단 재압축, 루프 끝에서 압력 회복 |
| `03_drag_buildup.png` | x 방향 누적 Cd, 압력·마찰 분리, 부품별 항력 | 항력의 96 %가 압력 항력. 루프 뒤 테일게이트·후면(x > 4 m)에서 Cd가 0.14 → 0.30으로 급증합니다. 앞바퀴 11 %, 뒷바퀴 6 % |
| `04_skin_friction.png` | Cf와 벽 근처 흐름 방향, 역류 영역(빨강) | 테일게이트 전체와 리어 코너, 미러 뒤쪽에서 박리. 회전 바퀴 위 빨강은 박리가 아닙니다(그림에 주석). |
| `05_centre_planes.png` | y = 0 평면의 Cp·|U|, 바퀴 중심 평면 |U| + 유선 | 후면 뒤 x ≈ 4.6–5.6 m에 위아래 한 쌍의 재순환 vortex, 휠 아치 내부 순환 |
| `06_horizontal_planes.png` | z = 0.5 m, z = 0.10 m(차량 하부) 평면 |U| + 유선 | 하부 고속 흐름(Ux > U∞), 뒷바퀴 후방 wake |
| `07_wake_planes.png` | x = 4.7 / 5.0 / 5.2 / 5.5 / 6.5 m의 total-pressure loss와 단면 속도 벡터 | wake가 좌우 한 쌍의 반대 회전 vortex로 정리되며 회복되는 과정 |
| `08_streamlines.png` | 3D 유선(중심, 측면, 후류 rake)의 측면·윗면 투영 | 루프에서 가속(|U|/U∞ ≈ 1.25)한 뒤 후면에서 감속·재순환 |
| `viewer3d.html` | **브라우저로 여는 인터랙티브 3D 뷰어**. Cp/Cf 표면 전환과 유선 on/off를 지원합니다. | three.js를 CDN에서 불러오므로 처음 열 때 인터넷 연결이 필요합니다. |
| `forces_breakdown.json` | 압력·마찰·부품별 Cd 수치 | 표면 적분 Cd는 0.301로, 같은 iteration의 solver 값 0.315보다 약 4.6 % 낮습니다. 점 데이터를 보간했기 때문이므로, **비율과 x 분포만 참고**하고 Cd 값 자체는 `result.json`을 쓰세요. |

## 변형 형상 10개(c001–c010): 해석 준비 완료

| case | 변경 | 메시 |
|---|---|---|
| c001 / c002 | 앞 오버행 −100 / +100 mm | 1.58 / 1.60 M |
| c003 / c004 | 뒤 오버행 −100 / +100 mm | 1.58 / 1.59 M |
| c005 / c006 | 축간거리(전장) −150 / +150 mm | 1.57 / 1.60 M |
| c007 / c008 | 전폭 −60 / +60 mm(정면 면적 2.495 / 2.654 m²) | 1.57 / 1.60 M |
| c009 / c010 | 스포일러 −30 / +50 mm | 1.59 / 1.59 M |

케이스별로 형상 복사 → snappyHexMesh → decompose까지 완료했고, 경계층 커버리지는 모두 98.8–98.9 %입니다.
Aref(케이스별 정면 면적), lRef(축간거리), 모멘트 중심, 바퀴 위치·회전 속도는 각 설계의 `meta.json` 값으로 들어가 있습니다.

남은 단계는 init(수렴된 c000 해를 mapFieldsPar로 매핑) → solve(800 it) → post입니다. 케이스당 약 20분, 10개 합쳐 약 3.5시간이 걸립니다.

```bash
bash test_ioniq5/run_variants.sh              # 10개 해석 → 케이스별 시각화 → test_ioniq5 갱신
bash test_ioniq5/run_variants.sh c007 c008    # 일부만
```

## 다시 만들기

| 대상 | 명령 |
|---|---|
| 이 폴더 갱신 | `python3 test_ioniq5/collect_results.py` |
| morph STL(케이스당 약 236 MB, 이 폴더에는 미포함) | `python3 2_shape_morphing/export_designs.py ioniq5` |
| c000 해석 | `python3 3_cfd/run_case.py ioniq5 c000` |
| 상세 시각화 | `python3 3_cfd/visualize_case.py ioniq5 <case>` |
| 비교 리포트 | `python3 3_cfd/compare.py ioniq5` |

OpenFOAM 메시와 필드(`processor*`, 케이스당 약 450 MB)는 `3_cfd/runs/ioniq5/cases/`에만 있습니다. c000의 해는 변형 형상의 초기값으로 쓰이므로 지우면 안 됩니다.
