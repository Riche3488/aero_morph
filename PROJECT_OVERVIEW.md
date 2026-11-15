# aero_project 전체 정리: 차량 외부 공력 설계 파이프라인

> 작성일 2026-11-15 · 대상 브랜치 `aero-pipeline`
> 이 문서는 프로젝트에 구현된 코드를 **사용자 관점**(무엇을 할 수 있는가)과 **개발 관점**(어떻게 구현되어 있는가)으로 정리하고, 남은 단계와 확장 방향을 제안합니다. 명령어 상세는 각 폴더의 `README.md`가 기준입니다.

---

## 1. 한눈에 보기

```
 사진 ──► gen3d ──┐
                  ├─► 1_vehicle_geometry ──► 2_shape_morphing ──► 3_cfd ──► 4_dl_training ──► 5_inference
 파라메트릭 모델 ─┘     기준 형상(STL)          설계안 + 변형 형상        OpenFOAM     대리모델(계획)       빠른 예측·최적화(계획)
                                               (STL + meta.json)       Cd, CdA, Cl
```

| 단계 | 폴더 | 상태 | 한 줄 요약 |
|---|---|---|---|
| 0 사진 → 3D | `gen3d/` | ✅ 구현 | Wikimedia 사진으로 TRELLIS 3D 생성 → 대칭·수밀(watertight) 차량 모델 |
| 0+ 형상 정제 | `gen3d/smooth_refine.py` | ✅ 구현 | 생성 모델의 타이어 돌출·윈드실드 함몰·번호판 등 결함을 SDF로 수정 |
| 1 기준 형상 | `1_vehicle_geometry/` | ✅ 구현 | 공개 제원 기반 IONIQ 5 파라메트릭 모델 (변수 9개) |
| 2 형상 모핑 | `2_shape_morphing/` | ✅ 구현 | 파라메트릭 / 치수(FOH·ROH·길이·폭·스포일러) / 드래그(RBF) 모핑, DOE, GUI 2종, 형상 세트 연동 |
| 3 CFD | `3_cfd/` | ✅ 구현 | OpenFOAM 12 가상 풍동 자동화 (메시→해석→후처리→비교 리포트) |
| 4 딥러닝 학습 | `4_dl_training/` | 📝 계획 | 데이터 규약만 정의 (README) |
| 5 추론·최적화 | `5_inference/` | 📝 계획 | 사용 시나리오만 정의 (README) |

코드 규모: Python/HTML/배치 약 9,700줄 (자동 생성된 케이스별 `post.py` 사본 포함).

---

## 2. 사용자 관점: 무엇을 할 수 있나

### 2.1 사진 몇 장으로 새 차량의 3D 모델 만들기
- Wikimedia Commons에서 라이선스 정보와 함께 사진을 검색합니다.
  ```bash
  cd gen3d && .venv/bin/python pipeline.py search "Kia EV6"
  ```
- `vehicles/<id>.yaml`에 제원(전장·전폭·전고·휠베이스·오버행·윤거·타이어)과 사진을 적고 실행합니다.
  ```bash
  .venv/bin/python pipeline.py run vehicles/<id>.yaml
  ```
- 결과는 다음과 같습니다.
  - 좌우 대칭 오차 0 mm, 수밀이고 제원 치수에 맞춘 STL/GLB/OBJ
  - 후보 비교 시트, 검사 이미지, `report.json`
- 검증된 차량: IONIQ 5, IONIQ 6 (RTX 5070 Ti 기준 TRELLIS 1회 약 8 s, repair 약 60 s).

### 2.2 생성 모델의 결함을 CFD에 쓸 수 있게 다듬기
`smooth_refine.py`는 입력 표면으로 정확한 SDF(부호 거리장)를 만든 뒤, 알려진 결함 유형을 SDF 위에서 직접 고칩니다.

| 결함 | 처리 |
|---|---|
| 타이어 아래 돌출("발") | 타이어 원 밖의 재료를 잘라내고 작은 평탄 접지면 생성 |
| 윈드실드 움푹 패임 | 저차 대칭 다항식 곡면으로 재피팅, A필러 하단 구멍 메움 |
| 번호판 구멍·돌기 | 주변 범퍼에서 장력 스플라인 inpainting으로 재구성 |
| 하부 단차·사이드실 리브 | 높이맵 평활, x방향 압출 |
| 계단 무늬 | SDF Gaussian + Taubin 평활 |

현재 IONIQ 5 기준 형상인 `gen3d/output/ioniq5_fixed.stl`이 이 결과물입니다.
- 약 100만 면, 수밀, Euler 2, 대칭 오차 0 mm
- 차체 폭 1890 mm, 체적 8.76 m³

### 2.3 파라메트릭 IONIQ 5로 설계 변수 연구하기
`1_vehicle_geometry/ioniq5_geometry.py`는 공개 제원으로 만든 단순화 외형입니다. DrivAer "smooth underbody" 수준입니다. 바퀴·미러는 별도 부품으로 나옵니다.

설계 변수는 9개입니다.

| 변수 | 범위 |
|---|---|
| `ride_height` | −25 … 30 mm |
| `hood_height` | ±40 mm |
| `front_corner` | −20 … 80 mm |
| `windshield_rake` | ±100 mm |
| `roof_rear_drop` | −20 … 60 mm |
| `spoiler_length` | ±60 mm |
| `tail_taper` | 0 … 80 mm |
| `diffuser_angle` | −2 … 6° |
| `mirror_scale` | 0 … 1.2 |

### 2.4 어떤 STL이든 치수 변경하기 (치수 모핑)
STL 기반 스터디에서 쓰는 변수 5개입니다. 단위는 mm, 0이 기준 형상입니다.

| 변수 | 의미 | 범위 |
|---|---|---|
| `dim:foh` | 앞 오버행 (+ 앞으로 늘어남) | ±100 |
| `dim:roh` | 뒤 오버행 (+ 뒤로 늘어남) | ±100 |
| `dim:length` | 휠베이스 (뒤 차축 이동) | ±150 |
| `dim:width` | 차체 폭 (윤거가 따라감) | ±60 |
| `dim:spoiler` | 루프 스포일러 끝단 높이 | −30 … +50 |

- 바퀴는 강체로 이동하므로 타이어 형상과 휠아치 간격이 유지됩니다.
- 변형장의 야코비안 행렬식을 해석적으로 계산하므로, 범위 안에서는 접힘(fold)이 생기지 않음이 보장됩니다.
- 랜드마크(바퀴 구역, 차체 폭, 스포일러 끝단)는 자동 검출하고, `study.json`에서 수동으로 덮어쓸 수 있습니다.

### 2.5 브라우저에서 차를 직접 끌어서 변형하기 (치수 GUI)
```bash
python3 2_shape_morphing/dim_gui.py ioniq5        # http://localhost:8766
```
- **핸들:** 변수마다 색상 핸들이 있습니다(코·꼬리·뒷바퀴·옆판·스포일러). 화살표 방향으로 끌면 해당 치수가 바뀝니다.
- **표면 드래그:** 차체 아무 곳이나 잡고 끕니다. 그 점의 변수별 민감도와 드래그 방향을 보고 어떤 변수를 움직일지 자동으로 고릅니다. 예: 코 → FOH, 꼬리 → ROH, 옆면 좌우 → 폭.
- **보조 기능:**
  - Shift는 10 mm 단위, Ctrl+Z는 되돌리기입니다.
  - 슬라이더와 숫자 입력을 함께 씁니다.
  - 치수 표와 det J 하한, 중심선·단면 프로파일을 보여 줍니다.
- **빠른 미리보기:** 브라우저는 약 20만 면짜리 미리보기 메시에 Python과 똑같은 변형장을 JS로 적용합니다(오차 0.0000 mm). 고해상도 메시는 만들지 않습니다.
- **필요할 때만 고해상도:**
  - *Check in Python*: 전체 해상도로 검증하고 실측 치수, det J, 뒤집힌 삼각형 수를 돌려줍니다.
  - *Add to designs*: 설계표에 추가합니다.
  - *Export case*: CFD용으로 export합니다.

### 2.6 자유 형상 변형을 그려서 설계 변수로 저장하기 (드래그/RBF GUI)
```bash
python3 2_shape_morphing/morph_gui.py <any.stl>   # http://localhost:8765
```
Windows에서는 `run-local.bat`을 더블클릭하거나 STL을 끌어다 놓아도 됩니다.
- 표면을 잡아 핸들을 만들고 끌어 변형합니다. 앵커로 고정점을 지정합니다.
- 반경과 축별 감쇠, y 대칭, 지면 고정을 설정할 수 있습니다.
- 변형 크기 색상맵과 min det J 접힘 검사를 실시간으로 보여 줍니다.
- 저장한 *모드*(`modes/<name>.json`)는 설계 변수 `mode:<name>`이 되고, 진폭으로 조절합니다(예: `mode:rear_roof_drop=1.5`).
- 어느 STL에서 그린 모드인지 기록되므로, 다른 기준 형상과 섞어 쓰면 거부됩니다.
- CLI로 일괄 적용할 수도 있습니다.
  ```bash
  python3 2_shape_morphing/drag_morph.py --in car.stl --mode modes/x.json=1.0 --out out --render
  ```

### 2.7 설계안 표(DOE) 만들기
```bash
python3 2_shape_morphing/doe.py ioniq5 --oat dim:foh dim:roh dim:length dim:width dim:spoiler   # 한 변수씩 범위 끝값
python3 2_shape_morphing/doe.py ioniq5 --lhs 20 --params dim:foh dim:roh --seed 1               # 라틴 하이퍼큐브
python3 2_shape_morphing/doe.py rear --sweep mode:rear_roof_drop=0.5,1,1.5                     # 스윕
python3 2_shape_morphing/doe.py rear --add tail_taper=40 mode:front_bumper_push=1              # 조합 1개
```
- 호출할 때마다 `designs.csv`에 추가하며, 중복은 건너뜁니다. c000은 항상 기준 형상입니다.
- 한 설계 안에서 파라메트릭 + 모드, 또는 치수 + 모드를 함께 쓸 수 있습니다.

### 2.8 이미 있는 형상 세트(1.stl … n.stl)를 파이프라인에 연결하기
매니페스트 형식은 다음과 같습니다(txt / csv / json 지원).
```text
1.stl : -40mm FOH
2.stl : +50mm ROH, +20 mm spoiler
```
```bash
python3 2_shape_morphing/dim_gui.py ioniq5 --set path/to/set                     # GUI "Shape set" 패널에서 탐색
python3 2_shape_morphing/shape_set.py path/to/set --study ioniq5 --compare      # 편차 리포트
python3 2_shape_morphing/shape_set.py path/to/set --study ioniq5 --to-designs   # 값만 설계표로 (모퍼 형상)
python3 2_shape_morphing/shape_set.py path/to/set --study ioniq5 --import NEW   # 주어진 STL 자체를 새 스터디로
```
- **비교:** 모퍼에 같은 값을 넣은 형상과 주어진 STL을 정렬한 뒤 부호 편차의 p99로 판정합니다.
  - 2 mm 이하: match
  - 10 mm 이하: close
  - 그 이상: differs
  - 매니페스트 오기재(예: +30을 +50으로 적음)는 differs로 드러납니다.
- **GUI 보기:** 모퍼 / 원본 / 겹쳐보기 / 편차 색상맵. ← →로 세트를 넘겨 봅니다.
- **import:** 주어진 형상 그대로 CFD를 돌리고, 값↔형상 쌍을 학습 데이터로 쓸 수 있습니다.

### 2.9 형상 export와 확인
```bash
python3 2_shape_morphing/export_designs.py ioniq5    # geometry/<case>/*.stl + meta.json + index.json
python3 2_shape_morphing/preview_designs.py ioniq5   # 기준 대비 중심선·평면·단면 비교 PNG
```
- `meta.json`에는 설계 변수(기준값·범위·단위), 전면 투영 면적, 바퀴 위치, bbox, 품질 검사(det J, 뒤집힘, 바퀴 간섭)가 들어갑니다.
- 서명(signature) 캐시가 있어서, 설계·모드·기준 형상·관련 코드가 바뀐 케이스만 다시 만듭니다.

### 2.10 CFD 일괄 실행과 결과 비교
```bash
python3 3_cfd/run_study.py ioniq5 --clean                  # index.json의 모든 케이스
python3 3_cfd/run_case.py  ioniq5 c003 --stop-after mesh   # 개별 케이스·단계 제어
python3 3_cfd/compare.py   ioniq5                          # 순위·민감도·최적 프로파일 리포트
```
- **조건:**
  - 해석: 정상 RANS k-ω SST, 100 km/h
  - 경계: 이동 벨트, 회전 바퀴, 반모델(대칭면)
  - 메시: 약 160만 셀, 8 MPI, 케이스당 약 20분
- **단계:** prepare → mesh → init → solve → post. 각 단계는 중단된 곳부터 재개합니다. 실패 케이스는 기록만 하고 다음 케이스로 넘어갑니다.
- **초기장:** 같은 스터디의 수렴된 c000을 매핑합니다. 없으면 potentialFoam으로 냉시동합니다.
- **안전장치:**
  - export 후 형상이 바뀐 케이스는 "changed"로 표시합니다. `--rerun-changed`나 `--redo prepare`로 다시 돌립니다.
  - 중단된 실행의 잔여 출력이 Cd 평균에 섞이지 않습니다.
- **결과:**
  - `runs/<study>/results.csv`: 설계 변수 → Cd, Cd_std, CdA, Cl, Cl_front/rear, 항력, 면적, 셀 수. 4단계 학습 테이블로 씁니다.
  - `report/`: ranking, sensitivity, profiles_best 그림
  - 케이스별 표면·단면 VTK

---

## 3. 단계별 구현 상세

### 3.0 `gen3d/`: 사진 → 3D (전용 `.venv`, GPU)
| 파일 | 역할 |
|---|---|
| `pipeline.py` | CLI `search` / `run`, 단계 fetch → generate → select → repair → render |
| `car3d/spec.py` | YAML → Spec (mm→m), repair 프리셋(rough 12 mm / fine 6 mm 복셀) |
| `car3d/images.py` | Wikimedia 검색·다운로드, 라이선스 기록(`ATTRIBUTION.md`), BiRefNet 배경 제거 |
| `car3d/generate.py` | TRELLIS 로딩·샘플링, RTX 50xx(sm_120) 우회(xformers FA2 강제), Gaussian 정점 색 |
| `car3d/align.py` | 차량 좌표계 정렬, 슬리버 제거, L/W/H 맞춤 |
| `car3d/score.py` | 후보 자동 점수(바퀴 4개 존재, 좌우 일치). 기하만 보며 사진 충실도는 보지 않음 |
| `car3d/repair.py` | 대칭면 탐색 → 복셀 솔리드 → 평탄 바닥 → 거울 SDF 평균 → 반쪽 decimate 후 미러 → 차축 warp → 제원 맞춤 |
| `car3d/render.py` | Matplotlib 미리보기 (OpenGL 불필요) |
| `smooth_refine.py` | 고해상도 정밀 SDF(winding number + 정확 거리) 기반 결함 수정. 2.2절 참고 |
| `shims/kaolin` | torch 2.7용 kaolin 대체 (FlexiCubes가 쓰는 함수 1개) |
| `legacy/` | 초기 일회성 스크립트 보관 |

- 라이선스: Hunyuan3D 2.0은 한국·EU·영국을 제외하므로 TRELLIS(MIT)를 씁니다.
- 사진 저작자 표시는 `runs/<id>/ATTRIBUTION.md`에 있습니다.

### 3.1 `1_vehicle_geometry/`
| 파일 | 역할 |
|---|---|
| `ioniq5_geometry.py` | 키포인트 프로파일 loft로 차체·바퀴 4개·미러 2개 생성. `MORPH_PARAMS`, `set_design()` |
| `preview.py` | 미리보기 렌더 |
| `gen3d_image_to_3d/` | `../gen3d` 링크 |

전면 투영 면적은 약 2.65 m²입니다. OEM CAD가 아니므로 절대 Cd보다는 경향·비교 연구용입니다.

### 3.2 `2_shape_morphing/`
| 파일 | 역할 |
|---|---|
| `design_table.py` | `designs.csv` 읽기·쓰기, 변수 종류(파라메트릭 / `dim:` / `mode:`) 판별, 검증, 라벨 |
| `doe.py` | OAT / sweep / LHS / add, `--base-stl`로 STL 스터디 생성, `--list` |
| `export_designs.py` | 형상 생성 → STL + `meta.json` + `index.json`. 서명 캐시(설계·모드·기준 형상·코드 해시), 바퀴 고정 마스크, 다중 영역 `body.stl`(body / wheelFront / wheelRear), import 케이스 처리 |
| `dim_morph.py` | 치수 모핑 엔진. 1-D C1 전이 함수로 만든 변위장 `u = (D(x), y·Δw/W, Δs·w_x·w_z)`, 해석적 야코비안, 랜드마크 검출, 실측(`measure`), 품질 판정(OK / FOLD RISK / FOLDED) |
| `rbf.py` | RBF 모핑 엔진. Wendland C2 컴팩트 지지 커널, 핸들·앵커 정확 보간, 품질 리포트 |
| `drag_morph.py` | RBF 모드 일괄 적용 CLI |
| `preview_designs.py` | 기준 대비 단면 비교 그림 |
| `dim_gui.py` + `dim_editor.html` | 치수 GUI 서버(ThreadingHTTPServer) + three.js 편집기 |
| `morph_gui.py` + `drag_editor.html` | RBF 드래그 GUI 서버 + 편집기 (JS와 `rbf.py` 결과 일치, 1e-12) |
| `shape_set.py` | 매니페스트 파싱, trimmed point-to-plane 정렬, 편차 비교, 설계표 / import 연계 |
| `mesh_preview.py` | GUI용 경량 메시 (vertex clustering, 뒤집힌 면 제거) |
| `local_server.py` | 로컬 GUI 보안 검사: Host / Origin 확인(DNS rebinding, CSRF 차단), POST는 JSON만 허용 |
| `modes/` | 저장된 RBF 모드 4개 (`front_bumper_push`, `rear_roof_drop`, `ioniq6_rear_taper`, `ioniq6_trunk_drop`) |
| `studies/<study>/` | `study.json`(기준 형상·바퀴·랜드마크·imports), `designs.csv`, `geometry/`, `previews/` |

GUI 서버 API (dim_gui):

| 메서드 | 엔드포인트 |
|---|---|
| GET | `/api/info`, `/api/mesh.bin`, `/api/designs`, `/api/set`, `/api/set/mesh.bin?i=` |
| POST | `/api/check`, `/api/add_design`, `/api/export?case=`, `/api/set/compare`, `/api/set/to_designs`, `/api/set/import` |

### 3.3 `3_cfd/`
| 파일 | 역할 |
|---|---|
| `run_case.py` | 케이스 1개 단계별 실행. 템플릿 채우기(`@GEOMETRY@`, `@AREF@`, 바퀴 중심·회전수·lRef), 형상 해시(헤더 제외, 스트리밍, 디스크 캐시), 재개 마커, 잔여 function-object 출력 정리 |
| `run_study.py` | 스터디 일괄 실행, changed / failed 케이스 처리, 끝나면 `compare` 실행 |
| `compare.py` | `results.csv` 생성, 순위·민감도·최적 프로파일 그림 |
| `template/` | OpenFOAM 사전 템플릿, `post.py`(최신 dat 선택, Cd 평균·표준편차, 그림) |
| `baseline_case/` | 원본 IONIQ 5 기준 해석(1500 it, Cd 0.315), 매핑 초기장 원본. `make_geometry.py`는 res 0.5 export만 허용 |

### 3.4 공통
| 파일 | 역할 |
|---|---|
| `pipeline_config.py` | 경로·핸드오프 위치, `load_module()`(단계 간 유일한 코드 의존을 지연 로드) |
| `common/mesh_io.py` | STL 입출력(binary, multi-solid ASCII), 수밀 검사, 부피, 전면 투영 면적, 평면 절단 |
| `run-local.bat` | Windows → WSL 드래그 GUI 런처 |

---

## 4. 데이터 규약

- **좌표계(모든 단계 공통):**
  - x: 앞 범퍼에서 뒤쪽이 +
  - y: 왼쪽이 +, y=0이 대칭면
  - z: 위쪽이 +, z=0이 지면
  - 단위는 m입니다. GUI와 설계표의 치수 변수만 mm입니다.
- **단계 독립성:** 단계 사이에는 파일로만 데이터를 넘깁니다. 각 스크립트는 자기 폴더와 `common/`만 import합니다. 예외는 파라메트릭 스터디가 1단계 생성기를 지연 로드하는 것 하나입니다.

| 핸드오프 | 위치 | 내용 |
|---|---|---|
| 1 → 2 | `MORPH_PARAMS` 또는 임의 STL | 기준 형상과 변수 |
| 2 → 3 | `2_shape_morphing/studies/<study>/geometry/` | `index.json`(실행할 케이스), `<case>/*.stl`, `<case>/meta.json` |
| 3 → 4 | `3_cfd/runs/<study>/results.csv`, `cases/<case>/` | 실행된 변수 → Cd·CdA·Cl, 표면·단면 필드(VTK) |
| 4 → 5 | `4_dl_training/checkpoints/<name>/` | 학습 모델 (계획) |

---

## 5. 품질 보증 장치

| 위치 | 장치 |
|---|---|
| 형상 | 모든 export STL의 수밀 검사. RBF는 min det J > 0.2를 강제합니다. 치수 모핑은 해석적 det J 하한과 실측 뒤집힘 검사를 함께 씁니다. 바퀴 간섭 검사 |
| 일관성 | 서명 캐시가 코드 변경까지 감지해 재export합니다. CFD는 형상 해시로 "export 후 바뀐 케이스"를 잡아냅니다 |
| GUI = 엔진 | JS 변형장이 Python 엔진과 같습니다(치수 0.0000 mm, RBF 1e-12) |
| 형상 세트 | p99 편차로 매니페스트 오류를 검출합니다. 중복 값과 범위 밖 값은 경고로 알립니다 |
| CFD 재현성 | 단계 마커로 재개하고, 잔여 출력을 정리합니다. 후처리는 항상 최신 `post.py`를 씁니다. 매핑 초기장 기준 해석이 전체 해석을 재현합니다(0.3169 vs 0.3154) |
| 보안 | GUI는 127.0.0.1에만 바인딩합니다. Host / Origin 검사와 JSON Content-Type 요구로 외부 웹페이지의 요청을 차단합니다 |

---

## 6. 현재 데이터 현황 (2026-11-15)

| 스터디 | 기준 형상 | 변수 | 설계 | export | CFD |
|---|---|---|---|---|---|
| `demo` | IONIQ 5 파라메트릭 | 파라메트릭 9개 + 모드 2개 | c000–c006 | ✅ | c000, c001 완료. c002, c005, c006은 중간에 중단(사용자 요청으로 일시정지) |
| `ioniq6` | gen3d IONIQ 6 rough STL | 모드 2개 | c000–c005 | ✅ | c000 메시까지(154만 셀) |
| `ioniq5` | `gen3d/output/ioniq5_fixed.stl` | `dim:` 5개 | c000–c010 (OAT) + c011–c015 | c000–c010 ✅ (약 2.2 GB) | 미실행 |

- 완료된 결과: demo c000 기준 Cd 0.3169 (CdA 0.840). c001 `tail_taper=80mm`은 Cd 0.2999 (CdA 0.795, −5.4 %).
- 반복 노이즈는 약 ±6 counts입니다. 전면 면적이 바뀌는 설계는 CdA로 비교해야 합니다.
- ioniq5의 c011–c015는 형상 세트 데모(`--to-designs`) 시험 중에 추가된 행입니다. 아직 export하지 않았고, 필요 없으면 지워도 됩니다.

---

## 7. 알려진 한계

- **형상 충실도:** 파라메트릭 모델과 사진 생성 모델 모두 OEM CAD가 아닙니다. 하부는 평탄하고, 그릴·냉각 유동, 도어 갭, 와이퍼가 없습니다. 절대 Cd 검증용이 아니라 상대 비교용입니다.
- **CFD 모델:**
  - 정상 RANS, 반모델이라 요(yaw)·비대칭 조건을 다룰 수 없습니다.
  - 메시는 약 160만 셀, 경계층 2층(wall function)입니다.
  - 풍동·실차 데이터로 검증하지 않았습니다.
- **변수 체계:**
  - 파라메트릭 변수는 IONIQ 5 생성기에서만, `dim:` 변수는 STL 기반에서만 쓸 수 있습니다.
  - 모드는 그린 기준 형상에서만 쓸 수 있습니다.
- **랜드마크 의존:** 치수 모핑은 차종마다 바퀴 위치와 스포일러 끝단 검출에 의존합니다. 검출이 약하면 경고가 나오며, 그때는 `study.json`에서 수동으로 지정합니다.
- **자동화 테스트 부재:** 지금까지 검증은 일회성 스크립트와 헤드리스 브라우저 E2E로 했습니다. 저장소에 회귀 테스트 묶음은 없습니다.
- **실행 환경:** 단일 워크스테이션(WSL2, 8코어, RTX 5070 Ti)을 전제로 합니다. GUI는 로컬 전용입니다.

---

## 8. 향후 확장 가능성

### 8.1 바로 이어서 할 수 있는 것 (기존 코드로 가능)
1. **ioniq5 치수 스터디 CFD 실행:** `run_study.py ioniq5`로 11 케이스(약 4시간)를 돌리면 FOH·ROH·길이·폭·스포일러의 Cd/CdA 민감도를 얻습니다.
2. **LHS 확장:** `doe.py ioniq5 --lhs 30 ...`로 4단계 학습에 쓸 다변수 샘플을 만듭니다.
3. **demo 스터디 재개:** 남은 케이스를 끝내 파라메트릭 변수 민감도를 완성합니다.
4. **형상 세트 import:** 외부에서 받은 형상(설계 대안, 스캔)을 그대로 CFD에 연결합니다.

### 8.2 4·5단계: 대리모델과 최적화 (README에 규약 정의됨)
| 기능 | 내용 | 기반 |
|---|---|---|
| 변수 → Cd/CdA/Cl | Gaussian Process 또는 MLP. 수십 샘플로 가능하고 불확실도를 줍니다 | `results.csv` |
| 능동 학습 | 불확실도가 큰 설계를 `doe.py --add`로 되돌려 CFD 실행 → 재학습 | DOE·CFD 자동화가 이미 있음 |
| 형상 → Cd | PointNet / GNN을 `body.stl`에 적용. 변수 체계와 무관하게 모드·임의 형상까지 다룹니다 | `constant/geometry/*.stl` |
| 형상 → 표면 Cp | GNN / neural operator | `postProcessing/car/*/patch.vtk` |
| 최적화 | 대리모델 위에서 CdA 최소화(치수·패키징 제약 포함) → 상위 설계를 CFD로 검증 | `5_inference/predict.py` (계획) |
| **GUI 실시간 Cd** | dim_gui에 `/api/predict`를 추가해 드래그 중 예상 Cd/CdA와 불확실도를 표시 | GUI가 이미 변수 값을 실시간으로 계산함 |

### 8.3 형상 모핑 확장
- **치수 변수 추가:** `dim_morph`의 구조(1-D 전이 함수 + 하삼각 야코비안)를 따르면 접힘 보장을 유지한 채 변수를 늘릴 수 있습니다.
  - 후보: 전고 / 루프 높이, 지상고, 윈드실드 경사, 후방 디퓨저 각, 범퍼 코너 R, 테일 테이퍼
- **역추정:** 주어진 STL에 가장 잘 맞는 치수 값을 편차 최소화로 찾습니다. `shape_set.compare`의 정렬·편차 계산을 목적함수로 재사용합니다. 매니페스트가 없는 형상 세트도 자동으로 라벨링할 수 있습니다.
- **볼륨 메시 모핑:** RBF와 치수 변형장은 모두 공간 전체에서 정의되므로, CFD 볼륨 메시를 직접 변형할 수 있습니다. 그러면 케이스마다 snappyHexMesh를 다시 돌리지 않아 메시 시간이 줄고, 메시 차이에서 오는 노이즈도 줄어듭니다.
- **adjoint 민감도 연계:** OpenFOAM adjoint로 표면 민감도를 구해 드래그 모드나 치수 변수 방향을 제안합니다.

### 8.4 CFD 확장
- 전체 모델 + 요 각(측풍), 비대칭 형상
- 고해상도 메시, 경계층 증가, 비정상(DES/LES)으로 박리 정확도 향상
- 휠 회전 모델 고도화(MRF / 슬라이딩 메시), 냉각 유동(그릴 개구)
- 클러스터 / 클라우드 배치(작업 큐), 케이스 병렬 실행
- 공개 DrivAer 데이터로 검증 케이스 구성

### 8.5 형상 생성(gen3d) 확장
- 여러 차종(세단·SUV·해치백)을 일괄 생성해 차종을 넘나드는 데이터셋 구축 → 차종 일반화 대리모델
- `smooth_refine` 결함 수정을 다른 차종에도 적용하고 검증(현재는 IONIQ 5에서 튜닝)
- fine 프리셋, 텍스처 베이크(nvdiffrast)

### 8.6 엔지니어링 기반
- 회귀 테스트(pytest): 엔진 결과값 일치, 서명 캐시, 매니페스트 파서, GUI API
- 결과 대시보드(스터디별 Cd 비교, 형상 뷰어 연동)
- 환경 패키징(Docker / conda), 실행 로그·메타데이터 표준화

---

## 9. 부록: 실행 환경

| 대상 | 인터프리터 / 도구 |
|---|---|
| 1–3단계 | 시스템 `python3` (numpy, scipy, matplotlib). trimesh 불필요 |
| gen3d | `gen3d/.venv/bin/python` (torch 2.7.1+cu128, TRELLIS, open3d, scikit-image, trimesh) |
| CFD | OpenFOAM 12 (`/opt/openfoam12/etc/bashrc`), 8 MPI |
| GUI | 로컬 브라우저, three.js r128 (CDN). dim_gui 8766 / morph_gui 8765 포트 |
