# AutoStr

AutoStr 是一個用來處理中文影片字幕的工具。它會把影片轉成音訊、做語音轉錄、可選擇 WhisperX 對齊，最後輸出整理好的 SRT 字幕。你也可以用它批次掃描資料夾，或另外輸出高光片段。

如果你第一次使用，建議先照著「快速開始」做，不用先看完整份文件。

---

## 快速開始

### 1. 建置映像

CPU 版：

```bash
docker build -t autostr .
```

GPU 版：

```bash
docker build \
  --build-arg BASE_IMAGE=pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime \
  -t autostr:cuda .
```

### 2. 準備資料夾

建立三個資料夾：

```bash
mkdir -p input output models
```

- `input`：放影片或音訊
- `output`：放輸出字幕和高光
- `models`：Whisper 模型快取

### 3. 先跑一個最基本範例

把影片放進 `input` 後，執行：

```bash
docker run --rm \
  -v /path/to/models:/models \
  -v /path/to/input:/input \
  -v /path/to/output:/output \
  autostr /input/input.mp4
```

執行後，字幕會輸出到：

```text
/output/input.srt
```

---

## 最常用的幾種用法

### 產生字幕到指定位置

```bash
docker run --rm \
  -v /path/to/models:/models \
  -v /path/to/input:/input \
  -v /path/to/output:/output \
  autostr /input/input.mp4 -o /output/subtitles.srt
```

### 字幕太早，往後延 150 毫秒

```bash
docker run --rm \
  -v /path/to/models:/models \
  -v /path/to/input:/input \
  -v /path/to/output:/output \
  autostr /input/input.mp4 --start-delay 150
```

### 整體時間都偏掉，整段平移

```bash
docker run --rm \
  -v /path/to/models:/models \
  -v /path/to/input:/input \
  -v /path/to/output:/output \
  autostr /input/input.mp4 --global-shift 500
```

### 想讓每行字少一點

```bash
docker run --rm \
  -v /path/to/models:/models \
  -v /path/to/input:/input \
  -v /path/to/output:/output \
  autostr /input/input.mp4 --max-chars 14
```

### 要輸出高光片段

```bash
docker run --rm \
  -v /path/to/models:/models \
  -v /path/to/input:/input \
  -v /path/to/output:/output \
  autostr /input/input.mp4 --highlights
```
如果同一支影片的字幕已經存在，highlight 模式會直接重用既有字幕，不會再重新抽音訊與轉錄一次。

### 批次處理整個資料夾

```bash
docker run --rm \
  -v /path/to/models:/models \
  -v /path/to/input:/input \
  -v /path/to/output:/output \
  autostr --batch
```

批次模式會掃描 `/input` 底下的影片或音訊檔，找出還沒有對應字幕的檔案再依序處理。
如果你要使用 `--batch`，就不要再另外加單一檔案路徑，例如 `/input/input.mp4`。

---

## 使用 Docker Compose

如果你想更簡單，可以直接用 Compose。

CPU 預設版：

```bash
docker compose up --build
```

GPU 版：

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Compose 會把主機上的資料夾掛到容器內固定位置：

- `./models` -> `/models`
- `./input` -> `/input`
- `./output` -> `/output`

如果你改了主機資料夾名稱，只要改左邊的路徑即可，容器內路徑不用改。

---

## 你可以調整哪些參數

這些是最常用、也最值得先知道的參數：

| 參數 | 用途 | 預設值 | 可用選項 |
|---|---|---|---|
| `--model` | Whisper 模型大小 | `medium` | `tiny`、`base`、`small`、`medium`、`large-v2`、`large-v3` |
| `--language` | 語言代碼 | `zh` | 任何 BCP-47 語言代碼，常用 `zh` |
| `--device` | 推論裝置 | `cpu` | `cpu`、`cuda` |
| `--compute-type` | 推論精度 | `int8` | `int8`、`float16`、`float32` |
| `--no-whisperx` | 停用 WhisperX 對齊 | 開啟 WhisperX | 無參數，加入後就會停用 |
| `--max-chars` | 每行最大字數 | `16` | 任意正整數 |
| `--start-delay` | 字幕起始延後毫秒數 | `0` | 任意整數（毫秒） |
| `--global-shift` | 全部字幕整體平移毫秒數 | `0` | 任意整數（毫秒） |
| `--min-duration` | 最短顯示時間 | `0.8` 秒 | 任意正數（秒） |
| `--max-duration` | 最長顯示時間 | `7.0` 秒 | 任意正數（秒） |
| `--highlights` | 輸出高光片段 | 關閉 | 無參數，加入後就會啟用 |
| `--batch` | 批次模式 | 關閉 | 無參數，加入後就會進入批次模式 |
| `--keep-audio` | 保留中間產生的 WAV 檔 | 關閉 | 無參數，加入後就會啟用 |
| `--model-dir` | Whisper 模型快取目錄 | 預設使用 `/models` | 任意路徑 |
| `--highlight-output-dir` | 高光輸出資料夾 | 使用字幕輸出資料夾 | 任意路徑 |
| `--highlight-count` | 最多輸出幾個高光片段 | `3` | 任意正整數 |
| `--highlight-min-duration` | 高光最短長度 | `15.0` 秒 | 任意正數（秒） |
| `--highlight-max-duration` | 高光最長長度 | `60.0` 秒 | 任意正數（秒） |
| `--highlight-padding` | 高光前後補充秒數 | `1.5` 秒 | 任意正數（秒） |
| `--highlight-encoder` | 高光輸出編碼器偏好 | `auto` | `auto`、`cpu`、`gpu` |
| `--verbose` | DEBUG 輸出 | 關閉 | 無參數，加入後就會啟用 |

如果你只想先做出可用字幕，通常先調 `--start-delay`、`--global-shift`、`--max-chars` 就夠了。

### 完整可用參數

如果你想知道目前可以傳哪些參數，可以直接看這份完整清單：

#### I/O 與模式

- `video`：輸入影片或音訊檔路徑
- `-o, --output`：輸出 SRT 檔路徑
- `--batch`：批次模式，掃描 `/input` 中缺少字幕的檔案並依序處理
- `--keep-audio`：保留中間產生的 WAV 檔
- `--model-dir`：Whisper 模型快取目錄

#### 轉錄與模型

- `--model`：`tiny`、`base`、`small`、`medium`、`large-v2`、`large-v3`
- `--language`：語言代碼，預設 `zh`
- `--device`：`cpu` 或 `cuda`
- `--compute-type`：`int8`、`float16`、`float32`
- `--no-whisperx`：停用 WhisperX 對齊

#### 高光輸出

- `--highlights`：輸出高光片段
- `--highlight-output-dir`：高光輸出資料夾
- `--highlight-count`：最多輸出幾個高光片段
- `--highlight-min-duration`：高光最短長度（秒）
- `--highlight-max-duration`：高光最長長度（秒）
- `--highlight-padding`：高光前後補充秒數
- `--highlight-encoder`：`auto`、`cpu`、`gpu`

#### 字幕重排與時間調整

- `--max-chars`：每行最大字數
- `--start-delay`：字幕起始延後毫秒數
- `--global-shift`：全部字幕整體平移毫秒數
- `--min-duration`：字幕最短顯示時間（秒）
- `--max-duration`：字幕最長顯示時間（秒）

#### 其他

- `-v, --verbose`：輸出 DEBUG 等級 log

如果你平常只想做一般字幕輸出，最常會碰到的通常是 `--model`、`--device`、`--compute-type`、`--start-delay`、`--global-shift`、`--max-chars`、`--highlights`。

---

## 中文影片常見建議

| 情況 | 建議 |
|---|---|
| 一般講座 | `--model medium --start-delay 100` |
| 語速較快 | `--model large-v2 --start-delay 150` |
| GPU 環境 | `--device cuda --compute-type float16 --model large-v3` |
| 想先快速出稿 | `--model small --no-whisperx` |
| 字幕太長 | `--max-chars 14` |

---

## 如果字幕不太準，先看這裡

### 字幕太早

先加一點延遲：

```bash
--start-delay 100
```

### 字幕整體都偏掉

整段一起往前或往後平移：

```bash
--global-shift 200
```

或：

```bash
--global-shift -200
```

### 字幕太長、不好讀

把每行字數調小：

```bash
--max-chars 14
```

### 字幕停留太短

把最短顯示時間調高：

```bash
--min-duration 1.2
```

---

## SRT 輸出格式

AutoStr 會輸出標準 SRT，格式是：

1. 編號
2. 時間範圍
3. 字幕文字
4. 空白行

如果你把字幕拿去其他平台，通常只要保持這個格式就能正常匯入。

---

## 本機執行

如果你不想用 Docker，也可以直接在本機跑：

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
python main.py input.mp4
```

如果你要使用 WhisperX，還需要另外安裝 WhisperX。

---

## 測試

```bash
python -m pytest tests/ -v
```

如果你只想先驗證 CLI，可以先跑：

```bash
python -m pytest tests/test_cli.py -v
```

---

## 專案結構

```text
AutoStr/
├── autostr/
│   ├── audio.py
│   ├── highlight.py
│   ├── transcribe.py
│   ├── align.py
│   ├── reflow.py
│   ├── srt_writer.py
│   └── pipeline.py
├── tests/
│   ├── test_cli.py
│   ├── test_highlight.py
│   ├── test_pipeline_batch.py
│   ├── test_pipeline_highlights.py
│   └── test_reflow_and_srt.py
├── main.py
├── model_loader.py
├── subtitle_utils.py
├── Dockerfile
├── docker-compose.yml
├── docker-compose.gpu.yml
├── requirements.txt
└── requirements-dev.txt
```

---

## 授權

MIT