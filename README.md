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

### 要用不同高光策略

```bash
docker run --rm \
  -v /path/to/models:/models \
  -v /path/to/input:/input \
  -v /path/to/output:/output \
  autostr /input/input.mp4 --highlights --highlight-strategy tutorial --highlight-min-gap 6
```

目前內建三種策略：

- `balanced`：平均權重，適合一般內容
- `tutorial`：更重視資訊密度與語句完整度
- `entertainment`：更重視標點節奏與完整句尾

如果你已經知道自己想偏重哪種訊號，也可以在 strategy 之上做少量權重 override：

```bash
docker run --rm \
  -v /path/to/models:/models \
  -v /path/to/input:/input \
  -v /path/to/output:/output \
  autostr /input/input.mp4 --highlights --highlight-strategy tutorial --highlight-cue-weight 0.24 --highlight-pause-weight 0.11
```

- `--highlight-cue-weight`：提高或降低提示語句的重要性
- `--highlight-pause-weight`：提高或降低停頓邊界的重要性

如果你想先保留目前規則式評分，但再多一層可選的重排，也可以開啟 reranker hook：

```bash
docker run --rm \
  -v /path/to/models:/models \
  -v /path/to/input:/input \
  -v /path/to/output:/output \
  autostr /input/input.mp4 --highlights --highlight-reranker narrative
```

- `none`：不做額外 rerank，維持目前分數排序
- `narrative`：對有明確 cue phrase、完整句尾與停頓邊界的片段再給一點 bonus

目前高光評分除了長度、密度、標點與句尾完整度之外，也會納入幾個輕量語意訊號：

- `cue_phrase`：像是「重點」、「關鍵」、「注意」、「總結」這類提示語
- `pause_boundary`：片段前後是否有比較明顯的停頓邊界
- `cohesion`：同一段候選片段內部是否連貫，而不是靠過大的間隔硬拼起來

高光輸出資料夾除了影片片段外，還會多一份 `_highlights.json` manifest，裡面會保留：

- 這次使用的 strategy 與選擇參數
- 這次是否套用了 reranker hook
- 候選片段的 timeline span、平均分數、入選覆蓋率等 summary metrics
- 落選原因統計，例如 overlap 與 target count 造成的淘汰數量
- 候選分數分布，方便看這次分數是集中還是分散
- 分數最高但沒入選的 top alternates，方便你看「差一點被選上」的片段
- 最後被選中的高光片段
- 沒被選上的候選片段與淘汰原因，例如重疊或已達 target count

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

下面這份表格是目前版本完整可用的 CLI 參數對照表，內容直接對齊現在的 `main.py`。

| 參數 | 類型 / 要填什麼 | 預設值 | 可用值 | 說明 |
|---|---|---|---|---|
| `video` | 位置參數 | 無 | 單一影片或音訊路徑 | 不用 `--batch` 時必填。 |
| `-o`, `--output` | 路徑 | 自動決定 | 任意輸出 `.srt` 路徑 | 單檔模式的字幕輸出位置。 |
| `--batch` | 開關 | 關閉 | 不帶值 | 掃描 `/input`，只處理缺少對應字幕的檔案。 |
| `--keep-audio` | 開關 | 關閉 | 不帶值 | 保留中間產生的 `.wav` 音訊檔。 |
| `--model-dir` | 路徑 | `None` | 任意路徑 | Whisper 模型快取資料夾。Docker 常用 `/models`。 |
| `--highlights` | 開關 | 關閉 | 不帶值 | 啟用高光偵測與片段輸出。 |
| `--highlight-output-dir` | 路徑 | `None` | 任意路徑 | 高光片段輸出資料夾；未指定時使用字幕輸出資料夾。 |
| `--highlight-count` | 整數 | `3` | 任意正整數 | 最多輸出幾個高光片段。 |
| `--highlight-min-duration` | 浮點數 | `15.0` | 任意正數（秒） | 高光最短長度。 |
| `--highlight-max-duration` | 浮點數 | `60.0` | 任意正數（秒） | 高光最長長度。 |
| `--highlight-min-gap` | 浮點數 | `4.0` | 任意正數（秒） | 兩個高光之間至少要隔開多久。 |
| `--highlight-padding` | 浮點數 | `1.5` | 任意正數（秒） | 匯出片段時，前後各額外保留多少秒。 |
| `--highlight-strategy` | 列舉 | `balanced` | `balanced`、`tutorial`、`entertainment` | 高光評分策略。`balanced` 是平均型，適合一般內容；`tutorial` 偏重資訊密度、句尾完整與教學重點；`entertainment` 偏重標點節奏、轉折感與較有戲劇性的片段。 |
| `--highlight-reranker` | 列舉 | `none` | `none`、`narrative` | 候選高光的第二層重排 hook。 |
| `--highlight-cue-weight` | 浮點數 | `None` | 任意非負數 | 覆寫 `cue_phrase` 權重。未填時沿用 strategy 預設。 |
| `--highlight-pause-weight` | 浮點數 | `None` | 任意非負數 | 覆寫 `pause_boundary` 權重。未填時沿用 strategy 預設。 |
| `--highlight-encoder` | 列舉 | `auto` | `auto`、`cpu`、`gpu` | 高光輸出時偏好的編碼方式。 |
| `--model` | 列舉 | `medium` | `tiny`、`base`、`small`、`medium`、`large-v2`、`large-v3` | faster-whisper 轉字幕模型大小。 |
| `--language` | 字串 | `zh` | 任意 BCP-47 語言代碼 | 轉錄語言，常用 `zh`。 |
| `--device` | 列舉 | `cpu` | `cpu`、`cuda` | 推論裝置。 |
| `--compute-type` | 列舉 | `int8` | `int8`、`float16`、`float32` | 推論精度；CPU 常用 `int8`，GPU 常用 `float16`。 |
| `--no-whisperx` | 開關 | 關閉 | 不帶值 | 停用 WhisperX 對齊。預設會使用 WhisperX。 |
| `--max-chars` | 整數 | `16` | 任意正整數 | 每行字幕的最大字數。 |
| `--start-delay` | 整數 | `0` | 任意整數（毫秒） | 所有字幕起始時間統一延後。 |
| `--global-shift` | 整數 | `0` | 任意整數（毫秒） | 所有字幕整體時間平移，可正可負。 |
| `--min-duration` | 浮點數 | `0.8` | 任意正數（秒） | 字幕最短顯示時間。 |
| `--max-duration` | 浮點數 | `7.0` | 任意正數（秒） | 字幕最長顯示時間。 |
| `-v`, `--verbose` | 開關 | 關閉 | 不帶值 | 輸出 DEBUG 等級 log。 |

如果你只想先做出可用字幕，最常先調的是 `--model`、`--start-delay`、`--global-shift`、`--max-chars`。
如果你要一起輸出高光，最常先調的是 `--highlights`、`--highlight-strategy`、`--highlight-count`、`--highlight-min-gap`。

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