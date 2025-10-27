# Whisper 音声文字起こしツール

OpenAI Whisperを使用して、ローカルで音声ファイルを文字起こしするPythonスクリプトです。インターネット接続は不要です（初回のモデルダウンロード後）。

## セットアップ

### 必要なライブラリのインストール

\`\`\`bash
pip install openai-whisper matplotlib numpy
\`\`\`

FFmpegも必要です：

**macOS:**
\`\`\`bash
brew install ffmpeg
\`\`\`

**Ubuntu/Debian:**
\`\`\`bash
sudo apt update && sudo apt install ffmpeg
\`\`\`

**Windows:**
[FFmpeg公式サイト](https://ffmpeg.org/download.html)からダウンロードしてインストール

## 使い方

### 基本的な文字起こし

\`\`\`bash
python scripts/transcribe_audio.py <音声ファイルパス>
\`\`\`

### モデルサイズを指定

\`\`\`bash
python scripts/transcribe_audio.py audio.mp3 small
\`\`\`

### 言語を指定（日本語）

\`\`\`bash
python scripts/transcribe_audio.py audio.mp3 base ja
\`\`\`

### タイムスタンプ付き文字起こし

\`\`\`bash
python scripts/transcribe_with_timestamps.py audio.mp3
\`\`\`

### タイムスタンプ付き文字起こし + 可視化 + CSV/JPG出力（授業記録の時間変化を図示）

\`\`\`bash
python scripts/transcribe_with_visualization.py audio.mp3
\`\`\`

このスクリプトは以下を生成します：
- 文字起こしテキストファイル（.txt）
- **CSV形式の文字起こしデータ（.csv）** - Excel等で開けます
  - 開始時刻、終了時刻、長さ、単語数、文字数、テキストを含む
- **時系列分析グラフ（.jpg）** - 高品質なJPG画像
  - セグメントのタイムライン分布
  - 時間ごとの単語数の変化
  - 話速（単語数/秒）の変化

例：
\`\`\`bash
# 日本語の授業音声を可視化
python scripts/transcribe_with_visualization.py lecture.mp3 small ja

# より高精度なモデルで処理
python scripts/transcribe_with_visualization.py lecture.wav medium ja
\`\`\`

## モデルサイズ

| サイズ | パラメータ数 | 必要メモリ | 速度 | 精度 |
|--------|-------------|-----------|------|------|
| tiny   | 39M         | ~1GB      | 最速 | 低   |
| base   | 74M         | ~1GB      | 速い | 中   |
| small  | 244M        | ~2GB      | 普通 | 高   |
| medium | 769M        | ~5GB      | 遅い | 高   |
| large  | 1550M       | ~10GB     | 最遅 | 最高 |

## 対応音声形式

- MP3
- WAV
- M4A
- FLAC
- OGG
- その他FFmpegが対応する形式

## 出力

### 基本版・タイムスタンプ版
- コンソールに文字起こし結果を表示
- `<元のファイル名>_transcription.txt` に結果を保存

### 可視化版（transcribe_with_visualization.py）
- `<元のファイル名>_transcription.txt` - 文字起こしテキスト
- `<元のファイル名>_transcription.csv` - **CSV形式のデータ（Excelで開けます）**
- `<元のファイル名>_visualization.jpg` - **時系列分析グラフ（JPG画像）**
- 統計情報（総時間、平均話速など）をコンソールに表示

## 注意事項

- 初回実行時は選択したモデルがダウンロードされます（インターネット接続が必要）
- ダウンロード後はローカルキャッシュが使用されるため、オフラインで動作します
- 長い音声ファイルの場合、処理に時間がかかります
- GPUがある場合は自動的に使用され、処理が高速化されます
