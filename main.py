import streamlit as st
import whisper
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
import tempfile
import os
from datetime import timedelta

# 日本語フォントの設定
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

st.set_page_config(
    page_title="Whisper 音声文字起こし",
    page_icon="🎤",
    layout="wide"
)

st.title("🎤 Whisper 音声文字起こしアプリ")
st.markdown("ローカルで動作するWhisperを使用して、音声ファイルを文字起こしします。")

# サイドバーでモデル選択
st.sidebar.header("設定")
model_size = st.sidebar.selectbox(
    "Whisperモデルを選択",
    ["tiny", "base", "small", "medium", "large"],
    index=2,
    help="大きいモデルほど精度が高いですが、処理時間が長くなります"
)

st.sidebar.markdown("""
### モデルサイズについて
- **tiny**: 最速、低精度
- **base**: 高速、基本的な精度
- **small**: バランス型（推奨）
- **medium**: 高精度、やや遅い
- **large**: 最高精度、最も遅い
""")

# ファイルアップロード
uploaded_file = st.file_uploader(
    "音声ファイルをアップロード",
    type=["mp3", "wav", "m4a", "ogg", "flac"],
    help="対応形式: MP3, WAV, M4A, OGG, FLAC"
)

if uploaded_file is not None:
    st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")
    
    if st.button("文字起こしを開始", type="primary"):
        # 一時ファイルに保存
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # モデルの読み込み
            with st.spinner(f"Whisperモデル ({model_size}) を読み込んでいます..."):
                model = whisper.load_model(model_size)
            
            # 文字起こし実行
            with st.spinner("音声を文字起こししています..."):
                result = model.transcribe(tmp_path, language="ja", verbose=False)
            
            st.success("✅ 文字起こしが完了しました！")
            
            # タブで結果を表示
            tab1, tab2, tab3 = st.tabs(["📝 文字起こし結果", "📊 可視化", "💾 ダウンロード"])
            
            with tab1:
                st.subheader("全文")
                st.write(result["text"])
                
                st.subheader("セグメント別")
                for i, segment in enumerate(result["segments"], 1):
                    start_time = str(timedelta(seconds=int(segment["start"])))
                    end_time = str(timedelta(seconds=int(segment["end"])))
                    st.markdown(f"**[{start_time} → {end_time}]**")
                    st.write(segment["text"])
                    st.divider()
            
            with tab2:
                st.subheader("授業記録の時間変化")
                
                # データ準備
                segments = result["segments"]
                times = [seg["start"] for seg in segments]
                durations = [seg["end"] - seg["start"] for seg in segments]
                word_counts = [len(seg["text"].split()) for seg in segments]
                char_counts = [len(seg["text"]) for seg in segments]
                
                # 3つのグラフを作成
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
                
                # グラフ1: セグメントのタイムライン
                ax1.barh(range(len(segments)), durations, left=times, height=0.8, color='steelblue', alpha=0.7)
                ax1.set_xlabel('Time (seconds)', fontsize=10)
                ax1.set_ylabel('Segment', fontsize=10)
                ax1.set_title('Segment Timeline Distribution', fontsize=12, fontweight='bold')
                ax1.grid(True, alpha=0.3)
                
                # グラフ2: 時間ごとの単語数
                ax2.plot(times, word_counts, marker='o', linestyle='-', linewidth=2, markersize=6, color='green')
                ax2.fill_between(times, word_counts, alpha=0.3, color='green')
                ax2.set_xlabel('Time (seconds)', fontsize=10)
                ax2.set_ylabel('Word Count', fontsize=10)
                ax2.set_title('Word Count Over Time', fontsize=12, fontweight='bold')
                ax2.grid(True, alpha=0.3)
                
                # グラフ3: 話速の変化（単語数/秒）
                speaking_rates = [wc / dur if dur > 0 else 0 for wc, dur in zip(word_counts, durations)]
                ax3.plot(times, speaking_rates, marker='s', linestyle='-', linewidth=2, markersize=6, color='red')
                ax3.fill_between(times, speaking_rates, alpha=0.3, color='red')
                ax3.set_xlabel('Time (seconds)', fontsize=10)
                ax3.set_ylabel('Speaking Rate (words/sec)', fontsize=10)
                ax3.set_title('Speaking Rate Over Time', fontsize=12, fontweight='bold')
                ax3.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # グラフをJPGとして保存
                jpg_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
                fig.savefig(jpg_path, format='jpg', dpi=300, bbox_inches='tight', quality=95)
                plt.close()
            
            with tab3:
                st.subheader("ファイルをダウンロード")
                
                # CSV作成
                csv_data = []
                for segment in segments:
                    csv_data.append({
                        "開始時刻": str(timedelta(seconds=int(segment["start"]))),
                        "終了時刻": str(timedelta(seconds=int(segment["end"]))),
                        "長さ(秒)": round(segment["end"] - segment["start"], 2),
                        "単語数": len(segment["text"].split()),
                        "文字数": len(segment["text"]),
                        "テキスト": segment["text"]
                    })
                
                df = pd.DataFrame(csv_data)
                csv = df.to_csv(index=False, encoding='utf-8-sig')
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        label="📄 CSV形式でダウンロード",
                        data=csv,
                        file_name=f"{Path(uploaded_file.name).stem}_transcription.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    with open(jpg_path, "rb") as f:
                        st.download_button(
                            label="📊 グラフ(JPG)をダウンロード",
                            data=f.read(),
                            file_name=f"{Path(uploaded_file.name).stem}_visualization.jpg",
                            mime="image/jpeg"
                        )
                
                # テキストファイルもダウンロード可能に
                st.download_button(
                    label="📝 全文テキストをダウンロード",
                    data=result["text"],
                    file_name=f"{Path(uploaded_file.name).stem}_transcription.txt",
                    mime="text/plain"
                )
        
        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
        
        finally:
            # 一時ファイルを削除
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            if 'jpg_path' in locals() and os.path.exists(jpg_path):
                os.unlink(jpg_path)

else:
    st.info("👆 音声ファイルをアップロードして開始してください")
    
    st.markdown("""
    ### 使い方
    1. サイドバーでWhisperモデルを選択
    2. 音声ファイルをアップロード
    3. 「文字起こしを開始」ボタンをクリック
    4. 結果を確認し、CSV・JPG・テキストファイルをダウンロード
    
    ### 対応形式
    - MP3, WAV, M4A, OGG, FLAC
    
    ### 特徴
    - 🔒 完全ローカル処理（ネット経由なし）
    - 📊 時間変化の可視化
    - 💾 CSV・JPG・テキスト形式でエクスポート
    - 🇯🇵 日本語対応
    """)
