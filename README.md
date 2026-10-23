# LyricSync 🎵 (Lrc & Srt AI 自動對齊與生成系統)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.14-blue.svg)](https://www.python.org/)
[![CUDA 13.0](https://img.shields.io/badge/CUDA-13.0-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Hardware](https://img.shields.io/badge/Hardware-RTX%2030%2F40%2F50%20Series-brightgreen.svg)]()

> **專為 Windows 打造的次世代 AI 歌詞同步與字幕生成系統。**  
> 整合 **Demucs 人聲抽離**、**Whisper Turbo 語意錨定** 與 **Meta MMS CTC 微秒發音對齊**，支援內嵌標籤自動提取、無歌詞純聽寫降級、以及 1:1 逆向的高精度 LRC 動態歌詞生成。

---

## 🌟 核心特色 (Key Features)

### 1. 🎵 智慧三階歌詞來源檢測（永不斷線）
- **第一優先（外部檔案）**：檢測是否存在同名 `.txt` 歌詞檔。
- **第二優先（內嵌標籤提取）**：若未附 `.txt`，自動檢索音訊中由 **Suno AI、Udio 或標準 ID3** 寫入之內嵌歌詞標籤。自動過濾清理 `[Verse]`、`[Chorus]`、`*guitar solo*` 等結構標籤，並**自動備份為同名 `.txt` 檔案**。
- **第三優先（純聽寫降級模式 Fallback ASR）**：若音訊完全無任何歌詞與標籤，系統不再中斷報錯，而是自動啟動「GPU 人聲抽離 + Whisper Turbo」純聽寫模式，依據停頓與標點自動斷句，產出 TXT、SRT 與 LRC。

### 2. 🎙️ GPU 人聲分離（Demucs v4）
- 對齊前透過 **Demucs (htdemucs)** 將背景伴奏與人聲乾淨分離。
- 消除和聲、重低音與樂器干擾，使語音辨識與時間軸對齊達到母帶級精準度。

### 3. ⚡ Whisper Turbo + Meta MMS CTC 雙引擎毫秒級對齊
- **Whisper `large-v3-turbo`**：快速全曲轉錄並進行動態規劃（DP）歌詞錨定，即使歌手漏唱、重複或略過某句，亦不會拖累全曲時間軸。
- **Meta MMS CTC Forced Aligner**：支援 1,130 種語言（涵蓋繁簡中文、日語、韓語、英語等），精準校正每個發音字元的開始與結束時間，並自動延伸尾音呼吸區間。

### 4. ⏱️ 1:1 逆向 `ailrcgenerator.com` 動態歌詞生成
- 1:1 逆向還原 `ailrcgenerator.com/tw/srt-to-lrc` 核心時間軸轉換演算法。
- 產出標準 `[mm:ss.xx]` 百分之一秒（Centisecond）精度動態 LRC 歌詞檔。
- **支援單獨拖曳轉換**：可直接將既有 `.srt` 拖入 `DROP_HERE.bat`，瞬間轉出 `.lrc`。

### 5. 📁 成果全套自動歸檔 (`songs/`)
- 每首歌曲處理完成後，音訊檔、純歌詞檔（`.txt`）、字幕檔（`.srt`）與動態歌詞檔（`.lrc`）自動同步歸檔至專屬 `songs/` 目錄，整齊不混亂。

### 6. 🚀 Windows 2026 / RTX 50 (Blackwell) 極致硬體優化
- **Direct-to-CUDA FP16 直載架構**：徹底避開官方 Whisper 預設在 CPU 主機記憶體建立張量導致的 `DefaultCPUAllocator: not enough memory` 崩潰問題，顯存佔用由 3.2 GB 腰斬至 **1.5 GB**。
- **LayerNorm FP32 數值相容**：精準只將 LayerNorm 轉換為 FP32，杜絕半精度計算崩潰。
- **模型生命週期解耦**：Demucs、Whisper、MMS 三大模型依序運作、用後立即釋放顯存，完全杜絕記憶體堆疊與 OOM。

---

## 🛠️ 系統需求 (Requirements)

- **作業系統**：Windows 10 / Windows 11 (64-bit)
- **Python 版本**：Python 3.11+ / Python 3.14（推薦搭配極速 `uv` 套件管理器）
- **GPU 支援**：NVIDIA GPU（RTX 30 / 40 / 50 系列，支援 Blackwell 架構及 CUDA 13.0）；無 GPU 時可降級使用 CPU。
- **系統工具**：需具備 `ffmpeg`（音訊轉碼支援）。

---

## 📦 安裝說明 (Installation)

### 快速一鍵安裝（推薦）：
1. 確保電腦已安裝 [uv](https://github.com/astral-sh/uv) 與 Python 3.11+。
2. 雙擊執行目錄下的 **`INSTALL.bat`**。
3. 腳本將自動建立虛擬環境，並極速安裝 PyTorch (cu130) 及所有相依套件。
4. 當視窗顯示 `LyricSync 安裝與配置成功！` 即代表就緒。

---

## 🚀 使用方法 (Usage)

### 方式一：最簡便用法（拖曳即用）
直接將以下檔案**拖曳到 `DROP_HERE.bat` 圖示上**：
- **歌曲檔**（`.mp3` / `.wav` / `.m4a`）
- **包含同名歌曲與歌詞的資料夾**
- **現成字幕檔**（`.srt`，將直接轉為 `.lrc`）

系統將全自動執行人聲抽離、歌詞解析、對齊精煉並產出字幕。

### 方式二：命令列執行 (CLI)
在虛擬環境中執行：
```powershell
# 標準強制對齊（自動檢測語言）
python lyricsync.py "path/to/song.mp3"

# 指定語言（如韓文 ko、日文 ja、中文 zh、英文 en）
python lyricsync.py "path/to/song.mp3" --lang ko

# 批量對齊整個資料夾
python lyricsync.py "D:/MyMusicFolder"

# SRT 直接轉 LRC
python lyricsync.py "path/to/subtitle.srt"
```

---

## 📂 專案結構 (Project Structure)

```text
lyricsync/
├── DROP_HERE.bat             # 核心入口：拖曳音訊/資料夾/SRT 立即執行
├── INSTALL.bat               # 一鍵安裝批次檔
├── install_uv.ps1            # uv 極速環境建置腳本
├── launcher.py               # 穩定啟動器與日誌收集器
├── lyricsync.py              # 核心主程式：人聲分離、標籤檢索、對齊與純聽寫
├── mms_refine.py             # Meta MMS CTC 1130 發音微秒精煉演算法
├── check_environment.py      # 硬體與 CUDA 運算能力驗證工具
├── requirements.txt          # Python 套件清單
├── songs/                    # 完成品與範例庫（音訊/TXT/SRT/LRC）
│   ├── 아직 네 온기가 남아.lrc
│   ├── 아직 네 온기가 남아.srt
│   ├── 아직 네 온기가 남아.txt
│   └── 每個停頓的理由.lrc
├── 使用說明.html             # 圖文詳細使用手冊
├── 先看這裡.txt              # 簡易文字快速指引
├── README.md                 # 專案說明文件
└── .gitignore                # Git 忽略設定（排除模型與大容量音訊）
```

---

## 📝 更新紀錄 (Changelog)

### [v1.2.0] - 2026-10-23
- **架構優化 (Direct-to-CUDA FP16)**：
  - 重構 Whisper 載入機制，以 `with torch.device("cuda"):` 與 `torch.set_default_dtype(torch.float16)` 直載 GPU。
  - 顯存佔用由 3.2 GB 腰斬至 **1.5 GB**，徹底解決系統虛擬記憶體飽和導致的 `DefaultCPUAllocator: not enough memory` 崩潰問題。
- **精度修復 (LayerNorm FP32)**：
  - 自動偵測並僅將 LayerNorm 轉換為 FP32，杜絕半精度計算的型態不相容問題。
- **顯存生命週期徹底解耦**：
  - Demucs、Whisper、MMS 三大模型依序運作、用後立即釋放顯存，確保顯存不累積。
- **Windows WDDM 最佳化**：
  - 移除易引起 Windows 核心虛擬顯存映射拒絕之 `expandable_segments`，回歸穩定物理顯存配置。

### [v1.1.0] - 2026-10-23
- **無歌詞純聽寫模式 (Fallback ASR)**：
  - 當音訊無同名 `.txt` 且無內嵌標籤時，自動降級為純聽寫模式，GPU 人聲分離後由 Whisper 轉錄，自動依呼吸停頓斷句並備份 `.txt`。
- **音訊內嵌歌詞檢索**：
  - 新增 Suno AI / Udio / ID3 標籤歌詞檢索與編碼修復（Mojibake Fix）。
  - 自動過濾 `[Verse]`、`[Chorus]`、`*guitar solo*` 等音樂結構標籤。
- **1:1 逆向 ailrcgenerator**：
  - 實作百分之一秒動態 LRC 歌詞生成演算法。
  - 支援直接拖曳 `.srt` 快速轉為 `.lrc`。
- **成果自動歸檔**：
  - 音訊、TXT、SRT、LRC 一鍵自動歸檔存入 `songs/` 目錄。

### [v1.0.0] - 2026-10-16
- 初始版本發布。
- 支援 Demucs 人聲分離、Whisper 歌詞對齊與 MMS 邊界微調。

---

## 📄 授權條款 (License)

本專案採用 [MIT License](LICENSE) 授權開放。
歡迎開源社群自由使用、修改與二次開發！
