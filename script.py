#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import subprocess
import tempfile
from datetime import timedelta
from faster_whisper import WhisperModel

def ffprobe_duration(path):
    try:
        # Videonun süresini saniye cinsinden al (opsiyonel, sadece bilgi amaçlı)
        out = subprocess.check_output([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path
        ], stderr=subprocess.STDOUT)
        return float(out.strip())
    except Exception:
        return None

def extract_audio(input_media, tmp_dir):
    """
    Whisper çoğu formatı açabiliyor; ancak problemli kapsayıcılar için
    ffmpeg ile temiz bir 16k mono wav çıkarıyoruz.
    """
    audio_path = os.path.join(tmp_dir, "audio_16k_mono.wav")
    cmd = [
        "ffmpeg", "-y",
        "-i", input_media,
        "-ac", "1",         # mono
        "-ar", "16000",     # 16 kHz
        "-vn",              # video yok
        "-hide_banner", "-loglevel", "error",
        audio_path
    ]
    subprocess.check_call(cmd)
    return audio_path

def format_timestamp(seconds):
    if seconds is None:
        seconds = 0
    td = timedelta(seconds=max(0, seconds))
    # SRT formatı: HH:MM:SS,mmm
    total_ms = int(td.total_seconds() * 1000)
    hrs = total_ms // 3600000
    rem = total_ms % 3600000
    mins = rem // 60000
    rem = rem % 60000
    secs = rem // 1000
    ms = rem % 1000
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"

def write_srt(segments, out_path):
    def is_meaningful(text):
        if not text or len(text.strip()) < 3:
            return False
        if text.strip().lower() in ["aaaa.", "hmm.", "mmm.", "...", "..."]:
            return False
        return True
    with open(out_path, "w", encoding="utf-8") as f:
        prev_text = None
        repeat_count = 0
        for i, seg in enumerate(segments, start=1):
            start = format_timestamp(seg.start)
            end = format_timestamp(seg.end)
            text = (seg.text or "").strip()
            if not is_meaningful(text):
                continue
            # Ardışık tekrar kontrolü
            if prev_text and text == prev_text:
                repeat_count += 1
                if repeat_count == 3:
                    print(f"UYARI: '{text}' metni 3 kez ardışık tekrar ediyor. SRT'ye yazılmayacak.")
                if repeat_count >= 3:
                    continue  # 3 ve sonrası tekrarları yazma
            else:
                repeat_count = 0
            prev_text = text
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

def main():
    parser = argparse.ArgumentParser(
        description="Videodan ses yakalayıp SRT altyazı çıkaran hızlı Whisper script'i."
    )
    parser.add_argument("input", help="Girdi video/ses dosyası (mp4, mov, mkv, mp3, wav, ...)")
    parser.add_argument("-o", "--output", help="Çıkış SRT yolu (varsayılan: input.srt)")
    parser.add_argument("-m", "--model", default="small",
                        help="Whisper model boyutu: tiny, base, small, medium, large-v3 vb. (varsayılan: small)")
    parser.add_argument("-l", "--language", default=None,
                        help="Dil kodu (ör. tr, en). Boş bırakılırsa otomatik tespit edilir.")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"],
                        help="Zorla cihaz seç (varsayılan: otomatik).")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam search boyutu (varsayılan: 5)")
    parser.add_argument("--vad", action="store_true", help="VAD (ses-aktivitesi) filtresi kullan (gürültüde faydalı).")
    parser.add_argument("--word-ts", action="store_true",
                        help="Kelime zaman damgaları çıkar (daha yavaş, SRT segmentleri yine cümle bazlı yazılır).")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Hata: {args.input} bulunamadı.", file=sys.stderr)
        sys.exit(1)

    out_srt = args.output or (os.path.splitext(args.input)[0] + ".srt")

    print("[1/5] Model yükleniyor...")
    # Model yükleme
    compute_type = "float16"  # GPU’da hızlı; CPU’da otomatik düşer
    if args.device == "cpu":
        compute_type = "int8"  # CPU için hız/performans dengesi

    model = WhisperModel(
        args.model,
        device=args.device or ("cuda" if shutil_which("nvidia-smi") else "cpu"),
        compute_type=compute_type
    )
    print("[2/5] Geçici klasör oluşturuluyor ve ses çıkarılıyor...")
    # Geçici klasör
    with tempfile.TemporaryDirectory() as tmp:
        # Ses çıkar (gerekirse)
        try:
            print(f"  ffmpeg ile {args.input} dosyasından ses çıkarılıyor...")
            audio_path = extract_audio(args.input, tmp)
            print(f"  Ses dosyası hazır: {audio_path}")
        except subprocess.CalledProcessError as e:
            print("ffmpeg ile ses çıkarılırken hata oluştu. ffmpeg kurulu mu?", file=sys.stderr)
            sys.exit(2)

        # İsteğe bağlı: süre bilgisi
        duration = ffprobe_duration(args.input)
        if duration:
            print(f"  Video/Ses süresi: {duration:.1f} saniye")

        print("[3/5] Transcribe işlemi başlıyor...")
        # Transcribe
        segments, info = model.transcribe(
            audio_path,
            language=args.language,
            task="transcribe",
            vad_filter=args.vad,
            vad_parameters=dict(min_silence_duration_ms=500) if args.vad else None,
            beam_size=args.beam_size,
            word_timestamps=args.word_ts,
            condition_on_previous_text=True,
            temperature=0.0,
        )
        print("  Transcribe tamamlandı. Segmentler toplanıyor...")
        # Segmanları topla ve ilerleme göster
        collected = []
        segments = list(segments)  # Eğer generator ise listeye çevir
        total_segments = len(segments)
        # İlk 3 ve son 3 segmenti ekrana bas
        print("Örnek segmentler (ilk 3):")
        for seg in segments[:3]:
            print(f"  [{format_timestamp(seg.start)} - {format_timestamp(seg.end)}] {seg.text}")
        print("Örnek segmentler (son 3):")
        for seg in segments[-3:]:
            print(f"  [{format_timestamp(seg.start)} - {format_timestamp(seg.end)}] {seg.text}")
        for idx, seg in enumerate(segments):
            collected.append(seg)
            if total_segments:
                percent = int((idx + 1) / total_segments * 100)
                print(f"    [{percent}%] Segment {idx+1}/{total_segments}", end='\r')
        print()  # Satır sonu

        if not collected:
            print("Hiç konuşma tespit edilemedi.", file=sys.stderr)
            sys.exit(3)

        print("[4/5] SRT dosyası yazılıyor...")
        write_srt(collected, out_srt)

    print(f"[5/5] ✔ SRT hazır: {out_srt}")

def shutil_which(cmd):
    from shutil import which
    return which(cmd) is not None

if __name__ == "__main__":
    main()
