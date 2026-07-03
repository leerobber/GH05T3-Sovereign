"""Real GhostEye OCR using Windows built-in OCR (no external dependencies).

Falls back to pytesseract if WinRT is unavailable.
"""
from __future__ import annotations
import asyncio
import base64
import io
import logging
import os
import subprocess
from pathlib import Path

LOG = logging.getLogger("ghost.ocr")


async def ocr_screenshot() -> str:
    """Take a screenshot and OCR it using Windows OCR."""
    try:
        import mss
        with mss.mss() as sct:
            img = sct.grab(sct.monitors[1])
            from PIL import Image
            pil = Image.frombytes("RGB", img.size, img.rgb)
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            return await _ocr_bytes(buf.getvalue())
    except Exception as e:
        LOG.warning("screenshot OCR failed: %s", e)
        return ""


async def _ocr_bytes(image_bytes: bytes) -> str:
    """Run Windows OCR on raw image bytes via PowerShell WinRT."""
    script = r"""
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Storage.StorageFile, Windows.Storage, ContentType=WindowsRuntime]
$null = [Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType=WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics, ContentType=WindowsRuntime]

function Await($WinRtTask, $ResultType) {
    $asTask = [System.WindowsRuntimeSystemExtensions].GetMethod('AsTask', [System.Type[]] @([System.Type]::GetType("Windows.Foundation.IAsyncOperation``1").MakeGenericType($ResultType)))
    $asTask.Invoke($null, @($WinRtTask)).GetAwaiter().GetResult()
}

$imgBytes = [Convert]::FromBase64String($env:IMG_B64)
$ms = [System.IO.MemoryStream]::new($imgBytes)
$ras = [System.IO.WindowsRuntimeStreamExtensions]::AsRandomAccessStream($ms)
$decoder = Await([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($ras)) ([Windows.Graphics.Imaging.BitmapDecoder])
$bitmap = Await($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
$result = Await($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
$result.Text
"""
    b64 = base64.b64encode(image_bytes).decode()
    env = {**os.environ, "IMG_B64": b64}
    try:
        proc = await asyncio.create_subprocess_exec(
            "powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
        text = stdout.decode("utf-8", errors="replace").strip()
        return text
    except Exception as e:
        LOG.warning("WinRT OCR failed: %s", e)
        return await _ocr_tesseract(image_bytes)


async def _ocr_tesseract(image_bytes: bytes) -> str:
    """Fallback: tesseract CLI if installed."""
    tmp = Path(os.environ.get("TEMP", "C:/Temp")) / "gh05t3_ocr_input.png"
    tmp.write_bytes(image_bytes)
    try:
        proc = await asyncio.create_subprocess_exec(
            "tesseract", str(tmp), "stdout", "--psm", "6",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
        return stdout.decode("utf-8", errors="replace").strip()
    except Exception:
        return ""
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


async def ocr_base64_image(b64: str) -> str:
    """OCR an image provided as base64 string."""
    return await _ocr_bytes(base64.b64decode(b64))
