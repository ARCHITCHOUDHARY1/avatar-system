"""
Download Wav2Lip model from correct URL
"""
import urllib.request
from pathlib import Path
import os

# Create model directory
model_dir = Path("models/wav2lip/checkpoints")
model_dir.mkdir(parents=True, exist_ok=True)

# Updated model URL (Google Drive mirror)
model_urls = [
    "https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/Eb3LEzbfuKlJiR600lQWRxgBIY27JZg80f7V9jtMfbNDaQ?download=1",
    "https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip_gan.pth"
]

model_path = model_dir / "wav2lip_gan.pth"

print("=" * 60)
print("Downloading Wav2Lip Model for CPU Avatar Generation")
print("=" * 60)
print(f"Destination: {model_path}")
print(f"Size: ~291 MB")
print("\nTrying multiple sources...")
print("=" * 60)

success = False
for i, url in enumerate(model_urls, 1):
    try:
        print(f"\n[{i}/{len(model_urls)}] Trying: {url[:60]}...")
        
        def progress_hook(count, block_size, total_size):
            if total_size > 0:
                percent = int(count * block_size * 100 / total_size)
                downloaded_mb = (count * block_size) / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                print(f"\rDownloading: {percent}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)", end='')
            else:
                downloaded_mb = (count * block_size) / (1024 * 1024)
                print(f"\rDownloaded: {downloaded_mb:.1f} MB", end='')
        
        urllib.request.urlretrieve(url, model_path, progress_hook)
        
        print("\n")
        file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        
        if file_size_mb > 200:  # Valid size
            print("=" * 60)
            print("✅ Download complete!")
            print(f"File size: {file_size_mb:.1f} MB")
            print("✅ Model ready for use!")
            success = True
            break
        else:
            print(f"⚠️ File too small ({file_size_mb:.1f} MB), trying next source...")
            model_path.unlink(missing_ok=True)
            
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        model_path.unlink(missing_ok=True)
        continue

if not success:
    print("\n" + "=" * 60)
    print("❌ All download attempts failed")
    print("\nManual Download Instructions:")
    print("1. Go to: https://github.com/Rudrabha/Wav2Lip")
    print("2. Check 'Releases' or 'README' for model download link")
    print("3. Download 'wav2lip_gan.pth' (~291 MB)")
    print(f"4. Place it in: {model_path}")
    print("=" * 60)
else:
    print("\nYou can now run: python test_generation_api.py")
