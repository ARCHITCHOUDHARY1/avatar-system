# Quick Start Guide - Google Colab

## 🚀 Fastest Way to Run (5 Minutes)

### Step 1: Open Colab Notebook
1. Go to: https://colab.research.google.com
2. Upload `notebooks/Colab_Avatar_System.ipynb`
3. Or click: File → Upload notebook → Select the file

### Step 2: Enable GPU
```
Runtime → Change runtime type → GPU → T4 GPU → Save
```

### Step 3: Run All Cells
```
Runtime → Run all (Ctrl+F9)
```

### Step 4: When Prompted
- Upload your audio file (.wav, .mp3)
- Upload your face image (.jpg, .png)

### Step 5: Get Result
- Video will appear in the output
- Click download button
- Done! 🎉

---

## 📱 Alternative: Gradio Interface

After Step 3, look for the public URL:
```
Running on public URL: https://xxxxxx.gradio.live
```

Click the link and use the web interface!

---

## ⚡ Pro Tips

1. **Save Models to Drive** (First Session Only)
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   # Models auto-saved to Drive for reuse
   ```

2. **Process Multiple Files**
   - Upload multiple audio files
   - Use same face image
   - Batch generate all at once

3. **Quality Settings**
   - Low: 30 seconds/video (good for testing)
   - Medium: 1 minute/video (recommended)
   - High: 3 minutes/video (best quality)

4. **Keep Session Alive**
   - Run code in browser console:
   ```javascript
   setInterval(() => document.querySelector("colab-toolbar-button#connect").click(), 60000)
   ```

---

## 🐛 Common Issues

### "No GPU Available"
→ Runtime → Change runtime type → Enable GPU

### "Out of Memory"
→ Restart runtime, use lower resolution (256x256)

### "Session Disconnected"
→ Save models to Drive first, reconnect and reload

### "Model Download Failed"
→ Re-run the download cell, check internet connection

---

## 💰 Free vs Paid

**Free Tier (Colab)**:
- T4 GPU (16GB)
- 12 hours max
- Good for 10-20 videos/day

**Colab Pro ($10/month)**:
- Better GPUs (V100, A100)
- 24 hours sessions
- Priority access

**Recommendation**: Start with free tier!

---

## 📊 Expected Performance (Free T4 GPU)

| Input Duration | Resolution | Quality | Processing Time |
|----------------|-----------|---------|-----------------|
| 10 seconds     | 512x512   | Medium  | ~1 minute       |
| 30 seconds     | 512x512   | Medium  | ~2-3 minutes    |
| 60 seconds     | 512x512   | Medium  | ~4-5 minutes    |
| 10 seconds     | 512x512   | High    | ~2 minutes      |

---

## 🔗 Need Help?

- Full Guide: See `docs/Colab_Deployment_Guide.md`
- Troubleshooting: Check logs in Colab output
- Issues: Check GitHub Issues page

---

**That's it! No local setup needed. Just upload and run!** ✨
