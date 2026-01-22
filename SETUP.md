# Koe Quick Setup Guide

This guide gets you up and running quickly. For comprehensive documentation, see [README.md](README.md).

---

## Desktop Setup (5 minutes)

### Prerequisites

- Windows 10/11 (64-bit)
- NVIDIA GPU with 6GB+ VRAM
- Python 3.10+

### Recommended: Setup Wizard

```powershell
# 1. Install dependencies (~3GB download)
cd C:\dev\koe
pip install -r requirements.txt

# 2. Run Koe - wizard launches automatically
python run.py
```

The **Setup Wizard** guides you through everything:
- System compatibility check
- API key configuration
- Model downloads (~3-4GB)
- User profile and voice enrollment
- Output folder setup

**Re-run wizard anytime:** `python run.py --setup`

### Alternative: Manual Setup

```powershell
# 1. Install dependencies
cd C:\dev\koe
pip install -r requirements.txt

# 2. Install GPU packages (for speaker diarization)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pyannote.audio

# 3. Configure environment
# Create .env file with your HuggingFace token:
echo HF_TOKEN=hf_your_token_here > .env

# 4. Configure your name in config.yaml
# profile:
#   user_name: YourName

# 5. Create setup marker (skips wizard)
echo. > .setup_complete

# 6. Start Koe
python run.py
```

### First Launch

- First startup takes 30-60 seconds (loading models)
- Look for the Koe tray icon
- Press `Ctrl+Shift+Space` to test

---

## Laptop Setup (2 minutes)

### Prerequisites

- Desktop already running Koe
- Tailscale installed on both machines

### Steps

```powershell
# 1. Copy koe folder to laptop

# 2. Install lightweight dependencies (~50MB)
cd C:\dev\koe
pip install -r requirements-remote.txt

# 3. Configure .env with your desktop's Tailscale IP
# WHISPER_SERVER_URL=http://100.x.x.x:9876

# 4. Start Koe Remote
python run.py
```

---

## Quick Reference

| Action | Desktop | Laptop |
|--------|---------|--------|
| Start Koe | `Start Koe Desktop` shortcut | `Start Koe Remote` shortcut |
| Start Scribe | `Start Scribe Desktop` shortcut | `Start Scribe Remote` shortcut |
| Hotkey | `Ctrl+Shift+Space` | `Ctrl+Shift+Space` |
| Exit | Right-click tray → Exit | Right-click tray → Exit |

---

## Essential Configuration

### Environment Variables (.env)

```ini
# Required for desktop (speaker diarization)
HF_TOKEN=hf_your_huggingface_token

# Required for laptop (remote connection)
WHISPER_SERVER_URL=http://100.x.x.x:9876

# Optional (AI meeting summaries)
ANTHROPIC_API_KEY=sk-ant-your_key
```

### User Settings (config.yaml)

```yaml
profile:
  user_name: YourName              # Labels your mic audio

recording_options:
  activation_key: ctrl+shift+space # Global hotkey

misc:
  noise_on_completion: true        # Beep when done
```

---

## Common Issues

| Problem | Solution |
|---------|----------|
| First startup slow | Normal - models loading into GPU (~30-60s) |
| Hotkey not working | Check tray icon exists, restart Koe |
| "Server Not Available" | Check Tailscale connected, desktop running |
| Model download fails | Verify HF_TOKEN, accept model licenses on HuggingFace |

---

## Where to Get API Keys

| Service | URL | Purpose |
|---------|-----|---------|
| HuggingFace | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | Speaker diarization models |
| Anthropic | [console.anthropic.com](https://console.anthropic.com) | AI meeting summaries |

---

## Next Steps

1. **Try Scribe**: Right-click tray → Start Scribe
2. **Enroll speakers after meetings**: Stop recording → click "Enroll Speakers" → name unknown speakers
3. **Your voice enrolled automatically**: Your voice embedding is captured during meetings
4. **Read full docs**: See [README.md](README.md) for complete documentation

---

## File Locations

| What | Where |
|------|-------|
| Configuration | `config.yaml` |
| Environment | `.env` |
| Hotkey snippets | `Snippets/` |
| Meeting transcripts | `Meetings/Transcripts/` |
| AI summaries | `Meetings/Summaries/` |
| Error logs | `logs/koe_errors.log` |
