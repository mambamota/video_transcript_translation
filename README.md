# Video Translator & Voice Generator

This application translates the audio of a video into a target language, generates synchronized audio using text-to-speech (TTS), and merges the translated audio with the original video. It ensures synchronization between the video and audio, dynamically adjusts timing, and provides robust error handling.

---

## Features

- **Video Translation**: Automatically transcribes the video, translates the text into the target language, and generates audio using neural TTS voices.
- **Dynamic Synchronization**: Ensures the generated audio matches the video timing, with adjustments for mismatches.
- **Multiple Voice Options**: Choose from a variety of neural voices for the translated audio.
- **Error Handling**: Validates input and output files, handles timing mismatches, and provides detailed error messages.
- **Debugging Tools**: Generates a detailed debug log for each processing session.
- **Streamlit UI**: User-friendly interface for uploading videos, configuring settings, and downloading the translated video.

---

## Requirements

- **Python**: 3.8 or higher
- **Dependencies**:
  - `streamlit`
  - `pydub`
  - `moviepy`
  - `deep-translator`
  - `edge-tts`
  - `whisper`
  - `nest-asyncio`
  - `ffmpeg` (must be installed and accessible in the system PATH)

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/video-translator.git
   cd video-translator