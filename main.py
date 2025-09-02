import collections
import json
import os
import wave
import time

import webrtcvad
import pyaudio
import openai
from pynvim import attach
# ——— VAD constants ———
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
FRAME_DURATION = 30    # ms
FRAME_SIZE = int(RATE * FRAME_DURATION / 1000)
PADDING_MS = 2000
NUM_PADDING_FRAMES = int(PADDING_MS / FRAME_DURATION)

# ——— Initialize PyAudio & VAD ———
pa = pyaudio.PyAudio()
stream = pa.open(format=FORMAT,
                 channels=CHANNELS,
                 rate=RATE,
                 input=True,
                 frames_per_buffer=FRAME_SIZE)
vad = webrtcvad.Vad(2)

def record_until_silence():
    ring = collections.deque(maxlen=NUM_PADDING_FRAMES)
    voiced_frames = []
    triggered = False

    while True:
        frame = stream.read(FRAME_SIZE, exception_on_overflow=False)
        is_speech = vad.is_speech(frame, RATE)

        if not triggered:
            ring.append((frame, is_speech))
            if sum(1 for _, speech in ring if speech) > 0.9 * ring.maxlen:
                triggered = True
                for f, _ in ring:
                    voiced_frames.append(f)
                ring.clear()
        else:
            voiced_frames.append(frame)
            ring.append((frame, is_speech))
            if sum(1 for _, speech in ring if not speech) > 0.9 * ring.maxlen:
                # include trailing padding
                for f, _ in ring:
                    voiced_frames.append(f)
                break

    buf = b"".join(voiced_frames)
    wf = wave.open("command.wav", "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pa.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(buf)
    wf.close()

# ——— History persistence ———
HISTORY_FILE = "history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    # first run: your original system prompt
    return [
        {
            "role": "system",
            "content": (
                "Return exactly one valid NVim command. IF YOU ARE UNSURE WHAT TO DO, "
                "DO AN ECHO TO ASK THE USER FOR CLARIFICATION. You are allowed to use "
                "the `:echo` command to ask the user for clarification on which command "
                "to execute. Basically you can communicate with him this way. if user says "
                "'do you hear me?' you can reply with how may i help you today via echo. Feel free to ask any further questions via echo."
            )
        }
    ]

def save_history(messages):
    with open(HISTORY_FILE, "w") as f:
        json.dump(messages, f, indent=2)

# ——— Main setup ———
openai.api_key = os.getenv("OPENAI_API_KEY", "sk-notoken")
nvim = attach('socket', path='/tmp/nvim.sock')
messages = load_history()
while True:
    print("Waiting for you to start speaking…")
    record_until_silence()
    print("Processing…")

    # Whisper transcription
    with open("command.wav", "rb") as f:
        transcript = openai.Audio.transcribe("whisper-1", f)
    cmd_text = transcript["text"].strip()
    print(f"Transcribed: {cmd_text}")
    # ── fetch code context from Neovim ──
    buf = nvim.current.buffer
    full_lines = nvim.current.buffer[:]
    (line, col) = nvim.current.window.cursor
    snippet = "\n".join(f"{i+1}: {l}" for i, l in enumerate(full_lines))
    # Add user turn
    user_prompt = (
        "Here’s the code around your cursor:\n"
        "```vim\n" + snippet + "\n```\n"
        f"Based on that, generate a single NVim Ex-command to “{cmd_text}” "
        "(no backticks, no leading colon)."
    )
    messages.append({"role": "user", "content": user_prompt})

    # Call GPT with full history
    resp = openai.ChatCompletion.create(
        model="o3",
        messages=messages,
        temperature=1
    )
    assistant_reply = resp.choices[0].message.content.strip()
    print(f"GPT-4o response: {assistant_reply}")

    # Append assistant turn and persist
    messages.append({"role": "assistant", "content": assistant_reply})
    save_history(messages)

    # Execute in Neovim
    print(f"→ Executing: {assistant_reply}")
    errored = True
    while errored:
        try:
            # Call GPT with full history
            resp = openai.ChatCompletion.create(
                model="o3",
                messages=messages,
                temperature=1
            )
            assistant_reply = resp.choices[0].message.content.strip()
            nvim.command(assistant_reply.replace("`", "").replace("vim", "").strip())
            errored = False
        except Exception as e:
            errored = True
            print(f"Error executing command: {e}")
            messages.append({"role": "assistant", "content": f"Error: {e}. Please fix the command and try again."})
            save_history(messages)
    time.sleep(1) 

