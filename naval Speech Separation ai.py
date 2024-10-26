import torch
import sounddevice as sd
import numpy as np
from asteroid.models import ConvTasNet
import noisereduce as nr
from scipy.io.wavfile import write, read
import tkinter as tk
from tkinter import messagebox, filedialog

# Load pre-trained Conv-TasNet model
model = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_16k")
model.eval()

# Real-time audio settings
sample_rate = 16000  # Set sample rate for model compatibility
duration = 5  # Recording duration per segment, in seconds

def record_audio(duration, sample_rate):
    """Records audio for a specified duration using the microphone."""
    print("Recording...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait for recording to complete
    print("Recording complete.")
    return torch.tensor(audio_data.T)  # Transpose to match model input shape

def separate_speech(model, waveform):
    """Separates speakers in the waveform using Conv-TasNet."""
    with torch.no_grad():
        separated_sources = model.separate(waveform)
    return separated_sources

def reduce_noise(waveform, sample_rate=16000):
    """Applies noise reduction on the audio waveform."""
    noise_reduced_waveform = nr.reduce_noise(y=waveform.numpy().flatten(), sr=sample_rate)
    return torch.tensor(noise_reduced_waveform).reshape(1, -1)

def process_live_audio():
    """Process live audio for speaker separation and noise reduction."""
    mixture_waveform = record_audio(duration, sample_rate)
    
    # Separate speakers
    separated_sources = separate_speech(model, mixture_waveform)
    
    # Apply noise reduction to separated sources
    for i, source in enumerate(separated_sources):
        clean_source = reduce_noise(source, sample_rate)
        
        # Save each noise-reduced speaker's audio
        output_path = f"live_noise_reduced_speaker_{i+1}.wav"
        write(output_path, sample_rate, clean_source.numpy().flatten())
        print(f"Saved noise-reduced audio for Speaker {i+1} at {output_path}")

    # Show a message box when processing is complete
    messagebox.showinfo("Processing Complete", "Noise-reduced audio files saved!")

def process_recorded_audio(file_path):
    """Process recorded audio for speaker separation and noise reduction."""
    sample_rate, mixture_waveform = read(file_path)
    mixture_waveform = torch.tensor(mixture_waveform).unsqueeze(0)  # Add batch dimension
    
    # Separate speakers
    separated_sources = separate_speech(model, mixture_waveform)
    
    # Apply noise reduction to separated sources
    for i, source in enumerate(separated_sources):
        clean_source = reduce_noise(source, sample_rate)
        
        # Save each noise-reduced speaker's audio
        output_path = f"recorded_noise_reduced_speaker_{i+1}.wav"
        write(output_path, sample_rate, clean_source.numpy().flatten())
        print(f"Saved noise-reduced audio for Speaker {i+1} at {output_path}")

    # Show a message box when processing is complete
    messagebox.showinfo("Processing Complete", "Noise-reduced audio files saved!")

def start_live_processing():
    """Start live audio processing."""
    process_live_audio()

def start_recorded_processing():
    """Start recorded audio processing after selecting a file."""
    file_path = filedialog.askopenfilename(title="Select Audio File", filetypes=[("WAV files", "*.wav")])
    if file_path:
        process_recorded_audio(file_path)
    else:
        messagebox.showwarning("Warning", "No file selected.")

# GUI setup
app = tk.Tk()
app.title("Speech Separation and Noise Suppression")

live_button = tk.Button(app, text="Process Live Audio", command=start_live_processing)
live_button.pack(pady=20)

recorded_button = tk.Button(app, text="Process Recorded Audio", command=start_recorded_processing)
recorded_button.pack(pady=20)

exit_button = tk.Button(app, text="Exit", command=app.quit)
exit_button.pack(pady=20)

# Run the GUI
app.mainloop()
