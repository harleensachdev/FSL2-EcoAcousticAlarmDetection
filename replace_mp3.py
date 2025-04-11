import os
import subprocess
import glob
from tqdm import tqdm

# Base directory for audio files
base_dir = "/Users/caramelloveschicken/Library/Mobile Documents/com~apple~CloudDocs/researchprojects/alarmcalldetection/data/audio_files/test/background"

# Output directory (same as input in this case)
output_dir = base_dir

def convert_mp3_to_wav(mp3_path):
    """Convert an MP3 file to WAV format using FFmpeg."""
    try:
        # Create WAV filename with same base name but .wav extension
        wav_filename = os.path.splitext(os.path.basename(mp3_path))[0] + ".wav"
        wav_path = os.path.join(output_dir, wav_filename)
        
        # Skip if WAV already exists
        if os.path.exists(wav_path):
            print(f"Skipping existing file: {wav_path}")
            return True
        
        # Run FFmpeg command
        cmd = [
            "ffmpeg", 
            "-i", mp3_path, 
            "-acodec", "pcm_s16le", 
            "-ar", "22050", 
            "-ac", "1",
            "-y",  # Overwrite output files without asking
            wav_path
        ]
        
        # Execute the command
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if result.returncode == 0:
            print(f"Successfully converted: {os.path.basename(mp3_path)} → {wav_filename}")
            return True
        else:
            print(f"Error converting {mp3_path}:")
            print(result.stderr.decode())
            return False
            
    except Exception as e:
        print(f"Exception processing {mp3_path}: {e}")
        return False

def main():
    # glob.glob searches for all files with * ('*' is wildcard, matching any char before .mp3)
    mp3_files = glob.glob(os.path.join(base_dir, "*.mp3"))
    
    if not mp3_files:
        print(f"No MP3 files found in {base_dir}")
        return
    
    print(f"Found {len(mp3_files)} MP3 files to convert")
    
    # Process each file with a progress bar
    success_count = 0
    for mp3_file in tqdm(mp3_files):
        if convert_mp3_to_wav(mp3_file):
            success_count += 1
    
    # Summary
    print(f"Conversion complete: {success_count}/{len(mp3_files)} files successfully converted")
    
    # Check if FFmpeg is installed
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        print("\nERROR: FFmpeg not found. Please install FFmpeg to use this script.")
        print("You can install it with: brew install ffmpeg (on macOS with Homebrew)")

if __name__ == "__main__":
    main()
