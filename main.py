import os
import subprocess

# Directories
input_directory = r"C:\Users\sarse\OneDrive\Рабочий стол\audio_wave"  # Change to your actual path
output_directory = r"C:\Users\sarse\OneDrive\Рабочий стол\transcribe_wave"  # Change to your actual path
os.makedirs(output_directory, exist_ok=True)

# Get all .mp3 files
mp3_files = [f for f in os.listdir(input_directory) if f.endswith('.mp3')]

# Transcribe each file
for filename in mp3_files:
    mp3_path = os.path.join(input_directory, filename)
    print(f"Processing: {mp3_path}")

    # Create an output directory for each file
    file_id = os.path.splitext(filename)[0]
    file_output_directory = os.path.join(output_directory, file_id)
    os.makedirs(file_output_directory, exist_ok=True)

    # Run Whisper command
    command = [
        "whisper", mp3_path,
        "--model", "small",
        "--language", "Russian",
        "--output_dir", file_output_directory
    ]
    subprocess.run(command)

# # Optionally, zip the output directory
# zip_command = ["zip", "-r", "/path/to/transcribe_wave.zip", output_directory]
# subprocess.run(zip_command)

print("Done!")