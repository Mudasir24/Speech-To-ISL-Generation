import subprocess

with open(r"C:\\Users\\Mohammed Mudasir\\OneDrive\\Desktop\\Major Project\\Towards-Automatic-Speech-to-SL\\Dataset\\train.txt") as f:
    for line in f:
        vid = line.strip()
        url = f"https://www.youtube.com/watch?v={vid}"

        subprocess.run([
            "python", "-m", "yt_dlp",
            "-f", "mp4",
            "--write-auto-subs",
            "--sub-lang", "en",
            "--sub-format", "vtt",
            "--ignore-errors",
            url
        ])