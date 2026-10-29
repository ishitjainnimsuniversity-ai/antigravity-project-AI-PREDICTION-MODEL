import os
import subprocess
import time
import re
import socket

SCRATCH_DIR = r"C:\Users\ishit jain\.gemini\antigravity\scratch"
DESKTOP_DIR = r"C:\Users\ishit jain\Desktop"
url_file_path = os.path.join(DESKTOP_DIR, "Vision_AI_Live_Website.txt")

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def log_status(message):
    print(message)
    try:
        with open(os.path.join(SCRATCH_DIR, "tunnel_debug.log"), "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
    except Exception:
        pass

log_status("Starting persistent tunnel launcher...")

# 1. Start Streamlit if not already running
if not is_port_in_use(8501):
    log_status("Streamlit is not running on port 8501. Starting it...")
    try:
        st_process = subprocess.Popen(
            ["streamlit", "run", "streamlit_app.py", "--server.port", "8501"],
            cwd=SCRATCH_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        log_status("Streamlit process spawned.")
    except Exception as e:
        log_status(f"Error starting Streamlit: {e}")
else:
    log_status("Streamlit is already running on port 8501.")

# 2. Wait for Streamlit port to become active (up to 15 seconds)
streamlit_ready = False
for _ in range(15):
    if is_port_in_use(8501):
        streamlit_ready = True
        log_status("Streamlit port 8501 is active!")
        break
    time.sleep(1)

if not streamlit_ready:
    log_status("Warning: Streamlit port 8501 did not become active in time. Proceeding anyway...")

# 3. Main Tunnel Loop (restarts tunnel if it dies)
while True:
    log_status("Starting SSH tunnel to localhost.run...")
    try:
        # Start SSH reverse tunnel
        ssh_command = ["ssh", "-o", "StrictHostKeyChecking=no", "-R", "80:127.0.0.1:8501", "nokey@localhost.run"]
        ssh_process = subprocess.Popen(
            ssh_command,
            cwd=SCRATCH_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        log_status("SSH tunnel process started. Waiting for public URL...")
        
        # Monitor output to extract public URL
        public_url = None
        start_time = time.time()
        
        # Read output line by line
        while ssh_process.poll() is None:
            line = ssh_process.stdout.readline()
            if not line:
                time.sleep(0.1)
                continue
            
            line_str = line.strip()
            log_status(f"Tunnel Output: {line_str}")
            
            # Look for a secure URL in the output
            # localhost.run URLs look like: https://xxxx.lhr.life or https://xxxx.lhr.pro
            match = re.search(r"https://[a-zA-Z0-9\-\.]+\.lhr\.(?:life|pro|run)", line_str)
            if match:
                public_url = match.group(0)
                log_status(f"Extracted Public URL: {public_url}")
                
                # Write to desktop file
                try:
                    with open(url_file_path, "w", encoding="utf-8") as f:
                        f.write("==================================================\n")
                        f.write("         VISION-AI LIVE CLINICAL DASHBOARD        \n")
                        f.write("==================================================\n\n")
                        f.write("Your public live website link is active!\n")
                        f.write(f"👉 {public_url}\n\n")
                        f.write("Note: This website is hosted securely from your PC.\n")
                        f.write("You can open this link on your phone, tablet,\n")
                        f.write("or share it with anyone in the world!\n")
                        f.write("==================================================\n")
                    log_status("Desktop file updated successfully!")
                except Exception as e:
                    log_status(f"Failed to write to desktop file: {e}")
                
            # If we've got the URL, we keep reading to keep the process alive
            # or if the process exits, poll will break the loop.
            
        exit_code = ssh_process.wait()
        log_status(f"SSH process terminated with exit code {exit_code}")
        
    except Exception as e:
        log_status(f"Error in tunnel loop: {e}")
        
    log_status("Tunnel disconnected. Retrying in 10 seconds...")
    time.sleep(10)
