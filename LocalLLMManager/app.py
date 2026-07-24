import os
import sqlite3
import re
import subprocess
import signal
import threading
import argparse
import logging
import sys
import shlex
import json
import socket
import requests
import traceback
import time
import pathlib
from time import sleep
from ggufvramestimator import run_estimator
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response
from flask_cors import CORS
from huggingface_hub import hf_hub_download, HfApi

# --- Configuration ---
logger = logging.getLogger(__name__)


APP_DIR = os.path.dirname(__file__)
# Set the folder where models will be stored
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
# Set the path for the local model database
DATABASE_FILE = os.path.join(os.path.dirname(__file__), "models.db")

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "123qazabcwsx098"


# Initialize Proxy App
proxy_app = Flask(__name__)
CORS(proxy_app)


llm_server_contexts = {}


STARTING_PORT = 8100


MULTI_PART_REGEX = re.compile(r"\d{5}-of-\d{5}.gguf$")


def check_default_env():
    env_path = os.path.join(os.path.abspath(APP_DIR), '.env')

    if not os.path.exists(env_path):
        default_env_content = (
            "# --- LocalLLMManager Environment Variables ---\n\n"
            "# Path to llama-server executable\n"
            "# LLAMA_BIN_PATH=\n\n"
            "# Host for the main web application\n"
            "# Default: 0.0.0.0 (listens on all available network interfaces)\n"
            "APP_HOST=0.0.0.0\n\n"
            "# Port for the main web application\n"
            "# Default: 5001\n"
            "APP_PORT=5001\n\n"
            "# Enable/disable Flask debug mode\n"
            "# Default: false\n"
            "APP_DEBUG=false\n\n"
            "# Enable/disable verbose (DEBUG level) logging\n"
            "# Default: false\n"
            "APP_VERBOSE=false\n\n"
            "# Host for the OpenAI-compatible proxy server\n"
            "# Default: 0.0.0.0\n"
            "PROXY_HOST=0.0.0.0\n\n"
            "# Port for the OpenAI-compatible proxy server\n"
            "# Default: 8080\n"
            "PROXY_PORT=8080\n\n"
            "# Directory to store downloaded GGUF models\n"
            "# Default: (path to LocalLLMManager/models)\n"
            "# You can override this with an absolute path if you like:\n"
            "# MODEL_DIR=\n\n"
            "# --- Multi-Node Cluster Configuration ---\n"
            "# Whether to include the local machine as Node 0 (default: true)\n"
            "# USE_LOCAL_NODE=true\n\n"
            "# Space-separated lists of hostnames, usernames, and SSH keyfiles\n"
            "# NODE_HOSTNAMES=\n"
            "# NODE_USERNAMES=\n"
            "# NODE_KEYFILES=\n\n"
            "# Space-separated lists matching NODE_HOSTNAMES for directories and bin paths on remote nodes\n"
            "# REMOTE_MODEL_DIRS=\n"
            "# REMOTE_LLAMA_BIN_PATHS=\n"
        )
        try:
            with open(env_path, 'w') as f:
                f.write(default_env_content)
            print(f"INFO: No .env file found. Created a default .env file at {env_path}")
        except Exception as e:
            print(f"WARNING: Failed to create default .env file at {env_path}: {e}")


check_default_env()
load_dotenv(os.path.join(APP_DIR, '.env'))

LLAMA_BIN_PATH = os.getenv("LLAMA_BIN_PATH", "")
LLAMA_SERVER_CMD = str(pathlib.Path(LLAMA_BIN_PATH).joinpath('llama-server'))

NODES = []

def parse_space_separated_env(var_name):
    val = os.getenv(var_name, "").strip()
    if not val:
        return []
    try:
        return shlex.split(val)
    except ValueError as e:
        logger.warning(f"Error parsing environment variable {var_name}: {e}")
        return val.split()


def load_nodes_config(use_local_node_flag=None):
    """
    Parses and returns the configured list of nodes.
    A node is dict with keys: id, host, username, keyfile, model_dir, llama_bin_path, is_local.
    """
    nodes = []
    
    use_local = True
    if use_local_node_flag is not None:
        use_local = use_local_node_flag
    else:
        env_val = os.getenv("USE_LOCAL_NODE", "true").lower()
        use_local = env_val in ("true", "1", "yes")

    if use_local:
        nodes.append({
            "id": 0,
            "host": "localhost",
            "username": "",
            "keyfile": "",
            "model_dir": MODEL_DIR,
            "llama_bin_path": LLAMA_BIN_PATH,
            "is_local": True
        })

    hostnames = parse_space_separated_env("NODE_HOSTNAMES")
    usernames = parse_space_separated_env("NODE_USERNAMES")
    keyfiles = parse_space_separated_env("NODE_KEYFILES")
    model_dirs = parse_space_separated_env("REMOTE_MODEL_DIRS")
    llama_bin_paths = parse_space_separated_env("REMOTE_LLAMA_BIN_PATHS")

    num_remotes = len(hostnames)
    if not (len(usernames) == num_remotes and 
            len(keyfiles) == num_remotes and 
            len(model_dirs) == num_remotes and 
            len(llama_bin_paths) == num_remotes):
        logger.error(
            f"Inconsistent remote nodes configuration. Hostnames: {len(hostnames)}, "
            f"Usernames: {len(usernames)}, Keyfiles: {len(keyfiles)}, "
            f"Model dirs: {len(model_dirs)}, Bin paths: {len(llama_bin_paths)}"
        )
        
    for i in range(num_remotes):
        username = usernames[i] if i < len(usernames) else ""
        keyfile = keyfiles[i] if i < len(keyfiles) else ""
        model_dir = model_dirs[i] if i < len(model_dirs) else ""
        llama_bin_path = llama_bin_paths[i] if i < len(llama_bin_paths) else ""
        
        nodes.append({
            "id": len(nodes),
            "host": hostnames[i],
            "username": username,
            "keyfile": keyfile,
            "model_dir": model_dir,
            "llama_bin_path": llama_bin_path,
            "is_local": False
        })
        
    return nodes


def run_ssh_command(node, command, timeout=60):
    """Runs a command on a node (local or remote via SSH). Returns (returncode, stdout, stderr)."""
    if node["is_local"]:
        try:
            res = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return res.returncode, res.stdout, res.stderr
        except subprocess.TimeoutExpired as e:
            return -1, "", f"Local command timed out: {e}"
        except Exception as e:
            return -1, "", f"Local command failed: {e}"
    else:
        ssh_cmd = [
            "ssh",
            "-i", node["keyfile"],
            "-o", "StrictHostKeyChecking=accept-new",
            f"{node['username']}@{node['host']}",
            command
        ]
        try:
            res = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return res.returncode, res.stdout, res.stderr
        except subprocess.TimeoutExpired as e:
            return -1, "", f"SSH command timed out: {e}"
        except Exception as e:
            return -1, "", f"SSH connection or execution failed: {e}"


def sync_code_to_node(node):
    """Syncs local code files (excluding models/DB) to the remote node via SCP."""
    if node["is_local"]:
        return True, "No sync needed for local node."
        
    try:
        # Copy app.py
        app_src = os.path.join(APP_DIR, "app.py")
        app_dest = f"{node['username']}@{node['host']}:~/LocalLLMManager/LocalLLMManager/app.py"
        res = subprocess.run([
            "scp", "-i", node["keyfile"], "-o", "StrictHostKeyChecking=accept-new",
            app_src, app_dest
        ], capture_output=True, text=True, timeout=30)
        if res.returncode != 0:
            return False, f"Failed to sync app.py: {res.stderr.strip()}"
            
        # Copy ggufvramestimator.py
        est_src = os.path.join(APP_DIR, "ggufvramestimator.py")
        est_dest = f"{node['username']}@{node['host']}:~/LocalLLMManager/LocalLLMManager/ggufvramestimator.py"
        res = subprocess.run([
            "scp", "-i", node["keyfile"], "-o", "StrictHostKeyChecking=accept-new",
            est_src, est_dest
        ], capture_output=True, text=True, timeout=30)
        if res.returncode != 0:
            return False, f"Failed to sync ggufvramestimator.py: {res.stderr.strip()}"
            
        # Copy templates
        tmpl_src = os.path.join(APP_DIR, "templates")
        tmpl_dest = f"{node['username']}@{node['host']}:~/LocalLLMManager/LocalLLMManager/"
        res = subprocess.run([
            "scp", "-i", node["keyfile"], "-o", "StrictHostKeyChecking=accept-new", "-r",
            tmpl_src, tmpl_dest
        ], capture_output=True, text=True, timeout=30)
        if res.returncode != 0:
            return False, f"Failed to sync templates: {res.stderr.strip()}"
            
        # Copy static
        static_src = os.path.join(APP_DIR, "static")
        static_dest = f"{node['username']}@{node['host']}:~/LocalLLMManager/LocalLLMManager/"
        res = subprocess.run([
            "scp", "-i", node["keyfile"], "-o", "StrictHostKeyChecking=accept-new", "-r",
            static_src, static_dest
        ], capture_output=True, text=True, timeout=30)
        if res.returncode != 0:
            return False, f"Failed to sync static assets: {res.stderr.strip()}"
            
        return True, "Code sync complete."
    except Exception as e:
        return False, str(e)


def setup_remote_node(node):
    """
    Sets up the remote node by verifying python 3.11+, cloning the repo,
    setting up a virtual environment, and installing requirements.
    Returns (success, message).
    """
    if node["is_local"]:
        return True, "Local node is already set up."

    # 1. Verify python 3.11+ is installed
    py_check_cmd = "python3 -c 'import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\")'"
    rc, stdout, stderr = run_ssh_command(node, py_check_cmd)
    if rc != 0:
        return False, f"Failed to run python3 check on remote node: {stderr.strip()}"
    
    version_str = stdout.strip()
    try:
        major, minor = map(int, version_str.split('.'))
        if major < 3 or (major == 3 and minor < 11):
            return False, f"Python version on remote node is too old: {version_str}. Require >= 3.11."
    except Exception as e:
        return False, f"Could not parse python version '{version_str}': {e}"

    # 2. Check and clone repository
    clone_cmd = (
        'if [ ! -d "$HOME/LocalLLMManager" ]; then '
        'git clone https://github.com/orcephrye/LocalLLMManager.git "$HOME"/LocalLLMManager; '
        'else '
        'cd "$HOME"/LocalLLMManager && git reset --hard HEAD && git pull; '
        'fi'
    )
    rc, stdout, stderr = run_ssh_command(node, clone_cmd)
    if rc != 0:
        return False, f"Failed to check/clone repository: {stderr.strip()}"

    # 3. Create virtual environment .venv if it doesn't exist
    venv_cmd = (
        'if [ ! -d "$HOME/LocalLLMManager/.venv" ]; then '
        'python3 -m venv "$HOME"/LocalLLMManager/.venv; '
        'else '
        'echo "Virtual environment already exists."; '
        'fi'
    )
    rc, stdout, stderr = run_ssh_command(node, venv_cmd)
    if rc != 0:
        return False, f"Failed to create virtual environment: {stderr.strip()}"

    # 4. Upgrade pip and install requirements.txt
    pip_install_cmd = (
        '"$HOME"/LocalLLMManager/.venv/bin/pip install --upgrade pip && '
        '"$HOME"/LocalLLMManager/.venv/bin/pip install -r "$HOME"/LocalLLMManager/requirements.txt'
    )
    rc, stdout, stderr = run_ssh_command(node, pip_install_cmd, timeout=180)
    if rc != 0:
        return False, f"Failed to install requirements: {stderr.strip()}"

    # 5. Sync local codebase to remote node via scp
    sync_success, sync_msg = sync_code_to_node(node)
    if not sync_success:
        return False, f"Initial code sync failed: {sync_msg}"

    return True, "Remote node setup completed successfully."


def parse_device_line(line):
    """
    Parses a single device line from 'llama-server --list-devices'.
    Returns a dict with 'device_id', 'device_name', 'memory_info', and 'raw'.
    
    Example input:
      "Vulkan0: AMD Radeon 890M Graphics (RADV STRIX1) (24266 MiB, 21575 MiB free)"
    Output:
      {
        "device_id": "Vulkan0",
        "device_name": "AMD Radeon 890M Graphics (RADV STRIX1)",
        "memory_info": "24266 MiB, 21575 MiB free",
        "raw": "Vulkan0: AMD Radeon 890M Graphics (RADV STRIX1) (24266 MiB, 21575 MiB free)"
      }
    """
    if ":" not in line:
        return None
    device_id, rest = line.split(":", 1)
    device_id = device_id.strip()
    rest = rest.strip()
    
    match = re.match(r"^(.*?)\s*\(([^()]*?(?:MiB|GiB|MB|GB|B|free|total|used|RAM)[^()]*?)\)$", rest, re.IGNORECASE)
    if not match:
        match = re.match(r"^(.*?)\s*\(([^()]+)\)$", rest)

    if match:
        dev_name = match.group(1).strip()
        mem_info = match.group(2).strip()
    else:
        dev_name = rest
        mem_info = "N/A"
        
    if not dev_name:
        dev_name = device_id

    return {
        "device_id": device_id,
        "device_name": dev_name,
        "memory_info": mem_info,
        "raw": line.strip()
    }


def get_node_devices(node_or_id):
    """
    Retrieves compute/acceleration devices seen by llama-server via 'llama-server --list-devices' on a specific node.
    Accepts either a node dictionary or node_id (int or str).
    
    Returns a list of dicts:
        [
            {
                "device_id": "Vulkan0",
                "device_name": "AMD Radeon 890M Graphics (RADV STRIX1)",
                "memory_info": "24266 MiB, 21575 MiB free",
                "raw": "Vulkan0: AMD Radeon 890M Graphics (RADV STRIX1) (24266 MiB, 21575 MiB free)"
            },
            ...
        ]
    """
    if isinstance(node_or_id, dict):
        node = node_or_id
    else:
        try:
            nid = int(node_or_id)
            nodes = load_nodes_config()
            node = next((n for n in nodes if n["id"] == nid), None)
        except (ValueError, TypeError):
            node = None

    if not node:
        raise ValueError(f"Node '{node_or_id}' not found.")

    llama_bin = node.get("llama_bin_path", "")
    if llama_bin:
        llama_server_cmd = str(pathlib.Path(llama_bin).joinpath("llama-server"))
    else:
        llama_server_cmd = "llama-server"

    devices_cmd = f"{llama_server_cmd} --list-devices"
    rc, stdout, stderr = run_ssh_command(node, devices_cmd)

    if rc != 0:
        raise RuntimeError(f"Failed to list devices on node {node['id']} ({node['host']}): {stderr.strip()}")

    devices = []
    found_header = False
    for line in stdout.splitlines():
        line_str = line.strip()
        if not line_str:
            continue
        if "Available devices:" in line_str:
            found_header = True
            continue
        if found_header:
            parsed = parse_device_line(line_str)
            if parsed:
                devices.append(parsed)

    return devices


def validate_node(node):
    """
    Validates a node configuration.
    For local node, always returns True.
    For remote nodes:
      1. Tests SSH connection.
      2. If remote setup has not run (repo or venv missing), runs setup_remote_node.
      3. Verifies llama-server --list-devices has at least 1 device.
      4. Verifies app.py --help runs successfully using the remote venv.
    Returns (is_valid, status_message).
    """
    if node["is_local"]:
        return True, "Local node is active."

    # 1. First test basic SSH access
    rc, stdout, stderr = run_ssh_command(node, "echo 'ping'")
    if rc != 0:
        return False, f"SSH connection failed: {stderr.strip()}"

    # 2. Check if remote setup is needed (check if venv exists)
    check_setup_cmd = '[ -d "$HOME/LocalLLMManager/.venv" ]'
    rc_setup, _, _ = run_ssh_command(node, check_setup_cmd)
    if rc_setup != 0:
        logger.info(f"Node {node['id']} ({node['host']}) setup not found. Initiating remote setup...")
        setup_success, setup_msg = setup_remote_node(node)
        if not setup_success:
            return False, f"Remote setup failed: {setup_msg}"
    else:
        # If setup already exists, just sync code files to ensure latest version is present
        sync_success, sync_msg = sync_code_to_node(node)
        if not sync_success:
            return False, f"Remote code sync failed: {sync_msg}"

    # 3. Verify llama-server --list-devices
    devices = []
    try:
        device_dicts = get_node_devices(node)
        if len(device_dicts) < 1:
            return False, "No active acceleration/compute devices found on the remote node (zero devices)."
        devices = [d["device"] for d in device_dicts]
    except Exception as e:
        return False, f"Failed to list devices on remote node: {e}"

    # 4. Verify remote app.py --help can run
    help_cmd = '"$HOME"/LocalLLMManager/.venv/bin/python3 "$HOME"/LocalLLMManager/LocalLLMManager/app.py --help'
    rc_help, stdout_help, stderr_help = run_ssh_command(node, help_cmd)
    if rc_help != 0:
        return False, f"Failed to run remote app.py --help: {stderr_help.strip()}"
    
    if "usage: app.py" not in stdout_help:
        return False, "Remote app.py help output format invalid."

    return True, f"Node is valid and active. Detected devices: {', '.join(devices)}"



def parse_logs(log_line, metrics):
    lines = log_line.strip().split()
    try:
        tokens_per_sec = float(lines[-4])
        metrics["total_requests"] += 1
        metrics["last_tps"] = tokens_per_sec
        if metrics["average_tps"] == 0.0:
            metrics["average_tps"] = tokens_per_sec
        else:
            metrics["average_tps"] = (metrics["average_tps"] + tokens_per_sec) / 2.0
    except ValueError as e:
        logger.warning(f"Failed to get tokens per second with error: {e}")
        logger.debug(f"llama-server (log): {log_line.strip()}")
    except Exception as e:
        logger.error(f"Error parsing log line: {e} | Line: {log_line.strip()}")


def drain_server_stderr(proc, metrics):
    """
    Reads stdout from the server process and just prints it.
    This prevents the stdout buffer from filling up and stalling the server.
    This runs in a background thread.
    """
    try:
        for line in iter(proc.stderr.readline, ''):
            if not line:
                break
            if 'eval time =' in line:
                parse_logs(line, metrics)
            logger.debug(f"llama-server (stderr): {line.strip()}")
            logger.info("Log reader thread (stderr) exiting.")
    except Exception as e:
        logger.error(f"Log reader thread (stderr) crashed: {e}")


def drain_server_stdout(proc):
    """
    Reads stdout from the server process and just prints it.
    This prevents the stdout buffer from filling up and stalling the server.
    This runs in a background thread.
    """
    try:
        for line in iter(proc.stdout.readline, ''):
            if not line:
                break
            logger.debug(f"llama-server (stderr): {line.strip()}")
            logger.info("Log reader thread (stderr) exiting.")
    except Exception as e:
        logger.error(f"Log reader thread (stderr) crashed: {e}")


def find_next_available_port(start_port):
    """Finds an unused port, starting from start_port."""
    port = start_port
    while True:
        if port in llm_server_contexts:
            port += 1
            continue

        try:
            # Check if port is system-wide free by trying to bind
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
            # If bind was successful, this port is free
            return port
        except OSError:
            # Port is already in use by another process
            logger.info(f"Port {port} is in use by another process, trying next.")
            port += 1


def find_remote_port(node):
    """Finds an available port on the remote node."""
    cmd = 'python3 -c "import socket; s=socket.socket(); s.bind((\'\', 0)); print(s.getsockname()[1]); s.close()"'
    rc, stdout, stderr = run_ssh_command(node, cmd)
    if rc == 0:
        try:
            return int(stdout.strip())
        except ValueError:
            pass
    return 8100 # Fallback



def init_db():
    """Initializes the SQLite database and tables."""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS gguf_models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            repo_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            quantization TEXT,
            last_scanned TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(repo_id, filename)
        )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_repo_id ON gguf_models (repo_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_quantization ON gguf_models (quantization)")

        c.execute("""
        CREATE TABLE IF NOT EXISTS model_args (
            model_filename TEXT PRIMARY KEY,
            custom_args TEXT
        )
        """)

        conn.commit()
        conn.close()
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")


def parse_quantization(filename):
    """Parses the quantization type from a GGUF filename using regex."""
    match = re.search(r"^.*-(?P<Quant>TQ[0-8]_[0-8]|BF16|Q[0-8]_([0-8]|K_?\S{0,2})|IQ[0-8]|F16|F32).*\.gguf$", filename, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return "unknown"


def update_model_database():
    """Scans Hugging Face for GGUF models and updates the local database."""
    logger.info("Starting Hugging Face model scan...")
    api = HfApi()
    conn = sqlite3.connect(DATABASE_FILE)
    c = conn.cursor()

    try:
        fetch_failure_count = 0
        models = api.list_models(filter="gguf", apps="llama.cpp", sort="downloads", limit=500)

        for model in models:
            repo_id = getattr(model, 'modelId', '')
            logger.debug(f"Scanning repo: {repo_id}")
            try:
                repo_info = api.repo_info(repo_id, files_metadata=True)
                if repo_info.siblings:
                    for sibling in repo_info.siblings:
                        filename = sibling.rfilename
                        if filename.endswith(".gguf"):
                            quant = parse_quantization(filename)
                            c.execute("""
                            INSERT OR IGNORE INTO gguf_models (repo_id, filename, quantization)
                            VALUES (?, ?, ?)
                            """, (repo_id, filename, quant))
                fetch_failure_count = 0
            except Exception as e:
                logger.warning(f"Could not scan repo {repo_id}: {e}")
                fetch_failure_count += 1
                if fetch_failure_count > 3:
                    logger.error(f"To many failures stopping scan...")
                    break
            finally:
              sleep(0.01)

        conn.commit()
        conn.close()
        logger.info("Database update complete.")
    except Exception as e:
        logger.error(f"Failed to fetch models from Hugging Face: {e}")
        if conn:
            conn.close()
        raise e


def search_models_db(query):
    """Searches the local SQLite database for models."""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        if not query:
            c.execute("SELECT * FROM gguf_models ORDER BY id DESC LIMIT 100")
        else:
            search_term = f"%{query}%"
            c.execute("""
            SELECT * FROM gguf_models
            WHERE repo_id LIKE ? OR filename LIKE ? OR quantization LIKE ?
            ORDER BY repo_id
            LIMIT 200
            """, (search_term, search_term, search_term))

        results = c.fetchall()
        conn.close()
        return results
    except Exception as e:
        logger.error(f"Error searching database: {e}")
        return []


def get_all_custom_args():
    """Fetches all saved custom arguments from the database."""
    args_dict = {}
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS model_args (
            model_filename TEXT PRIMARY KEY,
            custom_args TEXT
        )
        """)

        c.execute("SELECT model_filename, custom_args FROM model_args")
        results = c.fetchall()
        conn.close()
        for row in results:
            args_dict[row['model_filename']] = row['custom_args']
        return args_dict
    except Exception as e:
        logger.error(f"Error fetching custom args: {e}")
        return {}

def save_custom_args(model_file, custom_args_str):
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        c = conn.cursor()
        c.execute("""
                INSERT OR REPLACE INTO model_args (model_filename, custom_args)
                VALUES (?, ?)
                """, (model_file, custom_args_str))
        conn.commit()
        conn.close()
        logger.info(f"Saved custom args for {model_file}")
    except Exception as e:
        logger.error(f"Failed to save custom args for {model_file}: {e}")


def get_context_by_model_name(model_name):
    """Finds the server context by the model's filename."""
    for context in llm_server_contexts.values():
        if context['name'] == model_name:
            return context
    return None

def get_context_by_model_alias(model_alias):
    """Finds the server context by the model's alias."""
    for context in llm_server_contexts.values():
        if context['alias'] == model_alias:
            return context
    return None

def search_huggingface(query):
    """Searches Hugging Face for GGUF models supporting llama.cpp."""
    api = HfApi()
    try:
        print(f"Searching Hugging Face for '{query}'...")
        models = api.list_models(
            search=query,
            filter="gguf",
            apps="llama.cpp",
            sort="downloads",
            limit=200
        )
        return models
    except Exception as e:
        logger.error(f"Error searching Hugging Face: {e}")
        flash(f"Error searching Hugging Face: {e}", "error")
        return []

def list_gguf_files(repo_id):
    """Lists GGUF files for a given Hugging Face repository."""
    api = HfApi()
    try:
        logger.info(f"Listing GGUF files for repo: {repo_id}")
        repo_info = api.repo_info(repo_id, files_metadata=True)
        gguf_files = [sibling.rfilename for sibling in repo_info.siblings if sibling.rfilename.endswith(".gguf")]
        return gguf_files
    except Exception as e:
        logger.error(f"Error listing files for repo {repo_id}: {e}")
        flash(f"Error listing files: {e}", "error")
        return []


def get_local_models():
    """Finds all .gguf files in the MODEL_DIR recursively."""
    models = []
    if not os.path.exists(MODEL_DIR):
        return models
    for root, dirs, files in os.walk(MODEL_DIR):
        # Prevent walking hidden folders like .git
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for filename in files:
            if filename.endswith(".gguf"):
                # Skip mmproj models from primary select
                if "mmproj" in filename.lower():
                    continue
                rel_path = os.path.relpath(os.path.join(root, filename), MODEL_DIR)
                rel_path = rel_path.replace(os.path.sep, '/')
                if MULTI_PART_REGEX.findall(filename):
                    if '00001-of-' in filename:
                        models.append(rel_path)
                    continue
                models.append(rel_path)
    return models


import shutil

def format_size(size_bytes):
    if size_bytes is None:
        return "Unknown"
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def get_local_repos_and_files():
    repos = {}
    if not os.path.exists(MODEL_DIR):
        return []
    
    for root, dirs, files in os.walk(MODEL_DIR):
        # Prevent walking hidden folders like .git
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        rel_dir = os.path.relpath(root, MODEL_DIR)
        if rel_dir == '.':
            repo_name = "Root"
        else:
            repo_name = rel_dir.replace(os.path.sep, '/')
            
        for file in files:
            file_path = os.path.join(root, file)
            # Resolve symlink to get actual size
            real_path = os.path.realpath(file_path) if os.path.islink(file_path) else file_path
            
            size_bytes = 0
            if os.path.exists(real_path):
                size_bytes = os.path.getsize(real_path)
                
            file_entry = {
                "name": file,
                "rel_path": os.path.relpath(file_path, MODEL_DIR).replace(os.path.sep, '/'),
                "abs_path": file_path,
                "size_bytes": size_bytes,
                "size_str": format_size(size_bytes),
                "is_symlink": os.path.islink(file_path)
            }
            
            if repo_name not in repos:
                repos[repo_name] = {
                    "repo_name": repo_name,
                    "repo_path": root,
                    "files": [],
                    "total_size_bytes": 0
                }
            repos[repo_name]["files"].append(file_entry)
            repos[repo_name]["total_size_bytes"] += size_bytes

    repo_list = list(repos.values())
    
    for r in repo_list:
        r["total_size_str"] = format_size(r["total_size_bytes"])
        
    repo_list.sort(key=lambda x: (x["repo_name"] == "Root", x["repo_name"]))
    return repo_list


def get_disk_space_info():
    try:
        total, used, free = shutil.disk_usage(MODEL_DIR)
        return {
            "total": format_size(total),
            "used": format_size(used),
            "free": format_size(free),
            "used_pct": f"{(used / total) * 100:.1f}%" if total > 0 else "0%"
        }
    except Exception as e:
        logger.error(f"Error getting disk usage: {e}")
        return {
            "total": "Unknown",
            "used": "Unknown",
            "free": "Unknown",
            "used_pct": "0%"
        }


def is_path_in_model_dir(path):
    abs_path = os.path.abspath(path)
    abs_model_dir = os.path.abspath(MODEL_DIR)
    return abs_path.startswith(abs_model_dir) and abs_path != abs_model_dir


def delete_local_file(file_path):
    try:
        if os.path.islink(file_path):
            real_path = os.path.realpath(file_path)
            if os.path.exists(real_path):
                os.remove(real_path)
            os.remove(file_path)
        else:
            if os.path.exists(file_path):
                os.remove(file_path)
        return True, "File deleted successfully."
    except Exception as e:
        return False, str(e)


def delete_local_repo(repo_dir):
    try:
        if not os.path.exists(repo_dir):
            return False, "Repository directory does not exist."
        
        for root, dirs, files in os.walk(repo_dir, topdown=False):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.islink(file_path):
                    real_path = os.path.realpath(file_path)
                    if os.path.exists(real_path):
                        os.remove(real_path)
                    os.remove(file_path)
                else:
                    os.remove(file_path)
            for d in dirs:
                os.rmdir(os.path.join(root, d))
        os.rmdir(repo_dir)
        
        # Clean up any empty parent directories up to MODEL_DIR
        parent = os.path.dirname(repo_dir)
        while parent and parent != MODEL_DIR and len(os.path.abspath(parent)) > len(os.path.abspath(MODEL_DIR)):
            try:
                if not os.listdir(parent):
                    os.rmdir(parent)
                    parent = os.path.dirname(parent)
                else:
                    break
            except Exception:
                break
                
        return True, "Repository deleted successfully."
    except Exception as e:
        return False, str(e)


# --- Flask Routes ---
@app.route('/')
def index():
    """Main page - displays controls and output."""

    local_models = []
    first_active_node = next((n for n in NODES if n.get("is_active")), None)
    if first_active_node:
        if first_active_node["is_local"]:
            local_models = get_local_models()
        else:
            try:
                cmd_repos = (
                    f'"$HOME"/LocalLLMManager/.venv/bin/python3 -c "'
                    f'import sys, json; sys.path.append(\'$HOME/LocalLLMManager/LocalLLMManager\'); '
                    f'import app; app.MODEL_DIR=\'{first_active_node["model_dir"]}\'; '
                    f'print(json.dumps(app.get_local_models()))"'
                )
                rc, stdout, _ = run_ssh_command(first_active_node, cmd_repos)
                if rc == 0:
                    local_models = json.loads(stdout.strip())
            except Exception as e:
                logger.error(f"Error fetching remote models for index page: {e}")

    all_custom_args = get_all_custom_args()
    all_custom_args_json = json.dumps(all_custom_args)

    proxy_host = app.config.get('PROXY_HOST', 'N/A')
    proxy_port = app.config.get('PROXY_PORT', 'N/A')
    display_host = "127.0.0.1" if proxy_host == "0.0.0.0" else proxy_host

    downloads_list = []
    for key, data in background_downloads.items():
        node_id, repo_id, val = key
        downloads_list.append({
            "node_id": node_id,
            "repo_id": repo_id,
            "val": val,
            "status": data["status"],
            "message": data["message"],
            "timestamp": data["timestamp"]
        })

    return render_template(
        'index.html',
        models=local_models,
        contexts=llm_server_contexts,
        all_custom_args_json=all_custom_args_json,
        proxy_host=proxy_host,
        display_host=display_host,
        proxy_port=proxy_port,
        nodes=NODES,
        downloads_list=downloads_list
    )


@app.route('/browse')
def browse_models():
    """Displays the model database search/browse page."""
    search_query = request.args.get('q', '')
    models = search_models_db(search_query)

    return render_template(
        'browse.html',
        models=models,
        search_query=search_query,
        nodes=NODES
    )


@app.route('/manage')
def manage_local_models():
    """Displays local models and files with disk usage and delete actions."""
    node_id_str = request.args.get('node_id', '0')
    try:
        node_id = int(node_id_str)
    except ValueError:
        node_id = 0
        
    node = next((n for n in NODES if n["id"] == node_id), None)
    if not node or not node.get("is_active"):
        # Fallback to local active node
        node = next((n for n in NODES if n["is_local"]), NODES[0])
        node_id = node["id"]
        
    if node["is_local"]:
        repos = get_local_repos_and_files()
        disk_info = get_disk_space_info()
    else:
        # Fetch remote repos
        cmd_repos = (
            f'"$HOME"/LocalLLMManager/.venv/bin/python3 -c "'
            f'import sys, json; sys.path.append(\'$HOME/LocalLLMManager/LocalLLMManager\'); '
            f'import app; app.MODEL_DIR=\'{node["model_dir"]}\'; '
            f'print(json.dumps(app.get_local_repos_and_files()))"'
        )
        rc_repos, stdout_repos, stderr_repos = run_ssh_command(node, cmd_repos)
        if rc_repos == 0:
            try:
                repos = json.loads(stdout_repos.strip())
            except Exception as e:
                repos = []
                flash(f"Failed to parse remote models: {e}", "error")
        else:
            repos = []
            flash(f"Failed to load remote models list: {stderr_repos.strip()}", "error")
            
        # Fetch remote disk space
        cmd_disk = (
            f'"$HOME"/LocalLLMManager/.venv/bin/python3 -c "'
            f'import sys, json; sys.path.append(\'$HOME/LocalLLMManager/LocalLLMManager\'); '
            f'import app; app.MODEL_DIR=\'{node["model_dir"]}\'; '
            f'print(json.dumps(app.get_disk_space_info()))"'
        )
        rc_disk, stdout_disk, _ = run_ssh_command(node, cmd_disk)
        if rc_disk == 0:
            try:
                disk_info = json.loads(stdout_disk.strip())
            except Exception:
                disk_info = {"total": "N/A", "used": "N/A", "free": "N/A", "percent": 0.0}
        else:
            disk_info = {"total": "N/A", "used": "N/A", "free": "N/A", "percent": 0.0}

    return render_template(
        'manage.html',
        repos=repos,
        disk_info=disk_info,
        nodes=NODES,
        selected_node_id=node_id
    )


@app.route('/delete_file', methods=['POST'])
def delete_file_route():
    """Endpoint to delete a specific model file."""
    file_path = request.form.get('file_path')
    node_id_str = request.form.get('node_id', '0')
    
    if not file_path:
        flash("File path is required.", "error")
        return redirect(url_for('manage_local_models', node_id=node_id_str))

    try:
        node_id = int(node_id_str)
        node = next((n for n in NODES if n["id"] == node_id), None)
        if not node or not node.get("is_active"):
            flash("Selected node is inactive or invalid.", "error")
            return redirect(url_for('manage_local_models'))
    except ValueError:
        flash("Invalid node selection.", "error")
        return redirect(url_for('manage_local_models'))

    if node["is_local"]:
        if not is_path_in_model_dir(file_path):
            flash("Invalid file path.", "error")
            return redirect(url_for('manage_local_models', node_id=node_id_str))

        success, message = delete_local_file(file_path)
        if success:
            flash(message, "success")
        else:
            flash(f"Error deleting file: {message}", "error")
    else:
        # Delete file remotely via SSH by invoking python code on the remote machine
        cmd_delete = (
            f'"$HOME"/LocalLLMManager/.venv/bin/python3 -c "'
            f'import sys; sys.path.append(\'$HOME/LocalLLMManager/LocalLLMManager\'); '
            f'import app; app.MODEL_DIR=\'{node["model_dir"]}\'; '
            f'success, msg = app.delete_local_file(\'{file_path}\'); '
            f'print(f\'SUCCESS: {msg}\' if success else f\'ERROR: {msg}\')"'
        )
        rc, stdout, stderr = run_ssh_command(node, cmd_delete)
        if rc == 0:
            out = stdout.strip()
            if out.startswith("SUCCESS:"):
                flash(out.replace("SUCCESS:", "").strip(), "success")
            else:
                flash(out.replace("ERROR:", "").strip(), "error")
        else:
            flash(f"Failed to delete file on remote node: {stderr.strip()}", "error")

    return redirect(url_for('manage_local_models', node_id=node_id_str))


@app.route('/delete_repo', methods=['POST'])
def delete_repo_route():
    """Endpoint to delete an entire repository directory."""
    repo_path = request.form.get('repo_path')
    node_id_str = request.form.get('node_id', '0')
    
    if not repo_path:
        flash("Repository path is required.", "error")
        return redirect(url_for('manage_local_models', node_id=node_id_str))

    try:
        node_id = int(node_id_str)
        node = next((n for n in NODES if n["id"] == node_id), None)
        if not node or not node.get("is_active"):
            flash("Selected node is inactive or invalid.", "error")
            return redirect(url_for('manage_local_models'))
    except ValueError:
        flash("Invalid node selection.", "error")
        return redirect(url_for('manage_local_models'))

    if node["is_local"]:
        if not is_path_in_model_dir(repo_path):
            flash("Invalid repository path.", "error")
            return redirect(url_for('manage_local_models', node_id=node_id_str))

        success, message = delete_local_repo(repo_path)
        if success:
            flash(message, "success")
        else:
            flash(f"Error deleting repository: {message}", "error")
    else:
        # Delete repo remotely via SSH by invoking python code on the remote machine
        cmd_delete = (
            f'"$HOME"/LocalLLMManager/.venv/bin/python3 -c "'
            f'import sys; sys.path.append(\'$HOME/LocalLLMManager/LocalLLMManager\'); '
            f'import app; app.MODEL_DIR=\'{node["model_dir"]}\'; '
            f'success, msg = app.delete_local_repo(\'{repo_path}\'); '
            f'print(f\'SUCCESS: {msg}\' if success else f\'ERROR: {msg}\')"'
        )
        rc, stdout, stderr = run_ssh_command(node, cmd_delete)
        if rc == 0:
            out = stdout.strip()
            if out.startswith("SUCCESS:"):
                flash(out.replace("SUCCESS:", "").strip(), "success")
            else:
                flash(out.replace("ERROR:", "").strip(), "error")
        else:
            flash(f"Failed to delete repository on remote node: {stderr.strip()}", "error")

    return redirect(url_for('manage_local_models', node_id=node_id_str))


@app.route('/search_hf')
def search_hf_models():
    """Displays the Hugging Face search page."""
    search_query = request.args.get('q', '')
    models = []
    if search_query:
        models = search_huggingface(search_query)

    return render_template(
        'search.html',
        models=models,
        search_query=search_query,
        nodes=NODES
    )


@app.route('/list_files', methods=['GET'])
def list_files_route():
    """Route to list GGUF files for a given repo."""
    repo_id = request.args.get('repo_id')
    if not repo_id:
        flash("Repo ID is required to list files.", "error")
        return redirect(url_for('search_hf_models'))

    gguf_files = list_gguf_files(repo_id)

    return render_template(
        'list_files.html',
        repo_id=repo_id,
        gguf_files=gguf_files,
        nodes=NODES
    )


@app.route('/status')
def model_status():
    """Displays the status and inference page for the currently loaded model."""
    is_running = bool(llm_server_contexts)

    return render_template(
        'status.html',
        is_running=is_running,
        contexts=llm_server_contexts
    )

@app.route('/refresh_cache', methods=['POST'])
def refresh_cache():
    """Route to trigger the cache refresh."""
    try:
        update_model_database()
        flash("Model database refresh complete! Scanned top 500 GGUF repos.", "success")
    except Exception as e:
        logger.error(f"Error refreshing cache: {e}")
        flash(f"Error refreshing cache: {e}. Check console for details.", "error")
    return redirect(url_for('browse_models'))


def download_model_from_hf(repo_id, filename=None, quant_tag=None):
    """
    Downloads a GGUF model and its companion mmproj files from Hugging Face.
    Returns (success, filename_or_error, mmproj_filename_or_None)
    """
    if not repo_id:
        return False, "Repo ID is required.", None

    try:
        api = HfApi()
        
        # If filename is not provided, resolve it using the quant_tag
        if not filename and quant_tag:
            repo_info = api.repo_info(repo_id)
            matching_files = []
            for sibling in repo_info.siblings:
                rfile = sibling.rfilename
                if rfile.endswith(".gguf") and not "mmproj" in rfile.lower():
                    if quant_tag.lower() in rfile.lower():
                        matching_files.append(rfile)
            
            if not matching_files:
                return False, f"No GGUF file matching quantization tag '{quant_tag}' found in repo '{repo_id}'.", None
            
            # Select the first matching file, preferring exact casing if present
            filename = matching_files[0]
            for mf in matching_files:
                if quant_tag in mf:
                    filename = mf
                    break

        if not filename:
            return False, "Filename or Quant Tag is required.", None

        target_dir = os.path.join(MODEL_DIR, repo_id)
        os.makedirs(target_dir, exist_ok=True)
        dest_model_path = os.path.join(target_dir, filename)

        if os.path.exists(dest_model_path) or os.path.islink(dest_model_path):
            logger.info(f"Model file {filename} already exists at {dest_model_path}. Skipping download.")
        else:
            logger.info(f"Downloading {filename} from {repo_id} to standard cache...")
            cached_model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename
            )
            if os.path.exists(dest_model_path) or os.path.islink(dest_model_path):
                os.remove(dest_model_path)
            os.symlink(cached_model_path, dest_model_path)

        mmproj_filename = None
        # Check if a companion mmproj file exists in the repository (priority: BF16 -> F16 -> F32)
        try:
            repo_info = api.repo_info(repo_id)
            sibling_names = [sibling.rfilename for sibling in repo_info.siblings]

            def find_sibling(target):
                for name in sibling_names:
                    if name == target or name.endswith("/" + target):
                        return name
                return None

            for target_opts in [("mmproj-BF16.gguf", "mmproj-bf16.gguf"),
                                 ("mmproj-F16.gguf", "mmproj-f16.gguf"),
                                 ("mmproj-F32.gguf", "mmproj-f32.gguf")]:
                for opt in target_opts:
                    found = find_sibling(opt)
                    if found:
                        mmproj_filename = found
                        break
                if mmproj_filename:
                    break

            if mmproj_filename:
                local_mmproj_path = os.path.join(target_dir, mmproj_filename)
                if os.path.exists(local_mmproj_path) or os.path.islink(local_mmproj_path):
                    logger.info(f"mmproj file {mmproj_filename} already exists at {local_mmproj_path}. Skipping download.")
                else:
                    logger.info(f"Downloading mmproj file {mmproj_filename} from {repo_id} to standard cache...")
                    cached_mmproj_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=mmproj_filename
                    )
                    if os.path.exists(local_mmproj_path) or os.path.islink(local_mmproj_path):
                        os.remove(local_mmproj_path)
                    os.symlink(cached_mmproj_path, local_mmproj_path)
                    logger.info(f"Downloaded and symlinked mmproj file {mmproj_filename} successfully.")
        except Exception as mm_err:
            logger.warning(f"Could not check or download mmproj file: {mm_err}")

        logger.info("Download complete.")
        return True, filename, mmproj_filename

    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return False, str(e), None


import datetime
import shlex

# Dict to store active/past background downloads
# Key: (node_id, repo_id, filename or quant_tag)
# Value: {"status": "Downloading" | "Completed" | "Failed", "message": str, "timestamp": str}
background_downloads = {}

def download_worker(node, repo_id, filename, quant_tag):
    key = (node["id"], repo_id, filename or quant_tag or "default")
    background_downloads[key] = {
        "status": "Downloading",
        "message": f"Downloading {repo_id} on {node['host']}...",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    try:
        if node["is_local"]:
            success, result, mmproj_filename = download_model_from_hf(repo_id, filename, quant_tag)
            if success:
                msg = f"Successfully downloaded {result}"
                if mmproj_filename:
                    msg += f" and {os.path.basename(mmproj_filename)}"
                background_downloads[key]["status"] = "Completed"
                background_downloads[key]["message"] = msg
                logger.info(f"Local download worker success: {msg}")
            else:
                background_downloads[key]["status"] = "Failed"
                background_downloads[key]["message"] = result
                logger.error(f"Local download worker failed: {result}")
        else:
            # Remote download using remote app.py CLI
            cmd = f'"$HOME"/LocalLLMManager/.venv/bin/python3 "$HOME"/LocalLLMManager/LocalLLMManager/app.py --repo-id {shlex.quote(repo_id)}'
            if filename:
                cmd += f' --filename {shlex.quote(filename)}'
            if quant_tag:
                cmd += f' --quant-tag {shlex.quote(quant_tag)}'
            if node.get("model_dir"):
                cmd += f' --model-dir {shlex.quote(node["model_dir"])}'
                
            logger.info(f"Triggering remote download on Node {node['id']} ({node['host']}): {cmd}")
            rc, stdout, stderr = run_ssh_command(node, cmd, timeout=900) # 15 minutes timeout
            
            if rc == 0:
                success_msg = "Download completed."
                for line in stdout.split('\n'):
                    if "SUCCESS:" in line:
                        success_msg = line.replace("SUCCESS:", "").strip()
                        break
                background_downloads[key]["status"] = "Completed"
                background_downloads[key]["message"] = success_msg
                logger.info(f"Remote download success on Node {node['id']}: {success_msg}")
            else:
                err_msg = stderr.strip() if stderr.strip() else stdout.strip()
                if not err_msg:
                    err_msg = f"Process exited with code {rc}"
                background_downloads[key]["status"] = "Failed"
                background_downloads[key]["message"] = err_msg
                logger.error(f"Remote download failed on Node {node['id']}: {err_msg}")
                
    except Exception as e:
        background_downloads[key]["status"] = "Failed"
        background_downloads[key]["message"] = str(e)
        logger.error(f"Download worker exception for Node {node['id']}: {e}")


@app.route('/download', methods=['POST'])
def download_model():
    """Handles downloading a model from Hugging Face on one or more nodes."""
    repo_id = request.form.get('repo_id')
    filename = request.form.get('filename')
    quant_tag = request.form.get('quant_tag')
    node_ids_str = request.form.getlist('node_ids')

    if not repo_id:
        flash("Repo ID is required.", "error")
        return redirect(url_for('index'))

    if not node_ids_str:
        flash("Please select at least one target node for the download.", "error")
        return redirect(url_for('index'))

    started_nodes = []
    for nid_str in node_ids_str:
        try:
            nid = int(nid_str)
            node = next((n for n in NODES if n["id"] == nid), None)
            if node and node.get("is_active"):
                thread = threading.Thread(
                    target=download_worker,
                    args=(node, repo_id, filename, quant_tag),
                    daemon=True
                )
                thread.start()
                started_nodes.append(node["host"])
        except ValueError:
            continue

    if started_nodes:
        flash(f"Download task(s) started in the background on: {', '.join(started_nodes)}. You can monitor the progress on the Model Manager page.", "success")
    else:
        flash("Failed to start downloads. Ensure selected nodes are active.", "error")

    return redirect(url_for('index'))


@app.route('/clear_downloads', methods=['POST'])
def clear_downloads():
    """Clears completed and failed background download tasks from the list."""
    keys_to_remove = [
        k for k, v in background_downloads.items() 
        if v["status"] in ("Completed", "Failed")
    ]
    for k in keys_to_remove:
        del background_downloads[k]
    flash("Cleared completed and failed download tasks.", "success")
    return redirect(url_for('index'))


@app.route('/load', methods=['POST'])
def load_model():
    """Handles loading a GGUF model by starting a llama-server subprocess."""
    model_file = request.form.get('model_file')
    model_alias = request.form.get('model_alias', '').strip()
    custom_args_str = request.form.get('custom_args', '')
    node_id_str = request.form.get('node_id', '0')

    if not model_file:
        flash("No model selected.", "error")
        return redirect(url_for('index'))

    try:
        node_id = int(node_id_str)
        node = next((n for n in NODES if n["id"] == node_id), None)
        if not node or not node.get("is_active"):
            flash("Selected node is inactive or invalid.", "error")
            return redirect(url_for('index'))
    except ValueError:
        flash("Invalid node selection.", "error")
        return redirect(url_for('index'))

    # Check if model exists on the target node
    if node["is_local"]:
        model_path = os.path.join(MODEL_DIR, model_file)
        if not os.path.exists(model_path):
            flash(f"Model file not found: {model_file}", "error")
            return redirect(url_for('index'))
    else:
        remote_model_path = os.path.join(node["model_dir"], model_file)
        check_file_cmd = f'[ -f "{remote_model_path}" ]'
        rc_f, _, _ = run_ssh_command(node, check_file_cmd)
        if rc_f != 0:
            flash(f"Model file not found on remote node {node['host']}: {model_file}", "error")
            return redirect(url_for('index'))

    # Forbidden flags check
    FORBIDDEN_FLAGS = {
        '-m', '--model', '-mu', '--model-url',
        '-dr', '--docker-repo',
        '-hf', '-hfr', '--hf-repo',
        '-hfd', '-hfrd', '--hf-repo-draft',
        '-hff', '--hf-file',
        '-hfv', '-hfrv', '--hf-repo-v',
        '-hffv', '--hf-file-v',
        '--log-disable', '--log-file',
        '--host', '--port', '--api-prefix', '--no-webui',
        '-a', '--alias'
    }

    custom_args = []
    if custom_args_str:
        try:
            custom_args = shlex.split(custom_args_str)
        except ValueError as e:
            logger.warning(f"Error parsing custom arguments: {e}")
            flash(f"Error parsing custom arguments: {e}", "error")
            return redirect(url_for('index'))

        for arg in custom_args:
            if arg in FORBIDDEN_FLAGS:
                flash(f"Forbidden argument: '{arg}'. This is managed by the app.", "error")
                return redirect(url_for('index'))
            if '=' in arg:
                flag_part = arg.split('=', 1)[0]
                if flag_part in FORBIDDEN_FLAGS:
                    flash(f"Forbidden argument: '{flag_part}'. This is managed by the app.", "error")
                    return redirect(url_for('index'))

    # Determine mmproj companion path
    mmproj_path = None
    if node["is_local"]:
        model_dir_path = os.path.dirname(model_path)
        if os.path.exists(os.path.join(model_dir_path, "mmproj-BF16.gguf")):
            mmproj_path = os.path.join(model_dir_path, "mmproj-BF16.gguf")
        elif os.path.exists(model_dir_path):
            for f in os.listdir(model_dir_path):
                if f.startswith("mmproj") and f.endswith(".gguf"):
                    mmproj_path = os.path.join(model_dir_path, f)
                    break
    else:
        model_dir_remote = os.path.dirname(remote_model_path)
        cmd_mmproj = (
            f'python3 -c "import os; '
            f'd=\'{model_dir_remote}\'; '
            f'print(next((f for f in os.listdir(d) if f.startswith(\'mmproj\') and f.endswith(\'.gguf\')), \'\') '
            f'if os.path.exists(d) else \'\')"'
        )
        rc_mm, stdout_mm, _ = run_ssh_command(node, cmd_mmproj)
        mmproj_filename = stdout_mm.strip()
        if mmproj_filename:
            mmproj_path = os.path.join(model_dir_remote, mmproj_filename)

    if mmproj_path and '--mmproj' not in custom_args:
        custom_args.extend(['--mmproj', mmproj_path])

    if '-c' not in custom_args and '--context-size' not in custom_args:
        custom_args.extend(['-c', "4096"])

    if '-ngl' not in custom_args and '--n-gpu-layers' not in custom_args:
        custom_args.extend(['-ngl', "-1"])

    alias = model_alias if model_alias else model_file
    custom_args.extend(["-a", alias])

    try:
        if node["is_local"]:
            remote_port = find_next_available_port(STARTING_PORT)
            host = "localhost"
            command = [
                LLAMA_SERVER_CMD,
                "-m", model_path,
                "--host", "0.0.0.0",
                "--port", str(remote_port)
            ]
            command.extend(custom_args)

            logger.info(f"Starting local server with command: {' '.join(command)}")

            create_new_group_flags = {}
            if os.name == 'posix':
                create_new_group_flags['preexec_fn'] = os.setsid
            elif os.name == 'nt':
                create_new_group_flags['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP

            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                **create_new_group_flags
            )

            sleep(10)

            return_code = proc.poll()
            if return_code is not None:
                stderr_output = proc.stderr.read()
                logger.error(f"Failed to start llama-server. Return code: {return_code}")
                logger.error(f"Stderr: {stderr_output}")
                flash(f"Failed to start llama-server. Check console for error: {stderr_output[:200]}...", "error")
                return redirect(url_for('index'))

            new_context = {
                "process": proc,
                "node_id": node["id"],
                "remote_pid": None,
                "name": model_file,
                "alias": alias,
                "port": remote_port,
                "host": host,
                "metrics_thread_stderr": None,
                "metrics_thread_stdout": None,
                "metrics": {"total_requests": 0, "last_tps": 0.0, "average_tps": 0.0}
            }

            # Start stderr reader (for metrics)
            stderr_thread = threading.Thread(
                target=drain_server_stderr,
                args=(proc, new_context["metrics"]),
                daemon=True
            )
            stderr_thread.start()

            # Start stdout reader
            stdout_thread = threading.Thread(
                target=drain_server_stdout,
                args=(proc,),
                daemon=True
            )
            stdout_thread.start()

            llm_server_contexts[remote_port] = new_context
            new_context["metrics_thread_stderr"] = stderr_thread
            new_context["metrics_thread_stdout"] = stdout_thread

            logger.info(f"llama-server started successfully on port {remote_port}. PID: {proc.pid}")
            flash(f"Successfully loaded {model_file} on port {remote_port}!", "success")

        else:
            # Remote Node Loading
            remote_port = find_remote_port(node)
            remote_bin = str(pathlib.Path(node["llama_bin_path"]).joinpath('llama-server'))
            log_file = f"~/LocalLLMManager/llama_server_{remote_port}.log"
            
            # Construct the remote llama-server run command
            run_cmd_parts = [
                remote_bin,
                "-m", shlex.quote(remote_model_path),
                "--host", "0.0.0.0",
                "--port", str(remote_port)
            ]
            run_cmd_parts.extend(custom_args)
            
            run_cmd = " ".join(run_cmd_parts) + f" > {log_file} 2>&1 & echo $!"
            logger.info(f"Launching remote llama-server on Node {node['id']} ({node['host']}): {run_cmd}")
            
            rc, stdout, stderr = run_ssh_command(node, run_cmd)
            if rc != 0:
                flash(f"Failed to start remote llama-server command: {stderr.strip()}", "error")
                return redirect(url_for('index'))
                
            try:
                remote_pid = int(stdout.strip())
            except ValueError:
                flash(f"Failed to parse remote process PID from output: {stdout}", "error")
                return redirect(url_for('index'))

            sleep(10)

            # Check if remote process is still running
            check_cmd = f"ps -p {remote_pid}"
            rc_check, _, _ = run_ssh_command(node, check_cmd)
            if rc_check != 0:
                # Read remote logs to diagnose failure
                rc_log, stdout_log, _ = run_ssh_command(node, f"tail -n 20 {log_file}")
                logger.error(f"Remote process {remote_pid} failed to start. Logs:\n{stdout_log}")
                flash(f"Failed to start remote llama-server. Error logs:\n{stdout_log[:300]}...", "error")
                return redirect(url_for('index'))

            # Start local streaming process for remote logs
            log_stream_cmd = [
                "ssh",
                "-i", node["keyfile"],
                "-o", "StrictHostKeyChecking=accept-new",
                f"{node['username']}@{node['host']}",
                f"tail -f -n +1 {log_file}"
            ]
            
            proc = subprocess.Popen(
                log_stream_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            new_context = {
                "process": proc, # Pipe process
                "node_id": node["id"],
                "remote_pid": remote_pid,
                "name": model_file,
                "alias": alias,
                "port": remote_port,
                "host": node["host"],
                "metrics_thread_stderr": None,
                "metrics_thread_stdout": None,
                "metrics": {"total_requests": 0, "last_tps": 0.0, "average_tps": 0.0}
            }

            # Drain standard error to capture metrics
            stderr_thread = threading.Thread(
                target=drain_server_stderr,
                args=(proc, new_context["metrics"]),
                daemon=True
            )
            stderr_thread.start()

            # Drain stdout
            stdout_thread = threading.Thread(
                target=drain_server_stdout,
                args=(proc,),
                daemon=True
            )
            stdout_thread.start()

            llm_server_contexts[remote_port] = new_context
            new_context["metrics_thread_stderr"] = stderr_thread
            new_context["metrics_thread_stdout"] = stdout_thread

            logger.info(f"Remote llama-server started successfully on Node {node['id']} ({node['host']}):{remote_port}. Remote PID: {remote_pid}")
            flash(f"Successfully loaded {model_file} on Node {node['id']} ({node['host']}) port {remote_port}!", "success")

        save_custom_args(model_file, custom_args_str)

    except Exception as e:
        logger.error(f"Error starting llama-server: {e}")
        logger.error(traceback.format_exc())
        flash(f"Error starting llama-server: {e}", "error")

    return redirect(url_for('index'))


@app.route('/unload', methods=['POST'])
def unload_model():
    """Handles unloading the currently active model by terminating the server process."""
    port_to_unload = request.form.get('port')
    if not port_to_unload:
        flash("Port not specified for unloading.", "error")
        return redirect(url_for('index'))

    try:
        port_to_unload = int(port_to_unload)
    except ValueError:
        flash("Invalid port specified.", "error")
        return redirect(url_for('index'))

    if port_to_unload not in llm_server_contexts:
        logger.warning(f"Attempted to unload port {port_to_unload}, but it was not found.")
        flash(f"Model on port {port_to_unload} not found (already unloaded?).", "success")
        return redirect(url_for('index'))

    context = llm_server_contexts[port_to_unload]
    proc = context["process"]
    model_name = context["name"]
    stderr_thread = context["metrics_thread_stderr"]
    stdout_thread = context["metrics_thread_stdout"]
    node_id = context.get("node_id", 0)
    remote_pid = context.get("remote_pid")

    node = next((n for n in NODES if n["id"] == node_id), None)

    # 1. Kill remote process if remote
    if node and not node["is_local"] and remote_pid:
        try:
            logger.info(f"Unloading remote node {node['host']} process {remote_pid}")
            kill_cmd = f"kill -9 {remote_pid}"
            run_ssh_command(node, kill_cmd)
        except Exception as e:
            logger.error(f"Failed to kill remote llama-server process {remote_pid}: {e}")

    # 2. Terminate the local process (either local server or local tail log stream)
    if proc:
        try:
            logger.info(f"Attempting to stop local subprocess/log stream {proc.pid} ({model_name})")
            if os.name == 'posix':
                if 'ssh' in proc.args[0]:
                    os.kill(proc.pid, signal.SIGTERM)
                else:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            elif os.name == 'nt':
                os.kill(proc.pid, signal.CTRL_BREAK_EVENT)

            try:
                proc.wait(timeout=5)
                logger.info(f"Subprocess {proc.pid} terminated gracefully.")
            except subprocess.TimeoutExpired:
                logger.warning(f"Subprocess {proc.pid} did not terminate, forcing kill")
                if os.name == 'posix':
                    if os.name == 'posix':
                        if 'ssh' in proc.args[0]:
                            os.kill(proc.pid, signal.SIGTERM)
                        else:
                            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                elif os.name == 'nt':
                    proc.kill()
                logger.info("Subprocess killed.")
        except Exception as e:
            logger.error(f"An error occurred while terminating process {proc.pid}: {e}")

    if stderr_thread:
        stderr_thread.join(timeout=2)
    if stdout_thread:
        stdout_thread.join(timeout=2)

    logger.info("Log threads joined.")
    flash(f"Successfully unloaded {model_name}.", "success")

    if port_to_unload in llm_server_contexts:
        del llm_server_contexts[port_to_unload]

    return redirect(url_for('index'))


@app.route('/api/metrics')
def api_metrics():
    """Provides the latest metrics for a specific model as JSON."""
    port_str = request.args.get('port')
    if not port_str:
        return jsonify({"error": "Port parameter is required"}), 400

    try:
        port = int(port_str)
        if port in llm_server_contexts:
            return jsonify(llm_server_contexts[port]["metrics"])
        else:
            return jsonify({"error": "Model for this port not found or unloaded"}), 404
    except ValueError:
        return jsonify({"error": "Invalid port format"}), 400
    except Exception as e:
        logger.error(f"Error fetching metrics for port {port_str}: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/api/node_models')
def api_node_models():
    """Returns a list of local models available on a specific node."""
    node_id_str = request.args.get('node_id')
    if not node_id_str:
        return jsonify({"error": "node_id is required"}), 400
    try:
        node_id = int(node_id_str)
        node = next((n for n in NODES if n["id"] == node_id), None)
        if not node or not node.get("is_active"):
            return jsonify({"error": "Node not found or inactive"}), 404
            
        if node["is_local"]:
            return jsonify(get_local_models())
        else:
            cmd = (
                f'"$HOME"/LocalLLMManager/.venv/bin/python3 -c "'
                f'import sys, json; sys.path.append(\'$HOME/LocalLLMManager/LocalLLMManager\'); '
                f'import app; app.MODEL_DIR=\'{node["model_dir"]}\'; '
                f'print(json.dumps(app.get_local_models()))"'
            )
            rc, stdout, stderr = run_ssh_command(node, cmd)
            if rc != 0:
                return jsonify({"error": f"Failed to list remote models: {stderr.strip()}"}), 500
            return Response(stdout.strip(), mimetype='application/json')
    except ValueError:
        return jsonify({"error": "Invalid node_id"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/llama_help')
def api_llama_help():
    """Runs 'llama-server --help' and returns the output."""
    try:
        result = subprocess.run(
            [LLAMA_SERVER_CMD, "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )

        # Check if the command ran successfully
        if result.returncode == 0:
            return jsonify({"help_text": result.stdout})
        else:
            # If the command fails (e.g., --help isn't a valid flag)
            logger.error(f"llama-server --help failed: {result.stderr}")
            return jsonify({"help_text": f"Error running command:\n{result.stderr}"}), 500

    except FileNotFoundError:
        # This error happens if 'llama-server' isn't in the system's PATH
        logger.error("'llama-server' command not found.")
        return jsonify({"help_text": "Error: 'llama-server' command not found. Is it installed and in your PATH?"}), 500
    except Exception as e:
        # Catch any other potential errors
        logger.error(f"Error getting llama-server help: {e}")
        return jsonify({"help_text": f"An unknown error occurred: {e}"}), 500


@app.route('/api/node_devices')
def api_node_devices():
    """Returns a list of devices visible to llama-server on a specific node."""
    node_id_str = request.args.get('node_id', '0')
    try:
        devices = get_node_devices(node_id_str)
        return jsonify({"node_id": node_id_str, "devices": devices})
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def update_nodes_env(remote_nodes):
    """
    Rewrites the remote node environment variables in .env.
    remote_nodes is a list of node dicts (excluding local node 0).
    """
    env_path = os.path.join(APP_DIR, '.env')
    
    hostnames = [n["host"] for n in remote_nodes]
    usernames = [n["username"] for n in remote_nodes]
    keyfiles = [n["keyfile"] for n in remote_nodes]
    model_dirs = [n["model_dir"] for n in remote_nodes]
    llama_bin_paths = [n["llama_bin_path"] for n in remote_nodes]
    
    new_vars = {
        "NODE_HOSTNAMES": shlex.join(hostnames),
        "NODE_USERNAMES": shlex.join(usernames),
        "NODE_KEYFILES": shlex.join(keyfiles),
        "REMOTE_MODEL_DIRS": shlex.join(model_dirs),
        "REMOTE_LLAMA_BIN_PATHS": shlex.join(llama_bin_paths)
    }
    
    lines = []
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            lines = f.readlines()
            
    keys_updated = set()
    new_lines = []
    
    for line in lines:
        line_stripped = line.strip()
        matched_key = None
        for key in new_vars:
            if line_stripped.startswith(f"{key}=") or line_stripped.startswith(f"# {key}="):
                matched_key = key
                break
        if matched_key:
            new_lines.append(f"{matched_key}={new_vars[matched_key]}\n")
            keys_updated.add(matched_key)
        else:
            new_lines.append(line)
            
    for key, val in new_vars.items():
        if key not in keys_updated:
            new_lines.append(f"{key}={val}\n")
            
    with open(env_path, 'w') as f:
        f.writelines(new_lines)
        
    load_dotenv(env_path, override=True)


@app.route('/nodes')
def manage_nodes():
    """Renders the node management page."""
    for node in NODES:
        if "is_active" not in node:
            is_valid, msg = validate_node(node)
            node["is_active"] = is_valid
            node["status_msg"] = msg
    return render_template('nodes.html', nodes=NODES)


@app.route('/nodes/add', methods=['POST'])
def add_node_route():
    global NODES
    host = request.form.get('host', '').strip()
    username = request.form.get('username', '').strip()
    keyfile = request.form.get('keyfile', '').strip()
    model_dir = request.form.get('model_dir', '').strip()
    llama_bin_path = request.form.get('llama_bin_path', '').strip()

    if not host or not username or not keyfile or not model_dir or not llama_bin_path:
        flash("All 5 fields (Hostname, Username, SSH Key File, Model Directory, Llama Bin Path) are required.", "error")
        return redirect(url_for('manage_nodes'))

    current_nodes = load_nodes_config()
    remote_nodes = [n for n in current_nodes if not n["is_local"]]

    if any(n["host"] == host for n in remote_nodes):
        flash(f"Node with host '{host}' already exists.", "error")
        return redirect(url_for('manage_nodes'))

    new_node = {
        "id": len(current_nodes),
        "host": host,
        "username": username,
        "keyfile": keyfile,
        "model_dir": model_dir,
        "llama_bin_path": llama_bin_path,
        "is_local": False
    }

    is_valid, status_msg = validate_node(new_node)
    if not is_valid:
        flash(f"Failed to add node '{host}': {status_msg}", "error")
        return redirect(url_for('manage_nodes'))

    remote_nodes.append(new_node)
    update_nodes_env(remote_nodes)

    NODES = load_nodes_config()
    for n in NODES:
        is_v, m = validate_node(n)
        n["is_active"] = is_v
        n["status_msg"] = m

    flash(f"Successfully added node '{host}'! Status: {status_msg}", "success")
    return redirect(url_for('manage_nodes'))


@app.route('/nodes/remove', methods=['POST'])
def remove_node_route():
    global NODES
    node_id_str = request.form.get('node_id', '')
    try:
        node_id = int(node_id_str)
    except ValueError:
        flash("Invalid node ID.", "error")
        return redirect(url_for('manage_nodes'))

    current_nodes = load_nodes_config()
    target_node = next((n for n in current_nodes if n["id"] == node_id), None)
    if not target_node:
        flash("Node not found.", "error")
        return redirect(url_for('manage_nodes'))

    if target_node["is_local"]:
        flash("Cannot remove the local node (Node 0).", "error")
        return redirect(url_for('manage_nodes'))

    remote_nodes = [n for n in current_nodes if not n["is_local"] and n["id"] != node_id]
    update_nodes_env(remote_nodes)

    NODES = load_nodes_config()
    for n in NODES:
        is_v, m = validate_node(n)
        n["is_active"] = is_v
        n["status_msg"] = m

    flash(f"Successfully removed node {target_node['id']} ({target_node['host']}).", "success")
    return redirect(url_for('manage_nodes'))


@app.route('/api/nodes/test_connection', methods=['POST'])
def api_test_node_connection():
    data = request.get_json() or {}
    host = data.get('host', '').strip()
    username = data.get('username', '').strip()
    keyfile = data.get('keyfile', '').strip()
    
    if not host or not username or not keyfile:
        return jsonify({"success": False, "error": "Hostname, Username, and SSH Key File are required."}), 400
        
    temp_node = {
        "is_local": False,
        "host": host,
        "username": username,
        "keyfile": keyfile
    }
    
    rc, stdout, stderr = run_ssh_command(temp_node, "echo 'ping'", timeout=10)
    if rc == 0:
        return jsonify({"success": True, "message": f"Successfully connected to {username}@{host} via SSH!"})
    else:
        return jsonify({"success": False, "error": f"SSH connection failed: {stderr.strip() or stdout.strip()}"})


@app.route('/api/nodes_devices')
def api_nodes_devices():
    """Returns a list of all configured nodes and their compute/acceleration devices."""
    nodes_info = []
    nodes = load_nodes_config()
    for n in nodes:
        node_entry = {
            "id": n["id"],
            "host": n["host"],
            "is_local": n["is_local"],
            "is_active": True,
            "status_msg": "Active",
            "devices": []
        }
        try:
            node_entry["devices"] = get_node_devices(n)
        except Exception as e:
            node_entry["is_active"] = False
            node_entry["status_msg"] = str(e)
            node_entry["devices"] = []
        nodes_info.append(node_entry)
        
    return jsonify({"nodes": nodes_info})


@app.route('/api/estimate_vram')
def estimate_vram_route():
    filename = request.args.get('filename')
    if not filename:
        return jsonify({"error": "No filename provided"}), 400

    model_path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(model_path):
        return jsonify({"error": "Model file not found"}), 404

    # Run estimator:
    # We request 16384 context size.
    try:
        data = run_estimator(model_path, [16384], 2)
        print(f"Estimated VRAM: {data}")
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error estimating VRAM: {e}")
        return jsonify({"error": str(e)}), 500


@proxy_app.route('/v1/models', methods=['GET'])
def proxy_models():
    """
    Returns a list of all currently loaded models in OpenAI format.
    """
    model_data = []
    for port, context in llm_server_contexts.items():
        model_data.append({
            "id": context['alias'],  # Use filename as the model ID
            "object": "model",
            "created": int(os.path.getmtime(os.path.join(MODEL_DIR, context['name']))) if os.path.exists(
                os.path.join(MODEL_DIR, context['name'])) else int(time.time()),
            "owned_by": "user"
        })

    return jsonify({
        "object": "list",
        "data": model_data
    })


@proxy_app.route('/v1/chat/completions', methods=['POST'])
def proxy_chat_completions():
    """
    Receives an OpenAI-formatted request and forwards it (unmodified)
    to the correct llama-server backend.
    """
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        return jsonify({"error": "Authorization header is missing"}), 401

    try:
        data = request.json
        model_name = data.get('model')
        if not model_name:
            return jsonify({"error": "model field is required"}), 400
    except Exception:
        return jsonify({"error": "Invalid JSON body"}), 400

    context = get_context_by_model_alias(model_name)
    if not context:
        return jsonify({"error": f"Model '{model_name}' is not loaded."}), 404

    target_host = context.get('host', '127.0.0.1')
    if target_host == 'localhost':
        target_host = '127.0.0.1'
    target_port = context.get('port', 8100)
    target_url = f"http://{target_host}:{target_port}/v1/chat/completions"

    raw_data = request.get_data()

    forward_headers = {
        'Content-Type': request.headers.get('Content-Type', 'application/json'),
        'Authorization': auth_header,
        'Accept': request.headers.get('Accept', 'text/event-stream')
    }

    is_stream = data.get('stream', False)

    try:
        if is_stream:
            proxy_response = requests.post(
                target_url,
                data=raw_data,
                headers=forward_headers,
                stream=True
            )
            proxy_response.raise_for_status()

            return Response(
                proxy_response.iter_content(chunk_size=8192),
                status=proxy_response.status_code,
                content_type=proxy_response.headers.get('Content-Type')
            )
        else:
            proxy_response = requests.post(
                target_url,
                data=raw_data,
                headers=forward_headers,
                stream=False
            )
            proxy_response.raise_for_status()

            return jsonify(proxy_response.json()), proxy_response.status_code

    except requests.exceptions.RequestException as e:
        logger.error(f"Proxy request failed: {e}")
        return jsonify({"error": f"Failed to connect to backend model: {e}"}), 502
    except Exception as e:
        logger.error(f"Proxy error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"An internal error occurred: {e}"}), 500


if __name__ == '__main__':
    default_host = os.environ.get('APP_HOST', '0.0.0.0')
    default_port = int(os.environ.get('APP_PORT', 5001))
    default_debug = os.environ.get('APP_DEBUG', 'false').lower() in ('true', '1', 'yes')
    default_verbose = os.environ.get('APP_VERBOSE', 'false').lower() in ('true', '1', 'yes')
    default_proxy_host = os.environ.get('PROXY_HOST', '0.0.0.0')
    default_proxy_port = int(os.environ.get('PROXY_PORT', 8080))
    default_model_dir = os.environ.get('MODEL_DIR', os.path.join(APP_DIR, "models"))

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run the LLM Manager Flask app.")
    parser.add_argument(
        '--host',
        type=str,
        default=default_host,
        help="The host to bind the server to. (Default: 0.0.0.0)"
    )
    parser.add_argument(
        '--port',
        type=int,
        default=default_port,
        help="The port to run the server on. (Default: 5001)"
    )
    parser.add_argument(
        '--debug',
        action=argparse.BooleanOptionalAction,
        default=default_debug,
        help="Run in debug mode. (Default: False)"
    )
    parser.add_argument(
        '-v', '--verbose',
        action=argparse.BooleanOptionalAction,
        default=default_verbose,
        help="Enable verbose logging (DEBUG level). (Default: INFO level)"
    )
    parser.add_argument(
        '--print-model-dir-path',
        action='store_true',
        help="Print the absolute path to the model storage directory and exit."
    )
    parser.add_argument(
        '--print-app-path',
        action='store_true',
        help="Print the absolute path to the app directory and exit."
    )
    parser.add_argument(
        '--download-repo-id', '--repo-id',
        dest='download_repo_id',
        type=str,
        help="Hugging Face repository ID to download a GGUF model from, then exit."
    )
    parser.add_argument(
        '--download-file', '--filename',
        dest='download_file',
        type=str,
        help="GGUF filename to download (used with --download-repo-id)."
    )
    parser.add_argument(
        '--download-quant-tag', '--quant-tag',
        dest='download_quant_tag',
        type=str,
        help="Quantization tag to filter and download (used with --download-repo-id if filename not specified)."
    )
    parser.add_argument(
        '--proxy-host', type=str, default=default_proxy_host,
        help=f"The host for the OpenAI Proxy. (Env: PROXY_HOST, Default: {default_proxy_host})"
    )
    parser.add_argument(
        '--proxy-port', type=int, default=default_proxy_port,
        help=f"The port for the OpenAI Proxy. (Env: PROXY_PORT, Default: {default_proxy_port})"
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default=default_model_dir,
        help=f"Directory to store models. (Env: MODEL_DIR, Default: {default_model_dir})"
    )
    parser.add_argument(
        '--use-local-node',
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Specify whether the app can use the local instance as a node. (Default: True, or USE_LOCAL_NODE env var)"
    )
    args = parser.parse_args()

    MODEL_DIR = os.path.abspath(args.model_dir)

    # Create the directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)

    NODES = load_nodes_config(args.use_local_node)

    if args.print_model_dir_path:
        print(os.path.abspath(MODEL_DIR))
        sys.exit(0)

    if args.print_app_path:
        print(os.path.abspath(APP_DIR))
        sys.exit(0)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    if not args.verbose:
        logging.getLogger('werkzeug').setLevel(logging.WARNING)

    logger.info(f"Loaded {len(NODES)} nodes configuration. Validating node connections...")
    for node in NODES:
        is_valid, msg = validate_node(node)
        node["is_active"] = is_valid
        node["status_msg"] = msg
        if is_valid:
            logger.info(f"  Node {node['id']} ({node['host']}): ACTIVE - {msg}")
        else:
            logger.error(f"  Node {node['id']} ({node['host']}): OFFLINE - {msg}")

    if args.download_repo_id:
        success, result, mmproj_filename = download_model_from_hf(
            repo_id=args.download_repo_id,
            filename=args.download_file,
            quant_tag=args.download_quant_tag
        )
        if success:
            if mmproj_filename:
                print(f"SUCCESS: Successfully downloaded {result} and {os.path.basename(mmproj_filename)}!")
            else:
                print(f"SUCCESS: Successfully downloaded {result}!")
            sys.exit(0)
        else:
            print(f"ERROR: {result}", file=sys.stderr)
            sys.exit(1)

    init_db()

    app.config['PROXY_HOST'] = args.proxy_host
    app.config['PROXY_PORT'] = args.proxy_port
    def run_proxy_server():
        try:
            logger.info(f"Starting OpenAI proxy server on http://{args.proxy_host}:{args.proxy_port}")
            proxy_app.run(host=args.proxy_host, port=args.proxy_port)
        except Exception as e:
            logger.error(f"Proxy server failed to start: {e}")
            logger.error(traceback.format_exc())

    proxy_thread = threading.Thread(target=run_proxy_server, daemon=True)
    proxy_thread.start()

    sleep(1)

    print(f"Starting Flask app on http://{args.host}:{args.port}")
    if args.debug:
        print("Running in DEBUG mode. Do not use in production.")

    app.run(debug=args.debug, host=args.host, port=args.port)