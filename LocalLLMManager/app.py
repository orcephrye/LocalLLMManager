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
            "# MODEL_DIR=\n"
        )
        try:
            with open(env_path, 'w') as f:
                f.write(default_env_content)
            print(f"INFO: No .env file found. Created a default .env file at {env_path}")
        except Exception as e:
            print(f"WARNING: Failed to create default .env file at {env_path}: {e}")


check_default_env()
load_dotenv()

LLAMA_BIN_PATH = os.getenv("LLAMA_BIN_PATH", "")
LLAMA_SERVER_CMD = str(pathlib.Path(LLAMA_BIN_PATH).joinpath('llama-server'))


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
        models = api.list_models(filter="gguf", apps="llama.cpp", sort="downloads", direction=-1, limit=500)

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
            direction=-1,
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
    """Finds all .gguf files in the MODEL_DIR."""
    models = []
    for filename in os.listdir(MODEL_DIR):
        if filename.endswith(".gguf"):
            if MULTI_PART_REGEX.findall(filename):
                if '00001-of-' in filename:
                    models.append(filename)
                continue
            models.append(filename)
    return models


# --- Flask Routes ---
@app.route('/')
def index():
    """Main page - displays controls and output."""

    local_models = get_local_models()
    all_custom_args = get_all_custom_args()
    all_custom_args_json = json.dumps(all_custom_args)

    proxy_host = app.config.get('PROXY_HOST', 'N/A')
    proxy_port = app.config.get('PROXY_PORT', 'N/A')
    display_host = "127.0.0.1" if proxy_host == "0.0.0.0" else proxy_host

    return render_template(
        'index.html',
        models=local_models,
        contexts=llm_server_contexts,
        all_custom_args_json=all_custom_args_json,
        proxy_host=proxy_host,
        display_host=display_host,
        proxy_port=proxy_port
    )


@app.route('/browse')
def browse_models():
    """Displays the model database search/browse page."""
    search_query = request.args.get('q', '')
    models = search_models_db(search_query)

    return render_template(
        'browse.html',
        models=models,
        search_query=search_query
    )


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
        search_query=search_query
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
        gguf_files=gguf_files
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


@app.route('/download', methods=['POST'])
def download_model():
    """Handles downloading a model from Hugging Face."""
    repo_id = request.form.get('repo_id')
    filename = request.form.get('filename')

    if not repo_id or not filename:
        flash("Repo ID and Filename are required.", "error")
        return redirect(url_for('index'))

    try:
        logger.info(f"Downloading {filename} from {repo_id}...")
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False  # Good for Windows compatibility
        )
        logger.info("Download complete.")
        flash(f"Successfully downloaded {filename}!", "success")
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        flash(f"Error downloading model: {e}", "error")

    referrer = request.referrer
    if referrer:
        return redirect(referrer)
    else:
        return redirect(url_for('index'))


@app.route('/load', methods=['POST'])
def load_model():
    """Handles loading a GGUF model by starting a llama-server subprocess."""
    model_file = request.form.get('model_file')
    custom_args_str = request.form.get('custom_args', '')

    if not model_file:
        flash("No model selected.", "error")
        return redirect(url_for('index'))

    model_path = os.path.join(MODEL_DIR, model_file)

    if not os.path.exists(model_path):
        flash(f"Model file not found: {model_file}", "error")
        return redirect(url_for('index'))

    FORBIDDEN_FLAGS = {
        '-m', '--model', '-mu', '--model-url',
        '-dr', '--docker-repo',
        '-hf', '-hfr', '--hf-repo',
        '-hfd', '-hfrd', '--hf-repo-draft',
        '-hff', '--hf-file',
        '-hfv', '-hfrv', '--hf-repo-v',
        '-hffv', '--hf-file-v',
        '--log-disable', '--log-file',
        '--host', '--port', '--api-prefix', '--no-webui'
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

    try:
        port = find_next_available_port(STARTING_PORT)
        command = [
            LLAMA_SERVER_CMD,
            "-m", model_path,
            "--port", str(port)
        ]

        if '-c' not in custom_args and '--context-size' not in custom_args:
            custom_args.extend(['-c', "4096"])

        if '-ngl' not in custom_args and '--n-gpu-layers' not in custom_args:
            custom_args.extend(['-ngl', "-1"])

        command.extend(custom_args)

        logger.info(f"Starting server with command: {' '.join(command)}")

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
            "name": model_file,
            "port": port,
            "metrics_thread_stderr": None,
            "metrics_thread_stdout": None,
            "metrics": {"total_requests": 0, "last_tps": 0.0, "average_tps": 0.0}
        }

        save_custom_args(model_file, custom_args_str)

        # Start stderr reader (for metrics)
        stderr_thread = threading.Thread(
            target=drain_server_stderr,
            args=(proc, new_context["metrics"]),
            daemon=True
        )
        stderr_thread.start()

        # Start stdout reader (to drain buffer)
        stdout_thread = threading.Thread(
            target=drain_server_stdout,
            args=(proc,),
            daemon=True
        )
        stdout_thread.start()

        llm_server_contexts[port] = new_context
        new_context["metrics_thread_stderr"] = stderr_thread
        new_context["metrics_thread_stdout"] = stdout_thread

        logger.info(f"llama-server started successfully on port {port}. PID: {proc.pid}")
        flash(f"Successfully loaded {model_file} on port {port}!", "success")

    except Exception as e:
        logger.error(f"Error starting llama-server: {e}")
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

    if proc and proc.poll() is None:
        try:
            logger.info(f"Attempting to stop server process {proc.pid} ({model_name})")

            if os.name == 'posix':
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            elif os.name == 'nt':
                os.kill(proc.pid, signal.CTRL_BREAK_EVENT)

            try:
                proc.wait(timeout=5)
                logger.info(f"Process {proc.pid} terminated gracefully.")
            except subprocess.TimeoutExpired:
                logger.warning(f"Process {proc.pid} did not terminate, forcing kill")
                if os.name == 'posix':
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                elif os.name == 'nt':
                    proc.kill()
                logger.info("Process killed.")

            if stderr_thread:
                stderr_thread.join(timeout=2)
            if stdout_thread:
                stdout_thread.join(timeout=2)

            logger.info("Log threads joined.")
            flash(f"Successfully unloaded {model_name}.", "success")
        except Exception as e:
            logger.error(f"An error occurred while unloading model {model_name}: {e}")
            flash(f"Error unloading model: {e}", "error")
    else:
        logger.info("No model server process to unload.")
        flash("No model is currently loaded.", "success")

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
            "id": context['name'],  # Use filename as the model ID
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

    context = get_context_by_model_name(model_name)
    if not context:
        return jsonify({"error": f"Model '{model_name}' is not loaded."}), 404

    target_port = context['port']
    target_url = f"http://127.0.0.1:{target_port}/v1/chat/completions"

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
    args = parser.parse_args()

    MODEL_DIR = os.path.abspath(args.model_dir)

    # Create the directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)

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