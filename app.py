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
import uuid
import requests
import traceback
import time
from time import sleep
from dotenv import load_dotenv
from flask import Flask, render_template_string, request, redirect, url_for, flash, jsonify, Response
from flask_cors import CORS
from huggingface_hub import hf_hub_download, HfApi

# --- Configuration ---
logger = logging.getLogger(__name__)

# Set the folder where models will be stored
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
# Set the path for the local model database
DATABASE_FILE = os.path.join(os.path.dirname(__file__), "models.db")
# Create the directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "123qazabcwsx098"  # Needed for flashing messages


# Initialize Proxy App
proxy_app = Flask(__name__)
CORS(proxy_app)


llm_server_contexts = {}


STARTING_PORT = 8100


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

# --- Database Helper Functions ---

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

        # --- ADD THIS BLOCK ---
        c.execute("""
        CREATE TABLE IF NOT EXISTS model_args (
            model_filename TEXT PRIMARY KEY,
            custom_args TEXT
        )
        """)
        # --- END BLOCK ---

        conn.commit()
        conn.close()
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")


def parse_quantization(filename):
    """Parses the quantization type from a GGUF filename using regex."""
    # This regex looks for common GGUF quantizations like Q4_K_M, Q8_0, F16, etc.
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

    # We scan the top 1000 most downloaded GGUF-tagged models.
    # A full scan would take too long.
    try:
        fetch_failure_count = 0
        models = api.list_models(filter="gguf", apps="llama.cpp", sort="downloads", direction=-1, limit=500)

        for model in models:
            repo_id = model.modelId
            logger.debug(f"Scanning repo: {repo_id}")
            try:
                # Get file info for the repo
                repo_info = api.repo_info(repo_id, files_metadata=True)
                if repo_info.siblings:
                    for sibling in repo_info.siblings:
                        filename = sibling.rfilename
                        if filename.endswith(".gguf"):
                            quant = parse_quantization(filename)
                            # Insert or ignore if it already exists (due to UNIQUE constraint)
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
        # Re-raise the exception to be caught by the Flask route
        raise e


def search_models_db(query):
    """Searches the local SQLite database for models."""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        conn.row_factory = sqlite3.Row  # This lets us access columns by name
        c = conn.cursor()

        if not query:
            # If no query, show the 100 most recently added models
            c.execute("SELECT * FROM gguf_models ORDER BY id DESC LIMIT 100")
        else:
            # Simple search against repo_id, filename, and quantization
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
        # Ensure the table exists first (in case init_db hasn't run yet)
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
        # Use the list_models API with filters
        models = api.list_models(
            search=query,
            filter="gguf",
            apps="llama.cpp",
            sort="downloads",
            direction=-1,
            limit=200 # Limit results to avoid overwhelming the page/API
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


# --- Helper Function ---

def get_local_models():
    """Finds all .gguf files in the MODEL_DIR."""
    models = []
    for filename in os.listdir(MODEL_DIR):
        if filename.endswith(".gguf"):
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

    # Simple HTML template with CSS for styling
    # This is a single-file app, so we embed the HTML here.
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Local LLM Manager </title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                margin: 0; padding: 0; background-color: #f4f7f6; display: grid;
                place-items: center; min-height: 100vh;
            }
            .container {
                width: 90%; max-width: 800px; background: #ffffff;
                border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.07);
                padding: 2.5rem; margin: 2rem 0;
            }
            h1 { color: #2c3e50; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }
            h2 { color: #34495e; margin-top: 1.5rem; }
            .card { background: #ecf0f1; border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem; }
            label { display: block; font-weight: 600; margin-bottom: 8px; color: #555; }
            input[type="text"], select, textarea {
                width: 100%; padding: 12px; border: 1px solid #bdc3c7;
                border-radius: 6px; box-sizing: border-box; margin-bottom: 1rem;
            }
            textarea { min-height: 120px; resize: vertical; }
            button {
                background-color: #3498db; color: white; padding: 12px 20px;
                border: none; border-radius: 6px; cursor: pointer; font-size: 16px;
                transition: background-color 0.2s;
            }
            button:hover { background-color: #2980b9; }
            .status-bar {
                background: #2c3e50; color: white; padding: 1rem;
                border-radius: 6px; margin-bottom: 1.5rem;=
            }
            .status-item {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 10px 0;
                border-bottom: 1px solid #34495e;
            }
            .status-item:last-child {
                border-bottom: none;
            }
            .status-bar small { color: #bdc3c7; }
            .button-small {
                padding: 6px 12px;
                font-size: 14px;
            }
            .button-unload {
                background-color: #e67e22;
            }
            .button-unload:hover {
                background-color: #d35400;
            }
            .output {
                background: #fafafa; border: 1px solid #eee; border-radius: 6px;
                padding: 1.5rem; min-height: 100px; white-space: pre-wrap;
                word-wrap: break-word; font-family: "Courier New", Courier, monospace;
            }
            .flash {
                padding: 1rem; margin-bottom: 1rem; border-radius: 6px;
                font-weight: 600;
            }
            .flash.success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .flash.error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
            .nav-link {
                display: inline-block;
                background: #ecf0f1;
                color: #34495e;
                padding: 10px 15px;
                border-radius: 6px;
                text-decoration: none;
                font-weight: 600;
                margin-bottom: 1.5rem;
                transition: background-color 0.2s;
            }
            .nav-link:hover { background: #dfe6e9; }
            /* Basic Modal Styling */
            .modal {
                display: none; /* Hidden by default */
                position: fixed; /* Stay in place */
                z-index: 1; /* Sit on top */
                left: 0;
                top: 0;
                width: 100%; /* Full width */
                height: 100%; /* Full height */
                overflow: auto; /* Enable scroll if needed */
                background-color: rgba(0,0,0,0.4); /* Black w/ opacity */
            }
            .modal-content {
                background-color: #fefefe;
                margin: 15% auto; /* 15% from the top and centered */
                padding: 20px;
                border: 1px solid #888;
                width: 90%; /* Could be more or less, depending on screen size */
                border-radius: 10px;
                text-align: center;
                font-size: 1.2em;
            }
            .close {
                color: #aaa;
                float: right;
                font-size: 28px;
                font-weight: bold;
            }
            .close:hover,
            .close:focus {
                color: black;
                text-decoration: none;
                cursor: pointer;
            }

            /* --- Style for the help modal content --- */
            #helpModal .modal-content {
                width: 90%;            /* Use 90% of the browser width */
                max-width: 1200px;     /* Set a reasonable max width */
                margin: 5% auto;       /* Reduce top margin */
                text-align: left;
            }
            #helpModalContent {
                background: #f4f4f4;
                border: 1px solid #ddd;
                padding: 10px;
                border-radius: 5px;
                max-height: 60vh; /* Limit height and allow scroll */
                overflow-y: auto;
                white-space: pre-wrap; /* Preserve formatting */
                word-wrap: break-word;
                text-align: left; /* Align help text left */
                font-family: "Courier New", Courier, monospace;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Local LLM Manager</h1>

            {% with messages = get_flashed_messages(with_categories=true) %}
              {% if messages %}
                {% for category, message in messages %}
                  <div class="flash {{ category }}">{{ message }}</div>
                {% endfor %}
              {% endif %}
            {% endwith %}

            <div class_name="card">
                <h2>OpenAI API Proxy</h2>
                {% if proxy_host != 'N/A' %}
                    <p>Proxy is running and accessible at:</p>
                    <p><strong>API Base:</strong> <code>http://{{ display_host }}:{{ proxy_port }}/v1</code></p>
                    <p><strong>API Key:</strong> (Use any non-empty string)</p>
                {% else %}
                    <p>Proxy server is not configured or failed to start.</p>
                {% endif %}
            </div>

            <h2>Loaded Models</h2>
            <div class="status-bar">
                {% if contexts %}
                    {% for port, context in contexts.items() %}
                    <div class="status-item">
                        <div>
                            <strong>{{ context.name }}</strong>
                            <br><small>(Running on: 127.0.0.1:{{ port }})</small>
                        </div>
                        <form action="/unload" method="POST" style="margin: 0;">
                            <input type="hidden" name="port" value="{{ port }}">
                            <button type="submit" class="button-small button-unload">Unload</button>
                        </form>
                    </div>
                    {% endfor %}
                {% else %}
                    <p style="text-align: center; margin: 0; padding: 10px 0;">No models are currently loaded.</p>
                {% endif %}
            </div>

            <a href="/browse" class="nav-link">Browse Local Cache &raquo;</a>
            <a href="/search_hf" class="nav-link">Search Hugging Face &raquo;</a>
            <a href="/status" class="nav-link">Server Status &raquo;</a>


            <div class="card">
                <h2>1. Download Model</h2>
                <form action="/download" method="POST" class="download-form">
                    <label for="repo_id">Hugging Face Repo ID:</label>
                    <input type="text" id="repo_id" name="repo_id" placeholder="e.g., TheBloke/Llama-3-8B-Instruct-GGUF">
                    <label for="filename">GGUF Filename:</label>
                    <input type="text" id="filename" name="filename" placeholder="e.g., llama-3-8b-instruct.Q4_K_M.gguf">
                    <button type="submit">Download</button>
                </form>
            </div>

            <div class="card">
                <h2>2. Load Model</h2>
                <form action="/load" method="POST">
                    <label for="model_file">Select a downloaded model:</label>
                    <select id="model_file" name="model_file">
                        {% for model in models %}
                            <option value="{{ model }}">{{ model }}</option>
                        {% else %}
                            <option value="" disabled>No models found in 'models' folder</option>
                        {% endfor %}
                    </select>

                    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 1rem;">
                        <label for="custom_args" style="margin-bottom: 0;">Custom llama-server arguments:</label>
                        <button type="button" id="showHelpBtn" class="button-small" style="background-color: #7f8c8d; border: none;">Show Help</button>
                    </div>
                    <input type="text" id="custom_args" name="custom_args" placeholder="e.g., --temp 0.7 --top-k 40">
                    <button type="submit">Load Model</button>
                </form>
            </div>

        </div>

        <div id="loadingModal" class="modal">
            <div class="modal-content">
                <span class="close">&times;</span>
                <p>Downloading model... Please wait.</p>
            </div>
        </div>

        <div id="helpModal" class="modal">
            <div class="modal-content" style="text-align: left;">
                <span class="close-help">&times;</span>
                <h2>llama-server Help</h2>
                <pre id="helpModalContent">Loading...</pre>
            </div>
        </div>
        <script>
            const allArgs = {{ all_custom_args_json | safe }};
            const modelSelect = document.getElementById('model_file');
            const customArgsInput = document.getElementById('custom_args');

            function updateCustomArgs() {
                if (!modelSelect) return; // Guard against no models
                const selectedModel = modelSelect.value;
                if (allArgs.hasOwnProperty(selectedModel)) {
                    customArgsInput.value = allArgs[selectedModel];
                } else {
                    customArgsInput.value = ''; // Clear it if no args are saved
                }
            }
            if (modelSelect) {
                modelSelect.addEventListener('change', updateCustomArgs);
                
                // 5. Run on page load for the default selected model
                updateCustomArgs();
            }
            
            // Get the modals
            var loadingModal = document.getElementById("loadingModal");
            var helpModal = document.getElementById("helpModal");

            // Get the <span> elements that close the modals
            var loadingSpan = document.getElementsByClassName("close")[0];
            var helpSpan = document.getElementsByClassName("close-help")[0];

            // --- Download/Loading Modal Logic ---
            var downloadForms = document.querySelectorAll(".download-form");
            // When the user clicks on a download button, open the modal
            downloadForms.forEach(function(form) {
                form.addEventListener("submit", function(event) {
                    loadingModal.style.display = "block";
                });
            });
            // When the user clicks on <span> (x), close the loading modal
            loadingSpan.onclick = function() {
                loadingModal.style.display = "none";
            }

            // --- Help Modal Logic ---
            var showHelpBtn = document.getElementById("showHelpBtn");
            var helpModalContent = document.getElementById("helpModalContent");

            if (showHelpBtn) {
                showHelpBtn.onclick = function() {
                    helpModalContent.textContent = "Loading help output from 'llama-server --help'...";
                    helpModal.style.display = "block";

                    // Fetch the help content from the API
                    fetch('/api/llama_help')
                        .then(response => {
                            if (!response.ok) {
                                throw new Error(`HTTP error! Status: ${response.status}`);
                            }
                            return response.json();
                        })
                        .then(data => {
                            helpModalContent.textContent = data.help_text; // Populate the modal
                        })
                        .catch(error => {
                            console.error('Error fetching llama-server help:', error);
                            helpModalContent.textContent = 'Failed to load help content. Check server logs for details. Is llama-server in your PATH?';
                        });
                }
            }

            // When the user clicks on <span> (x) for the help modal
            if (helpSpan) {
                helpSpan.onclick = function() {
                    helpModal.style.display = "none";
                }
            }

            // When the user clicks anywhere outside of *either* modal, close it
            window.onclick = function(event) {
                if (event.target == loadingModal) {
                    loadingModal.style.display = "none";
                }
                if (event.target == helpModal) {
                    helpModal.style.display = "none";
                }
            }
        </script>
        </body>
    </html>
    """

    return render_template_string(
        html_template,
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

    # The HTML template for the new browse page
    html_template_browse = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Browse Local Cache - LLM Manager</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                margin: 0; padding: 0; background-color: #f4f7f6; display: grid;
                place-items: center; min-height: 100vh;
            }
            .container {
                width: 90%; max-width: 1000px; background: #ffffff;
                border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.07);
                padding: 2.5rem; margin: 2rem 0;
            }
            h1 { color: #2c3e50; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }
            button, .button {
                background-color: #3498db; color: white; padding: 12px 20px;
                border: none; border-radius: 6px; cursor: pointer; font-size: 16px;
                transition: background-color 0.2s; text-decoration: none; display: inline-block;
            }
            button:hover, .button:hover { background-color: #2980b9; }
            .button-small { padding: 6px 12px; font-size: 14px; }
            .button-secondary { background-color: #2ecc71; }
            .button-secondary:hover { background-color: #27ae60; }

            .flash {
                padding: 1rem; margin-bottom: 1rem; border-radius: 6px;
                font-weight: 600;
            }
            .flash.success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .flash.error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }

            .nav-link {
                display: inline-block;
                background: #ecf0f1;
                color: #34495e;
                padding: 10px 15px;
                border-radius: 6px;
                text-decoration: none;
                font-weight: 600;
                margin-bottom: 1.5rem;
                transition: background-color 0.2s;
            }
            .nav-link:hover { background: #dfe6e9; }

            .search-bar { display: flex; gap: 10px; margin-bottom: 1.5rem; }
            input[type="text"] {
                flex-grow: 1; padding: 12px; border: 1px solid #bdc3c7;
                border-radius: 6px; box-sizing: border-box;
            }

            .table-container {
                width: 100%;
                overflow-x: auto;
                border: 1px solid #ecf0f1;
                border-radius: 8px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
            }
            th, td {
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #ecf0f1;
                white-space: nowrap;
            }
            th { background-color: #f9fafb; color: #555; font-size: 14px; }
            tr:last-child td { border-bottom: none; }
            tr:hover { background-color: #fdfdfd; }
            .col-repo { min-width: 250px; }
            .col-file { min-width: 300px; }
            .col-quant { width: 120px; }
            .col-action { width: 120px; text-align: center; }
            /* Basic Modal Styling */
            .modal {
                display: none; /* Hidden by default */
                position: fixed; /* Stay in place */
                z-index: 1; /* Sit on top */
                left: 0;
                top: 0;
                width: 100%; /* Full width */
                height: 100%; /* Full height */
                overflow: auto; /* Enable scroll if needed */
                background-color: rgba(0,0,0,0.4); /* Black w/ opacity */
            }
            .modal-content {
                background-color: #fefefe;
                margin: 15% auto; /* 15% from the top and centered */
                padding: 20px;
                border: 1px solid #888;
                width: 80%; /* Could be more or less, depending on screen size */
                max-width: 400px;
                border-radius: 10px;
                text-align: center;
                font-size: 1.2em;
            }
            .close {
                color: #aaa;
                float: right;
                font-size: 28px;
                font-weight: bold;
            }
            .close:hover,
            .close:focus {
                color: black;
                text-decoration: none;
                cursor: pointer;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <a href="/" class="nav-link">&laquo; Back to Model Manager</a>
             <a href="/search_hf" class="nav-link">Search Hugging Face &raquo;</a>
            <h1>Browse Local GGUF Model Cache</h1>

            <!-- Flash Messages -->
            {% with messages = get_flashed_messages(with_categories=true) %}
              {% if messages %}
                {% for category, message in messages %}
                  <div class="flash {{ category }}">{{ message }}</div>
                {% endfor %}
              {% endif %}
            {% endwith %}

            <div class="search-bar">
                <form action="/refresh_cache" method="POST" style="margin:0;" id="refreshCacheForm">
                    <button type="submit" class="button-secondary">Refresh Cache (Top 500 Repos)</button>
                </form>
                <form action="/browse" method="GET" style="margin:0; display:flex; flex-grow:1;">
                    <input type="text" name="q" placeholder="Search by repo, filename, or quantization..." value="{{ search_query }}">
                    <button type="submit" style="margin-left:10px;">Search</button>
                </form>
            </div>

            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th class="col-repo">Repo ID</th>
                            <th class="col-file">Filename</th>
                            <th class="col-quant">Quantization</th>
                            <th class="col-action">Action</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for model in models %}
                        <tr>
                            <td>{{ model['repo_id'] }}</td>
                            <td>{{ model['filename'] }}</td>
                            <td>{{ model['quantization'] }}</td>
                            <td class="col-action">
                                <form action="/download" method="POST" class="download-form" style="margin:0;">
                                    <input type="hidden" name="repo_id" value="{{ model['repo_id'] }}">
                                    <input type="hidden" name="filename" value="{{ model['filename'] }}">
                                    <button type="submit" class="button-small">Download</button>
                                </form>
                            </td>
                        </tr>
                        {% else %}
                        <tr>
                            <td colspan="4" style="text-align:center; padding: 2rem; color: #777;">
                                No models found in cache. Try refreshing!
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>

        </div>

        <!-- The Modal -->
        <div id="loadingModal" class="modal">
            <div class="modal-content">
                <span class="close">&times;</span>
                <p id="modalMessage">Downloading model... Please wait.</p>
            </div>
        </div>

         <script>
            // Get the modal
            var modal = document.getElementById("loadingModal");
            var modalMessage = document.getElementById("modalMessage");

            // Get the <span> element that closes the modal
            var span = document.getElementsByClassName("close")[0];

            // Get all download forms
            var downloadForms = document.querySelectorAll(".download-form");
            var refreshCacheForm = document.getElementById("refreshCacheForm");

            // When the user clicks on a download button, open the modal
            downloadForms.forEach(function(form) {
                form.addEventListener("submit", function(event) {
                    modalMessage.textContent = "Downloading model... Please wait.";
                    modal.style.display = "block";
                });
            });

            // When the user clicks on the refresh cache button, open the modal
            if (refreshCacheForm) {
                 refreshCacheForm.addEventListener("submit", function(event) {
                    modalMessage.textContent = "Refreshing model cache... This may take a few minutes.";
                    modal.style.display = "block";
                });
            }


            // When the user clicks on <span> (x), close the modal
            span.onclick = function() {
                modal.style.display = "none";
            }

            // When the user clicks anywhere outside of the modal, close it
            window.onclick = function(event) {
                if (event.target == modal) {
                    modal.style.display = "none";
                }
            }
        </script>
    </body>
    </html>
    """

    return render_template_string(
        html_template_browse,
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

    # The HTML template for the Hugging Face search page
    html_template_search_hf = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Search Hugging Face - LLM Manager</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                margin: 0; padding: 0; background-color: #f4f7f6; display: grid;
                place-items: center; min-height: 100vh;
            }
            .container {
                width: 90%; max-width: 1000px; background: #ffffff;
                border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.07);
                padding: 2.5rem; margin: 2rem 0;
            }
            h1 { color: #2c3e50; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }
             button, .button {
                background-color: #3498db; color: white; padding: 12px 20px;
                border: none; border-radius: 6px; cursor: pointer; font-size: 16px;
                transition: background-color 0.2s; text-decoration: none; display: inline-block;
            }
            button:hover, .button:hover { background-color: #2980b9; }
            .button-small { padding: 6px 12px; font-size: 14px; }

            .flash {
                padding: 1rem; margin-bottom: 1rem; border-radius: 6px;
                font-weight: 600;
            }
            .flash.success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .flash.error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }

            .nav-link {
                display: inline-block;
                background: #ecf0f1;
                color: #34495e;
                padding: 10px 15px;
                border-radius: 6px;
                text-decoration: none;
                font-weight: 600;
                margin-bottom: 1.5rem;
                transition: background-color 0.2s;
            }
            .nav-link:hover { background: #dfe6e9; }

            .search-bar { display: flex; gap: 10px; margin-bottom: 1.5rem; }
            input[type="text"] {
                flex-grow: 1; padding: 12px; border: 1px solid #bdc3c7;
                border-radius: 6lypx; box-sizing: border-box;
            }

            .table-container {
                width: 100%;
                overflow-x: auto;
                border: 1px solid #ecf0f1;
                border-radius: 8px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
            }
            th, td {
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #ecf0f1;
                white-space: nowrap;
            }
            th { background-color: #f9fafb; color: #555; font-size: 14px; }
            tr:last-child td { border-bottom: none; }
            tr:hover { background-color: #fdfdfd; }
            .col-repo { min-width: 250px; }
            .col-action { width: 250px; text-align: center; } /* Increased width for two buttons */
             /* Basic Modal Styling */
            .modal {
                display: none; /* Hidden by default */
                position: fixed; /* Stay in place */
                z-index: 1; /* Sit on top */
                left: 0;
                top: 0;
                width: 100%; /* Full width */
                height: 100%; /* Full height */
                overflow: auto; /* Enable scroll if needed */
                background-color: rgba(0,0,0,0.4); /* Black w/ opacity */
            }
            .modal-content {
                background-color: #fefefe;
                margin: 15% auto; /* 15% from the top and centered */
                padding: 20px;
                border: 1px solid #888;
                width: 80%; /* Could be more or less, depending on screen size */
                max-width: 400px;
                border-radius: 10px;
                text-align: center;
                font-size: 1.2em;
            }
            .close {
                color: #aaa;
                float: right;
                font-size: 28px;
                font-weight: bold;
            }
            .close:hover,
            .close:focus {
                color: black;
                text-decoration: none;
                cursor: pointer;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <a href="/" class="nav-link">&laquo; Back to Model Manager</a>
             <a href="/browse" class="nav-link">Browse Local Cache &raquo;</a>
            <h1>Search Hugging Face for GGUF Models (llama.cpp)</h1>

            <!-- Flash Messages -->
            {% with messages = get_flashed_messages(with_categories=true) %}
              {% if messages %}
                {% for category, message in messages %}
                  <div class="flash {{ category }}">{{ message }}</div>
                {% endfor %}
              {% endif %}
            {% endwith %}

            <div class="search-bar">
                <form action="/search_hf" method="GET" style="margin:0; display:flex; flex-grow:1;">
                    <input type="text" name="q" placeholder="Enter search query..." value="{{ search_query }}">
                    <button type="submit" style="margin-left:10px;">Search</button>
                </form>
            </div>

            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th class="col-repo">Repo ID</th>
                            <th>Downloads</th>
                             <th class="col-action">Action</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for model in models %}
                        <tr>
                            <td>{{ model.modelId }}</td>
                            <td>{{ model.downloads }}</td>
                             <td class="col-action">
                                <a href="https://huggingface.co/{{ model.modelId }}" target="_blank" class="button button-small">View Repo</a>
                                <form action="/list_files" method="GET" style="margin:0; display:inline-block;">
                                    <input type="hidden" name="repo_id" value="{{ model.modelId }}">
                                    <button type="submit" class="button-small button-secondary">List Files</button>
                                </form>
                             </td>
                        </tr>
                        {% else %}
                         {% if search_query %}
                            <tr>
                                <td colspan="3" style="text-align:center; padding: 2rem; color: #777;">
                                    No models found matching your search criteria.
                                </td>
                            </tr>
                          {% else %}
                             <tr>
                                <td colspan="3" style="text-align:center; padding: 2rem; color: #777;">
                                    Enter a search query to find models on Hugging Face.
                                </td>
                            </tr>
                          {% endif %}
                        {% endfor %}
                    </tbody>
                </table>
            </div>

        </div>

        <!-- The Modal -->
        <div id="loadingModal" class="modal">
            <div class="modal-content">
                <span class="close">&times;</span>
                <p>Downloading model... Please wait.</p>
            </div>
        </div>

         <script>
            // Get the modal
            var modal = document.getElementById("loadingModal");

            // Get the <span> element that closes the modal
            var span = document.getElementsByClassName("close")[0];

            // Get all download forms
            var downloadForms = document.querySelectorAll(".download-form");

            // When the user clicks on a download button, open the modal
            downloadForms.forEach(function(form) {
                form.addEventListener("submit", function(event) {
                    modal.style.display = "block";
                });
            });


            // When the user clicks on <span> (x), close the modal
            span.onclick = function() {
                modal.style.display = "none";
            }

            // When the user clicks anywhere outside of the modal, close it
            window.onclick = function(event) {
                if (event.target == modal) {
                    modal.style.display = "none";
                }
            }
        </script>
    </body>
    </html>
    """

    return render_template_string(
        html_template_search_hf,
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

    html_template_list_files = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>GGUF Files - {{ repo_id }}</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                margin: 0; padding: 0; background-color: #f4f7f6; display: grid;
                place-items: center; min-height: 100vh;
            }
            .container {
                width: 90%; max-width: 800px; background: #ffffff;
                border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.07);
                padding: 2.5rem; margin: 2rem 0;
            }
            h1 { color: #2c3e50; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }
             button, .button {
                background-color: #3498db; color: white; padding: 12px 20px;
                border: none; border-radius: 6px; cursor: pointer; font-size: 16px;
                transition: background-color 0.2s; text-decoration: none; display: inline-block;
            }
            button:hover, .button:hover { background-color: #2980b9; }
            .button-small { padding: 6px 12px; font-size: 14px; }

            .flash {
                padding: 1rem; margin-bottom: 1rem; border-radius: 6px;
                font-weight: 600;
            }
            .flash.success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .flash.error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }

            .nav-link {
                display: inline-block;
                background: #ecf0f1;
                color: #34495e;
                padding: 10px 15px;
                border-radius: 6px;
                text-decoration: none;
                font-weight: 600;
                margin-bottom: 1.5rem;
                transition: background-color 0.2s;
            }
            .nav-link:hover { background: #dfe6e9; }

            ul { list-style: none; padding: 0; }
            li {
                background: #ecf0f1;
                padding: 12px 15px;
                border-radius: 6px;
                margin-bottom: 8px;
                word-break: break-all; /* Break long filenames */
                display: flex; /* Use flexbox for layout */
                justify-content: space-between; /* Space out filename and button */
                align-items: center; /* Vertically align items */
            }
             /* Basic Modal Styling */
            .modal {
                display: none; /* Hidden by default */
                position: fixed; /* Stay in place */
                z-index: 1; /* Sit on top */
                left: 0;
                top: 0;
                width: 100%; /* Full width */
                height: 100%; /* Full height */
                overflow: auto; /* Enable scroll if needed */
                background-color: rgba(0,0,0,0.4); /* Black w/ opacity */
            }
            .modal-content {
                background-color: #fefefe;
                margin: 15% auto; /* 15% from the top and centered */
                padding: 20px;
                border: 1px solid #888;
                width: 80%; /* Could be more or less, depending on screen size */
                max-width: 400px;
                border-radius: 10px;
                text-align: center;
                font-size: 1.2em;
            }
            .close {
                color: #aaa;
                float: right;
                font-size: 28px;
                font-weight: bold;
            }
            .close:hover,
            .close:focus {
                color: black;
                text-decoration: none;
                cursor: pointer;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <a href="/search_hf" class="nav-link">&laquo; Back to Search Results</a>
            <h1>GGUF Files for {{ repo_id }}</h1>

            <!-- Flash Messages -->
            {% with messages = get_flashed_messages(with_categories=true) %}
              {% if messages %}
                {% for category, message in messages %}
                  <div class="flash {{ category }}">{{ message }}</div>
                {% endfor %}
              {% endif %}
            {% endwith %}

            {% if gguf_files %}
                <ul>
                    {% for filename in gguf_files %}
                        <li>
                            <span>{{ filename }}</span>
                            <form action="/download" method="POST" class="download-form" style="margin:0; display:inline-block;">
                                <input type="hidden" name="repo_id" value="{{ repo_id }}">
                                <input type="hidden" name="filename" value="{{ filename }}">
                                <button type="submit" class="button-small">Download</button>
                            </form>
                        </li>
                    {% endfor %}
                </ul>
            {% else %}
                <p>No GGUF files found for this repository, or an error occurred.</p>
            {% endif %}

        </div>

        <!-- The Modal -->
        <div id="loadingModal" class="modal">
            <div class="modal-content">
                <span class="close">&times;</span>
                <p>Downloading model... Please wait.</p>
            </div>
        </div>

         <script>
            // Get the modal
            var modal = document.getElementById("loadingModal");

            // Get the <span> element that closes the modal
            var span = document.getElementsByClassName("close")[0];

            // Get all download forms
            var downloadForms = document.querySelectorAll(".download-form");

            // When the user clicks on a download button, open the modal
            downloadForms.forEach(function(form) {
                form.addEventListener("submit", function(event) {
                    modal.style.display = "block";
                });
            });


            // When the user clicks on <span> (x), close the modal
            span.onclick = function() {
                modal.style.display = "none";
            }

            // When the user clicks anywhere outside of the modal, close it
            window.onclick = function(event) {
                if (event.target == modal) {
                    modal.style.display = "none";
                }
            }
        </script>
    </body>
    </html>
    """
    return render_template_string(
        html_template_list_files,
        repo_id=repo_id,
        gguf_files=gguf_files
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

    # Redirect back to the page where the download was initiated
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
            # Use shlex.split to correctly parse arguments (handles quotes)
            custom_args = shlex.split(custom_args_str)
        except ValueError as e:
            logger.warning(f"Error parsing custom arguments: {e}")
            flash(f"Error parsing custom arguments: {e}", "error")
            return redirect(url_for('index'))

        # Check for any forbidden flags
        for arg in custom_args:
            # Check for exact flag matches (e.g., '--port')
            if arg in FORBIDDEN_FLAGS:
                flash(f"Forbidden argument: '{arg}'. This is managed by the app.", "error")
                return redirect(url_for('index'))
            # Check for flags with values (e.g., '--port=8080')
            if '=' in arg:
                flag_part = arg.split('=', 1)[0]
                if flag_part in FORBIDDEN_FLAGS:
                    flash(f"Forbidden argument: '{flag_part}'. This is managed by the app.", "error")
                    return redirect(url_for('index'))

    try:
        port = find_next_available_port(STARTING_PORT)
        command = [
            "llama-server",
            "-m", model_path,
            "--port", str(port)
        ]

        if '-c' not in custom_args and '--context-size' not in custom_args:
            custom_args.extend(['-c', "4096"])

        if '-ngl' not in custom_args and '--n-gpu-layers' not in custom_args:
            custom_args.extend(['-ngl', "-1"])

        command.extend(custom_args)

        logger.info(f"Starting server with command: {' '.join(command)}")

        # Use process group flags to ensure we can terminate the server and its children
        create_new_group_flags = {}
        if os.name == 'posix':
            create_new_group_flags['preexec_fn'] = os.setsid
        elif os.name == 'nt':
            create_new_group_flags['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP

        # Start the subprocess
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            **create_new_group_flags
        )

        # Give it a few seconds to start up or fail
        sleep(10)

        return_code = proc.poll()
        if return_code is not None:
            # Process terminated, which means it failed to start
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


@app.route('/status')
def model_status():
    """Displays the status and inference page for the currently loaded model."""

    # Check if the server process is actually running
    is_running = bool(llm_server_contexts)

    html_template_status = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Server Status - LLM Manager</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                margin: 0; padding: 0; background-color: #f4f7f6; display: grid;
                place-items: center; min-height: 100vh;
            }
            .container {
                width: 90%; max-width: 800px; background: #ffffff;
                border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.07);
                padding: 2.5rem; margin: 2rem 0;
            }
            h1 { color: #2c3e50; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }
            h2 { color: #34495e; margin-top: 1.5rem; }
            .card { background: #ecf0f1; border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem; }
            .nav-link {
                display: inline-block;
                background: #ecf0f1;
                color: #34495e;
                padding: 10px 15px;
                border-radius: 6px;
                text-decoration: none;
                font-weight: 600;
                margin-bottom: 1.5rem;
                transition: background-color 0.2s;
            }
            .nav-link:hover { background: #dfe6e9; }

            .metric-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 1rem;
            }
            .metric-box {
                background: #fff;
                padding: 1rem;
                border-radius: 8px;
                border: 1px solid #dfe6e9;
                text-align: center;
            }
            .metric-box-label {
                font-size: 0.9em;
                color: #555;
                font-weight: 600;
                margin-bottom: 8px;
            }
            .metric-box-value {
                font-size: 2em;
                font-weight: 700;
                color: #2c3e50;
            }
            .card a {
                color: #3498db;
                text-decoration: none;
                font-weight: 600;
            }
            .card a:hover {
                text-decoration: underline;
            }
            .model-status-separator {
                border-bottom: 2px solid #ecf0f1;
                margin: 2rem 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <a href="/" class="nav-link">&laquo; Back to Model Manager</a>

            {% if contexts %}
                
                {% for port, context in contexts.items() %}
                
                <div class="model-status-card">
                    <h1>Server Status: {{ context['name'] }}</h1>

                    <div class="card">
                        <h2>Server Info</h2>
                        <p><strong>Model:</strong> {{ context['name'] }}</p>
                        <p><strong>Endpoint:</strong> 
                            <code>
                                <a href="http://127.0.0.1:{{ port }}" target="_blank">
                                    http://127.0.0.1:{{ port }}
                                </a>
                            </code>
                        </p>
                    </div>

                    <div class="card">
                        <h2>Live Metrics (All Requests)</h2>
                        <div class="metric-grid">
                            <div class="metric-box">
                                <div class="metric-box-label">Tokens/Second (Last Req)</div>
                                <div class="metric-box-value" id="tps-display-{{ port }}">0.00</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-box-label">Total Requests (Since Load)</div>
                                <div class="metric-box-value" id="req-count-display-{{ port }}">0</div>
                            </div>
                        </div>
                    </div>
                </div>
                
                {% if not loop.last %}
                    <div class="model-status-separator"></div>
                {% endif %}

                {% endfor %}
            
            {% else %}
                <h1>Server Status</h1>
                <p>No model is currently loaded. Go back to the <a href="/">main page</a> to load one.</p>
            {% endif %}
            </div>

        {% if contexts %}
        <script>
            // Define the fetch function ONCE
            async function fetchMetrics(port) {
                // Get elements based on unique port
                const tpsDisplay = document.getElementById('tps-display-' + port);
                const reqCountDisplay = document.getElementById('req-count-display-' + port);
                const metricsUrl = '/api/metrics?port=' + port;

                if (!tpsDisplay || !reqCountDisplay) {
                    // Stop trying if elements don't exist
                    return;
                }

                try {
                    const response = await fetch(metricsUrl);
                    if (!response.ok) {
                        throw new Error('Network response was not ok');
                    }
                    const metrics = await response.json();

                    // Update the UI
                    tpsDisplay.textContent = metrics.last_tps.toFixed(2);
                    reqCountDisplay.textContent = metrics.total_requests;

                } catch (error) {
                    // console.error('Error fetching metrics:', error);
                    tpsDisplay.textContent = 'Error';
                    reqCountDisplay.textContent = 'Error';
                }
            }

            // Loop through the ports from Flask and start the fetchers
            {% for port in contexts.keys() %}
                fetchMetrics({{ port }});
                setInterval(() => fetchMetrics({{ port }}), 2000);
            {% endfor %}

        </script>
        {% endif %}
        </body>
    </html>
    """

    return render_template_string(
        html_template_status,
        is_running=is_running,
        contexts=llm_server_contexts
    )


@app.route('/api/llama_help')
def api_llama_help():
    """Runs 'llama-server --help' and returns the output."""
    try:
        # Use subprocess.run for a simple one-off command.
        # We add a timeout for safety.
        result = subprocess.run(
            ["llama-server", "--help"],
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
    # 1. Check for Authorization
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        return jsonify({"error": "Authorization header is missing"}), 401

    # 2. Find the target model and port
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
    # The llama-server backend is also listening on this path
    target_url = f"http://127.0.0.1:{target_port}/v1/chat/completions"

    # 3. Get the raw request data
    raw_data = request.get_data()

    # 4. Prepare headers for the forwarded request
    forward_headers = {
        'Content-Type': request.headers.get('Content-Type', 'application/json'),
        'Authorization': auth_header,
        'Accept': request.headers.get('Accept', 'text/event-stream')
    }

    # 5. Check if streaming is requested
    is_stream = data.get('stream', False)

    try:
        # 6. Make the request and stream or return the response
        if is_stream:
            # Forward as a streaming request
            proxy_response = requests.post(
                target_url,
                data=raw_data,
                headers=forward_headers,
                stream=True
            )
            proxy_response.raise_for_status()

            # Return a streaming response
            return Response(
                proxy_response.iter_content(chunk_size=8192),
                status=proxy_response.status_code,
                content_type=proxy_response.headers.get('Content-Type')
            )
        else:
            # Forward as a non-streaming request
            proxy_response = requests.post(
                target_url,
                data=raw_data,
                headers=forward_headers,
                stream=False
            )
            proxy_response.raise_for_status()

            # Return the complete JSON response
            return jsonify(proxy_response.json()), proxy_response.status_code

    except requests.exceptions.RequestException as e:
        logger.error(f"Proxy request failed: {e}")
        return jsonify({"error": f"Failed to connect to backend model: {e}"}), 502
    except Exception as e:
        logger.error(f"Proxy error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"An internal error occurred: {e}"}), 500


# --- Run the App ---

if __name__ == '__main__':
    load_dotenv()
    default_host = os.environ.get('APP_HOST', '0.0.0.0')
    default_port = int(os.environ.get('APP_PORT', 5001))
    default_debug = os.environ.get('APP_DEBUG', 'false').lower() in ('true', '1', 'yes')
    default_verbose = os.environ.get('APP_VERBOSE', 'false').lower() in ('true', '1', 'yes')
    default_proxy_host = os.environ.get('PROXY_HOST', '0.0.0.0')
    default_proxy_port = int(os.environ.get('PROXY_PORT', 8080))

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
        '--model-dir-path',
        action='store_true',
        help="Print the absolute path to the model storage directory and exit."
    )
    parser.add_argument(
        '--proxy-host', type=str, default=default_proxy_host,
        help=f"The host for the OpenAI Proxy. (Env: PROXY_HOST, Default: {default_proxy_host})"
    )
    parser.add_argument(
        '--proxy-port', type=int, default=default_proxy_port,
        help=f"The port for the OpenAI Proxy. (Env: PROXY_PORT, Default: {default_proxy_port})"
    )
    args = parser.parse_args()

    if args.model_dir_path:
        print(os.path.abspath(MODEL_DIR))
        sys.exit(0)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    if not args.verbose:
        logging.getLogger('werkzeug').setLevel(logging.WARNING)

    # Initialize the database on startup
    init_db()

    app.config['PROXY_HOST'] = args.proxy_host
    app.config['PROXY_PORT'] = args.proxy_port
    def run_proxy_server():
        try:
            # Note: We don't use proxy_app.run(debug=True) as it conflicts with threading
            logger.info(f"Starting OpenAI proxy server on http://{args.proxy_host}:{args.proxy_port}")
            proxy_app.run(host=args.proxy_host, port=args.proxy_port)
        except Exception as e:
            logger.error(f"Proxy server failed to start: {e}")
            logger.error(traceback.format_exc())

    proxy_thread = threading.Thread(target=run_proxy_server, daemon=True)
    proxy_thread.start()

    sleep(1)

    # Print a helpful startup message
    print(f"Starting Flask app on http://{args.host}:{args.port}")
    if args.debug:
        print("Running in DEBUG mode. Do not use in production.")

    # Run the app with the parsed arguments
    app.run(debug=args.debug, host=args.host, port=args.port)