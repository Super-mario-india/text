
import streamlit as st
import sqlite3
import re
import hashlib
import datetime
import Levenshtein
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
from textblob import TextBlob
from transformers import pipeline
import torch
import plotly.express as px
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import json
import requests
import base64
from PIL import Image
import io
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.card import card
import time
import speech_recognition as sr
import pyttsx3
from googletrans import Translator
import docx
import PyPDF2
import os
from io import BytesIO
from streamlit_mic_recorder import mic_recorder
import base64
import sqlite3
import textstat
from evaluate import load
from rouge_score import rouge_scorer
import multiprocessing
import sqlite3, os, time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import Dataset
import torch
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import string
import html
import zipfile
import random
import threading
import time
from googletrans import Translator, LANGUAGES
import spacy
from sentence_transformers import SentenceTransformer, util
import difflib
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import train_test_split
from transformers import DataCollatorForSeq2Seq
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from streamlit_autorefresh import st_autorefresh
from transformers import TrainerCallback


DB_PATH = "users.db"

def calculate_diversity_metrics(predictions, references):
    """Calculates the average edit distance between two lists of strings."""
    distances = [Levenshtein.distance(pred, ref) for pred, ref in zip(predictions, references)]
    return np.mean(distances) if distances else 0.0

class ProgressCallback(TrainerCallback):
    def __init__(self, model_id, db_path="users.db"):
        self.model_id = model_id
        self.db_path = db_path

    def on_step_end(self, args, state, control, **kwargs):
        # Calculate progress
        progress_pct = state.global_step / state.max_steps

        # Get the latest loss
        try:
            loss = state.log_history[-1].get('loss', 0.0)
            status_text = f"Step {state.global_step}/{state.max_steps} | Loss: {loss:.4f}"
        except (IndexError, KeyError):
            status_text = f"Step {state.global_step}/{state.max_steps}"

        # Update the database
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            UPDATE models
            SET training_progress = ?, training_status_text = ?
            WHERE model_id = ?
        """, (progress_pct, status_text, self.model_id))
        conn.commit()
        conn.close()

# --- VVV REPLACE this entire function VVV ---

def fetch_and_inspect_kaggle_dataset(dataset_identifier):
    """
    Downloads a Kaggle dataset, unzips it, finds the largest CSV or JSON file,
    and returns it as a pandas DataFrame along with its info.
    """
    download_path = "" # Initialize to handle potential errors before assignment
    try:
        # Initialize Kaggle API
        api = KaggleApi()
        api.authenticate()

        # Sanitize the identifier to get the user/dataset part
        if "kaggle.com/datasets/" in dataset_identifier:
            dataset_identifier = dataset_identifier.split("datasets/")[1].strip()

        # Create a unique temporary directory for the download
        safe_identifier = re.sub(r'[^a-zA-Z0-9_-]', '_', dataset_identifier)
        download_path = f"./kaggle_temp_{safe_identifier}"
        os.makedirs(download_path, exist_ok=True)

        # Download and unzip the dataset files
        api.dataset_download_files(dataset_identifier, path=download_path, unzip=True)

        # --- NEW: Intelligent File Detection ---
        # 1. Look for both .csv and .json/.jsonl files
        data_files = [f for f in os.listdir(download_path) if f.endswith(('.csv', '.json', '.jsonl'))]

        if not data_files:
            shutil.rmtree(download_path) # Clean up
            return None, "No CSV or JSON data files were found in this dataset."

        # 2. Assume the largest file is the primary data source (a common convention)
        primary_data_file = max(data_files, key=lambda f: os.path.getsize(os.path.join(download_path, f)))
        full_file_path = os.path.join(download_path, primary_data_file)

        # 3. Read the file using the correct pandas function based on its extension
        if primary_data_file.endswith('.csv'):
            df = pd.read_csv(full_file_path)
        else:  # Handles .json and .jsonl
            # .jsonl (JSON Lines) is common for large datasets, where each line is a separate JSON object
            is_jsonl = primary_data_file.endswith('.jsonl')
            df = pd.read_json(full_file_path, orient='records', lines=is_jsonl)

        # Clean up the downloaded files
        shutil.rmtree(download_path)

        return df, None  # Success

    except Exception as e:
        # Ensure cleanup happens even if there's an error
        if os.path.exists(download_path):
            shutil.rmtree(download_path)
        return None, f"Failed to fetch dataset. Error: {e}. Ensure the identifier is correct and your kaggle.json is set up."

def get_smtp_settings(db_path="users.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT email, password, registration_msg, reset_msg FROM smtp_settings LIMIT 1")
    row = cur.fetchone()
    conn.close()
    return row if row else (None, None, None, None)

def send_mail(to_email, subject, message, db_path="users.db"):
    sender, password, _, _ = get_smtp_settings(db_path)
    if not sender or not password:
        raise ValueError("❌ SMTP settings not configured in Admin Dashboard!")

    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(message, "plain"))

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender, password)
            server.sendmail(sender, to_email, msg.as_string())
        return True
    except Exception as e:
        print(f"❌ Failed to send mail: {e}")
        return False

def send_registration_otp(to_email, username, otp, db_path="users.db"):
    sender, _, reg_msg, _ = get_smtp_settings(db_path)
    if not reg_msg:
        reg_msg = "Hello {username}, your OTP is {otp}."
    message = reg_msg.format(username=username, otp=otp)
    return send_mail(to_email, "🔐 Your OTP", message, db_path)

def send_password_reset(to_email, username, link, db_path="users.db"):
    sender, _, _, reset_msg = get_smtp_settings(db_path)
    if not reset_msg:
        reset_msg = "Hello {username}, click here to reset your password: {link}"
    message = reset_msg.format(username=username, link=link)
    return send_mail(to_email, "🔑 Password Reset", message, db_path)


# --- VVV REPLACE this entire function VVV ---

def train_worker(base_model_id, training_data, base_task_type, db_path, output_model_dir, test_size=0.2):
    global tokenizer
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # ✅ Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model_id).to(device)

        # ✅ Create data collator (IMPORTANT - must be after tokenizer & model)
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        train_pairs, eval_pairs = train_test_split(training_data, test_size=test_size, random_state=42)

        def create_dataset(data_pairs):
            inputs = [str(pair[0]) for pair in data_pairs]
            outputs = [str(pair[1]) for pair in data_pairs]
            return Dataset.from_dict({"input_text": inputs, "labels": outputs})

        train_dataset = create_dataset(train_pairs)
        eval_dataset = create_dataset(eval_pairs)

        def preprocess(batch):
            model_inputs = tokenizer(batch["input_text"], text_target=batch["labels"], max_length=512, truncation=True)
            return model_inputs

        tokenized_train_dataset = train_dataset.map(preprocess, batched=True)
        tokenized_eval_dataset = eval_dataset.map(preprocess, batched=True)

        training_args = TrainingArguments(
            output_dir=output_model_dir,
            learning_rate=5e-5,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            num_train_epochs=2,
            weight_decay=0.01,
            logging_dir=f"./logs/{os.path.basename(output_model_dir)}",
            report_to=[],
            eval_strategy="epoch",  # ✅ Changed from evaluation_strategy
            save_strategy="epoch",
            load_best_model_at_end=True
        )
        progress_callback = ProgressCallback(model_id=output_model_dir, db_path=db_path)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,  # ✅ Now this is properly defined
            compute_metrics=compute_metrics_for_trainer,
            callbacks=[progress_callback]
        )

        trainer.train()
        trainer.save_model(output_model_dir)
        tokenizer.save_pretrained(output_model_dir)

        # --- Post-Training Evaluation ---
        print("Starting post-training evaluation...")
        predictions = trainer.predict(tokenized_eval_dataset)

        # ✅ Convert logits to token IDs safely
        pred_ids = predictions.predictions
        if isinstance(pred_ids, tuple):  # sometimes predictions come as a tuple
            pred_ids = pred_ids[0]
        pred_ids = np.argmax(pred_ids, axis=-1)

        # ✅ Replace -100 with pad token ID in labels
        labels = predictions.label_ids
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        # ✅ Decode properly
        decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        eval_loss = predictions.metrics.get("test_loss", 0.0)
        final_remarks = f"Trained on {len(train_pairs)} samples, evaluated on {len(eval_pairs)}. Base: {base_model_id}"

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        if base_task_type == 'paraphrasing':
            bleu = load("bleu")
            bertscore = load("bertscore")

            bleu_score = bleu.compute(predictions=decoded_preds, references=decoded_labels)['bleu'] * 100
            bert_scores = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
            bert_f1 = np.mean(bert_scores['f1']) * 100
            edit_distance = calculate_diversity_metrics(decoded_preds, decoded_labels)

            cur.execute("""
                UPDATE models SET status='active', remarks=?, eval_loss=?, eval_bleu=?, eval_bertscore=?, eval_edit_distance=?
                WHERE model_id = ?
            """, (final_remarks, eval_loss, bleu_score, bert_f1, edit_distance, output_model_dir))

        else:  # Summarization or other tasks
            rouge = load("rouge")
            rouge_score = rouge.compute(predictions=decoded_preds, references=decoded_labels)['rougeL'] * 100

            cur.execute("""
                UPDATE models SET status='active', remarks=?, eval_loss=?, eval_rougeL=?
                WHERE model_id = ?
            """, (final_remarks, eval_loss, rouge_score, output_model_dir))

        conn.commit()
        conn.close()
        print(f"✅ Training and evaluation complete for {output_model_dir}")

    except Exception as e:
        print(f"❌ Training failed for {output_model_dir}: {e}")


def run_model_task(model_id, task_type, text, result_dict):
    """
    A target function for threading. Runs an AI task and calculates metrics.
    """
    try:
        start_time = time.time()
        output_text = ""

        # Select the correct pipeline based on the task
        if task_type == "Summarization":
            pipeline, tokenizer = load_summarizer(model_id)
            if pipeline and tokenizer:
                output_text = summarize_text(pipeline, tokenizer, text, min_length=50, max_length=150)
        elif task_type == "Paraphrasing":
            pipeline, tokenizer = load_paraphraser(model_id)
            if pipeline and tokenizer:
                # Paraphrasing returns a list, join them for comparison
                paraphrases = paraphrase_text(pipeline, tokenizer, text, num_return_sequences=1)
                output_text = paraphrases[0] if paraphrases else "No paraphrase generated."

        end_time = time.time()

        # --- Calculate all metrics ---
        execution_time = end_time - start_time

        # Readability Scores
        try:
            readability = {
                "Flesch Reading Ease": textstat.flesch_reading_ease(output_text),
                "Flesch-Kincaid Grade": textstat.flesch_kincaid_grade(output_text),
                "Gunning Fog": textstat.gunning_fog(output_text),
            }
        except Exception: # Fallback for very short text
            readability = {k: 0 for k in ["Flesch Reading Ease", "Flesch-Kincaid Grade", "Gunning Fog"]}

        # ROUGE Scores
        try:
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(text, output_text)
            rouge = {
                "ROUGE-1": scores['rouge1'].fmeasure,
                "ROUGE-2": scores['rouge2'].fmeasure,
                "ROUGE-L": scores['rougeL'].fmeasure,
            }
        except Exception:
            rouge = {k: 0 for k in ["ROUGE-1", "ROUGE-2", "ROUGE-L"]}

        # Store results in the shared dictionary
        result_dict[model_id] = {
            "output": output_text,
            "time": execution_time,
            "readability": readability,
            "rouge": rouge
        }
    except Exception as e:
        result_dict[model_id] = {"error": str(e)}


def get_all_table_names(conn):
    """Fetches a list of all tables in the database, excluding sqlite internals."""
    cursor = conn.cursor()
    # MODIFIED: Exclude sqlite_sequence table
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence' ORDER BY name;")
    return [table[0] for table in cursor.fetchall()]


def generate_mysql_dump(conn):
    """
    Reads an SQLite database and generates a fully-compatible MySQL dump script
    that can be executed directly in MySQL Workbench.
    """

    def map_type(sqlite_type, is_pk):
        """Maps SQLite data types to appropriate MySQL data types."""
        sqlite_type = sqlite_type.upper()
        if "INT" in sqlite_type:
            return "BIGINT" if is_pk else "INT"
        if "TEXT" in sqlite_type or "CHAR" in sqlite_type:
            # Use VARCHAR for PKs as TEXT can't be a key without length
            return "VARCHAR(255)" if is_pk else "TEXT"
        if "BLOB" in sqlite_type:
            return "BLOB"
        if "REAL" in sqlite_type or "FLOAT" in sqlite_type or "DOUBLE" in sqlite_type:
            return "DOUBLE"
        return "VARCHAR(255)" # A safe default

    script_parts = [
        "-- Generated by TextMorph App for MySQL Workbench",
        "SET NAMES utf8mb4;",
        "SET FOREIGN_KEY_CHECKS=0;",
        "START TRANSACTION;",
        ""
    ]

    table_names = get_all_table_names(conn)
    cursor = conn.cursor()

    # First, generate all CREATE TABLE statements
    for table_name in table_names:
        script_parts.append(f"-- Table structure for table `{table_name}`")
        script_parts.append(f"DROP TABLE IF EXISTS `{table_name}`;")

        cursor.execute(f"PRAGMA table_info('{table_name}')")
        columns = cursor.fetchall()

        col_defs = []
        primary_keys = []
        is_autoincrement = False

        for col in columns:
            name, dtype, notnull, _, pk = col[1], col[2], col[3], col[4], col[5]
            is_pk = pk > 0
            col_def = f"  `{name}` {map_type(dtype, is_pk)}"
            if notnull: col_def += " NOT NULL"

            # Check for SQLite's specific AUTOINCREMENT behavior
            if is_pk and "INTEGER" in dtype.upper() and len([c for c in columns if c[5] > 0]) == 1:
                col_def += " AUTO_INCREMENT"
                is_autoincrement = True

            col_defs.append(col_def)
            if is_pk:
                primary_keys.append(f"`{name}`")

        # Add PRIMARY KEY constraint
        if primary_keys and not is_autoincrement:
            col_defs.append(f"  PRIMARY KEY ({', '.join(primary_keys)})")

        # Add FOREIGN KEY constraints
        cursor.execute(f"PRAGMA foreign_key_list('{table_name}')")
        foreign_keys = cursor.fetchall()
        for fk in foreign_keys:
            col_defs.append(f"  CONSTRAINT `fk_{table_name}_{fk[3]}` FOREIGN KEY (`{fk[3]}`) REFERENCES `{fk[2]}` (`{fk[4]}`)")

        create_table_sql = f"CREATE TABLE `{table_name}` (\n" + ",\n".join(col_defs) + "\n) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;"
        script_parts.append(create_table_sql)
        script_parts.append("")

    # Second, generate all INSERT statements
    for table_name in table_names:
        cursor.execute(f'SELECT * FROM "{table_name}"')
        rows = cursor.fetchall()
        if not rows: continue

        script_parts.append(f"-- Dumping data for table `{table_name}`")
        col_names = ", ".join([f"`{desc[0]}`" for desc in cursor.description])

        for row in rows:
            values = []
            for val in row:
                if val is None:
                    values.append("NULL")
                elif isinstance(val, bytes):
                    values.append(f"X'{val.hex()}'") # Correct BLOB handling
                elif isinstance(val, (int, float)):
                    values.append(str(val))
                else:
                    escaped_val = str(val).replace("'", "''").replace("\\", "\\\\")
                    values.append(f"'{escaped_val}'")

            insert_sql = f"INSERT INTO `{table_name}` ({col_names}) VALUES ({', '.join(values)});"
            script_parts.append(insert_sql)
        script_parts.append("")

    script_parts.extend(["COMMIT;", "SET FOREIGN_KEY_CHECKS=1;"])

    return "\n".join(script_parts)

def generate_pdf_for_table(df, table_name):
    """Creates a properly formatted, multi-page PDF representation of a DataFrame."""
    try:
        pdf = FPDF(orientation='L', unit='mm', format='A4')
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, f"Table: {table_name}", 0, 1, 'C')
        pdf.ln(5)

        # Prepare DataFrame for PDF
        df = df.astype(str)  # Convert all data to string to prevent errors
        cols = df.columns.tolist()

        # Set column widths (you can adjust these ratios)
        effective_page_width = pdf.w - 2 * pdf.l_margin
        col_widths = [effective_page_width / len(cols)] * len(cols)

        # --- Table Header ---
        pdf.set_font("Arial", 'B', 8)
        for i, col_name in enumerate(cols):
            pdf.cell(col_widths[i], 8, col_name, border=1, align='C')
        pdf.ln()

        # --- Table Body ---
        pdf.set_font("Arial", '', 8)
        for index, row in df.iterrows():
            # Check if there is enough space for the row, if not, add a new page
            if pdf.get_y() > (pdf.h - 30): # 30mm margin from bottom
                pdf.add_page()
                # Re-draw header on new page
                pdf.set_font("Arial", 'B', 8)
                for i, col_name in enumerate(cols):
                    pdf.cell(col_widths[i], 8, col_name, border=1, align='C')
                pdf.ln()
                pdf.set_font("Arial", '', 8)

            # Calculate the required height for the row by finding the max number of lines in any cell
            max_lines = 1
            for i, item in enumerate(row):
                lines = pdf.multi_cell(col_widths[i], 5, str(item), split_only=True)
                if len(lines) > max_lines:
                    max_lines = len(lines)
            row_height = max_lines * 5

            # Draw all cells in the row
            y_before = pdf.get_y()
            for i, item in enumerate(row):
                x_pos = pdf.l_margin + sum(col_widths[:i])
                pdf.set_xy(x_pos, y_before)
                pdf.multi_cell(col_widths[i], 5, str(item), border=1, align='L')

            # Move the cursor to the bottom of the tallest cell
            pdf.set_y(y_before + row_height)

        return pdf.output(dest="S").encode("latin-1")
    except Exception as e:
        # Fallback in case of a critical PDF generation error
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, f"Error generating PDF for table: {table_name}", 0, 1, 'C')
        pdf.multi_cell(0, 10, str(e))
        return pdf.output(dest="S").encode("latin-1")

def get_user_agents(username):
    """Fetches all models created by a specific user."""
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM models WHERE created_by_username = ?", conn, params=(username,))
    conn.close()
    return df

def delete_user_agent(model_id, username):
    """Deletes a model if the user is the owner."""
    conn = get_conn()
    cur = conn.cursor()
    # Security check: ensure the user owns this model
    cur.execute("SELECT model_id FROM models WHERE model_id = ? AND created_by_username = ?", (model_id, username))
    if cur.fetchone():
        cur.execute("DELETE FROM models WHERE model_id = ?", (model_id,))
        conn.commit()
        # Also delete the model files from the server
        if os.path.isdir(model_id):
            shutil.rmtree(model_id)
        st.success("AI Agent deleted successfully.")
    else:
        st.error("You do not have permission to delete this model.")
    conn.close()

def toggle_agent_public_status(model_id, username):
    """Toggles the is_public flag for a user's model."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT is_public FROM models WHERE model_id = ? AND created_by_username = ?", (model_id, username))
    result = cur.fetchone()
    if result:
        new_status = 1 - result[0] # Flip the bit (0->1, 1->0)
        cur.execute("UPDATE models SET is_public = ? WHERE model_id = ?", (new_status, model_id))
        conn.commit()
        st.success(f"Agent visibility set to {'Public' if new_status else 'Private'}.")
    conn.close()

def clone_agent_to_public(model_id):
    """Admin function to copy a user's agent and make it a public system model."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM models WHERE model_id = ?", (model_id,))
    original_model = cur.fetchone()

    if original_model:
        # Define new path for the public model
        original_path = original_model[0]
        new_path = os.path.join("fine_tuned_models", f"public_clone_{os.path.basename(original_path)}")

        # Copy model files
        if os.path.isdir(original_path):
            shutil.copytree(original_path, new_path)
        else:
            st.error("Original model files not found.")
            conn.close()
            return

        # Create new DB entry
        original_display_name = original_model[1]
        new_display_name = f"{original_display_name} (Public Clone)"
        new_remarks = f"Cloned from a user-trained agent. Original path: {original_path}"
        task_type = original_model[4]

        try:
            cur.execute("""
                INSERT INTO models (model_id, display_name, remarks, status, task_type, created_by_username, is_public)
                VALUES (?, ?, ?, 'active', ?, 'admin', 1)
            """, (new_path, new_display_name, new_remarks, task_type))
            conn.commit()
            st.success("Agent cloned and made public successfully!")
        except sqlite3.IntegrityError:
            st.error("A model with this path already exists.")
    conn.close()

def generate_full_database_backup_zip():
    """Generates a zip file in memory containing PDFs, CSVs, and a full SQL dump."""
    conn = get_conn()
    table_names = get_all_table_names(conn)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # 1. Add CSV and PDF for each table
        for table in table_names:
            df = pd.read_sql_query(f"SELECT * FROM \"{table}\"", conn)

            # Add CSV to zip
            csv_data = df.to_csv(index=False).encode('utf-8')
            zf.writestr(f"csv_export/{table}.csv", csv_data)

            # Add PDF to zip
            pdf_data = generate_pdf_for_table(df, table)
            zf.writestr(f"pdf_export/{table}.pdf", pdf_data)

        # 2. Add full SQL dump
        sql_dump_parts = []
        for line in conn.iterdump():
            sql_dump_parts.append(line)
        sql_dump = "\n".join(sql_dump_parts).encode('utf-8')
        zf.writestr("full_database_backup.sql", sql_dump)

        mysql_dump_str = generate_mysql_dump(conn)
        zf.writestr("fullsqlquery.txt", mysql_dump_str.encode('utf-8'))

    conn.close()
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

# --- VVV REPLACE this function VVV ---

def train_model(base_model_id, training_data, base_task_type, output_model_dir, test_size=0.2, db_path="users.db"):
    # The 'selected_ids' parameter is no longer used in the new pipeline, so it's removed.
    p = multiprocessing.Process(
        target=train_worker,
        args=(base_model_id, training_data, base_task_type, db_path, output_model_dir, test_size)
    )
    p.start()
    return True

def generate_join_code(length=8):
    """Generates a random alphanumeric join code."""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

def create_team(team_name, admin_username):
    """Creates a new team and adds the creator as admin."""
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    join_code = generate_join_code()
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cur.execute("INSERT INTO teams (team_name, admin_username, join_code, created_at) VALUES (?, ?, ?, ?)",
                (team_name, admin_username, join_code, now))
    team_id = cur.lastrowid
    cur.execute("INSERT INTO team_members (team_id, username, role, joined_at) VALUES (?, ?, ?, ?)",
                (team_id, admin_username, 'admin', now))
    conn.commit()
    conn.close()
    return team_id

def join_team(username, join_code):
    """Adds a user to a team using a join code."""
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("SELECT id FROM teams WHERE join_code=?", (join_code,))
    team = cur.fetchone()
    if not team:
        conn.close()
        return "Invalid Code"
    team_id = team[0]
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        cur.execute("INSERT INTO team_members (team_id, username, role, joined_at) VALUES (?, ?, ?, ?)",
                    (team_id, username, 'member', now))
        conn.commit()
        return "Success"
    except sqlite3.IntegrityError:
        return "Already a member"
    finally:
        conn.close()

def get_user_teams(username):
    """Fetches all teams a user is a member of."""
    conn = sqlite3.connect("users.db")
    query = """
        SELECT t.id, t.team_name, tm.role
        FROM teams t
        JOIN team_members tm ON t.id = tm.team_id
        WHERE tm.username = ? AND tm.status = 'active'
    """
    df = pd.read_sql_query(query, conn, params=(username,))
    conn.close()
    return df

def get_team_details(team_id):
    """Fetches details for a specific team."""
    conn = sqlite3.connect("users.db")
    df = pd.read_sql_query("SELECT * FROM teams WHERE id=?", conn, params=(team_id,))
    conn.close()
    return df.iloc[0] if not df.empty else None

def get_team_members(team_id):
    """Fetches all members of a team with their details."""
    conn = sqlite3.connect("users.db")
    query = """
        SELECT u.username, u.name, u.profile_pic, u.last_seen, tm.role, tm.status
        FROM users u
        JOIN team_members tm ON u.username = tm.username
        WHERE tm.team_id = ?
    """
    df = pd.read_sql_query(query, conn, params=(team_id,))
    conn.close()
    return df

def get_team_messages(team_id):
    """Fetches all chat messages for a team."""
    conn = sqlite3.connect("users.db")
    # Use LEFT JOIN to include messages from non-user authors like the AI
    # Use COALESCE to use the chat username if no matching user is found
    query = """
        SELECT
            c.username,
            COALESCE(u.name, c.username) as display_name,
            u.profile_pic,
            c.message,
            c.timestamp,
            c.message_type
        FROM team_chats c
        LEFT JOIN users u ON c.username = u.username
        WHERE c.team_id = ? ORDER BY c.timestamp ASC
    """
    df = pd.read_sql_query(query, conn, params=(team_id,))
    conn.close()
    return df

def post_team_message(team_id, username, message, message_type='user'):
    """Posts a new plain-text message to the team chat."""
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # ✅ Store only the raw message text (no HTML)
    cur.execute("""
        INSERT INTO team_chats (team_id, username, message, timestamp, message_type)
        VALUES (?, ?, ?, ?, ?)
    """, (team_id, username, message, now, message_type))

    conn.commit()
    conn.close()

def update_user_last_seen(username):
    """Updates the last_seen timestamp for a user."""
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cur.execute("UPDATE users SET last_seen=? WHERE username=?", (now, username))
    conn.commit()
    conn.close()

def update_team_settings(team_id, new_name, new_code, new_model, new_api_key):
    """Updates a team's settings."""
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("UPDATE teams SET team_name=?, join_code=?, default_model_id=?, gemini_api_key=? WHERE id=?",
                (new_name, new_code, new_model, new_api_key, team_id))
    conn.commit()
    conn.close()

def handle_document_upload(team_id, uploader_username, uploaded_file):
    """Saves a document, logs it, sets it as active, and returns its text."""
    # Ensure the main upload directory exists
    base_upload_path = "team_uploads"
    os.makedirs(base_upload_path, exist_ok=True)

    # Create a team-specific directory
    team_upload_path = os.path.join(base_upload_path, f"team_{team_id}")
    os.makedirs(team_upload_path, exist_ok=True)

    # Save the file
    file_path = os.path.join(team_upload_path, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Log the document in the database
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cur.execute("""
        INSERT INTO team_documents (team_id, uploader_username, file_name, file_path, uploaded_at)
        VALUES (?, ?, ?, ?, ?)
    """, (team_id, uploader_username, uploaded_file.name, file_path, now))
    doc_id = cur.lastrowid

    # Set this new document as the active one for the team
    cur.execute("UPDATE teams SET active_document_id=? WHERE id=?", (doc_id, team_id))

    conn.commit()
    conn.close()

    # Extract text from the saved file
    if uploaded_file.type == "application/pdf":
        return extract_text_from_pdf(file_path)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return extract_text_from_docx(file_path)
    else: # Assume text/plain
        return uploaded_file.getvalue().decode("utf-8")

def get_active_document_text(team_id):
    """Retrieves the text content of the active document for a team."""
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    # First, get the active document ID from the teams table
    cur.execute("SELECT active_document_id FROM teams WHERE id=?", (team_id,))
    result = cur.fetchone()
    if not result or not result[0]:
        conn.close()
        return None, "No active document. Please upload one first."

    # Then, get the file path from the documents table
    doc_id = result[0]
    cur.execute("SELECT file_path, file_name FROM team_documents WHERE id=?", (doc_id,))
    doc_result = cur.fetchone()
    conn.close()

    if not doc_result:
        return None, "Error: Active document not found."

    file_path, file_name = doc_result
    try:
        if file_path.endswith(".pdf"):
            return extract_text_from_pdf(file_path), file_name
        elif file_path.endswith(".docx"):
            return extract_text_from_docx(file_path), file_name
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read(), file_name
    except FileNotFoundError:
        return None, f"Error: File '{file_name}' not found on the server."

# For skipping terminal input while training
os.environ["WANDB_DISABLED"] = "true"

# -----------------------------------------------
# HUGGING FACE - AI Model Loading and Functions
# -----------------------------------------------
device = 0 if torch.cuda.is_available() else -1


# OPTIMIZATION: Cache the model loading functions. This is the single biggest performance improvement.
# The model will now be loaded only ONCE and reused across reruns.

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_summarizer(model_id="google/pegasus-cnn_dailymail"):
    """Loads the summarization model and tokenizer from Hugging Face."""
    try:
        st.info(f"Loading summarization model: {model_id}. This might take a moment...")
        summarizer_pipeline = pipeline("summarization", model=model_id, device=device)
        tokenizer = summarizer_pipeline.tokenizer
        return summarizer_pipeline, tokenizer
    except Exception as e:
        st.error(f"Failed to load summarization model {model_id}: {e}")
        return None, None

# OPTIMIZATION: Cache the model loading.
@st.cache_resource
def load_paraphraser(model_id="humarin/chatgpt_paraphraser_on_T5_base"):
    """Loads the paraphrasing model and tokenizer from Hugging Face."""
    try:
        st.info(f"Loading paraphrasing model: {model_id}. This might take a moment...")
        paraphraser_pipeline = pipeline("text2text-generation", model=model_id, device=device)
        tokenizer = paraphraser_pipeline.tokenizer
        return paraphraser_pipeline, tokenizer
    except Exception as e:
        st.error(f"Failed to load paraphrasing model {model_id}: {e}")
        return None, None

# OPTIMIZATION: Cache the model loading.
@st.cache_resource
def load_qa_model(model_id="distilbert-base-cased-distilled-squad"):
    """Loads a Question Answering model."""
    try:
        st.info(f"Loading Q&A model: {model_id}. This might take a moment...")
        qa_pipeline = pipeline("question-answering", model=model_id, device=device)
        return qa_pipeline
    except Exception as e:
        st.error(f"Failed to load Q&A model {model_id}: {e}")
        return None

# OPTIMIZATION: Cache the model loading.
@st.cache_resource
def load_text_generation_model(model_id="gpt2"):
    """Loads a Text Generation model."""
    try:
        st.info(f"Loading Text Generation model: {model_id}. This might take a moment...")
        text_gen_pipeline = pipeline("text-generation", model=model_id, device=device)
        return text_gen_pipeline
    except Exception as e:
        st.error(f"Failed to load Text Generation model {model_id}: {e}")
        return None

def get_keywords_and_entities(text):
    """Extracts keywords and named entities from text using SpaCy."""
    nlp = load_spacy_model()
    doc = nlp(text)
    keywords = [token.text.lower() for token in doc if token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and not token.is_stop]
    entities = [ent.text for ent in doc.ents]
    return list(set(keywords)), entities

def calculate_semantic_similarity(text1, text2):
    """Calculates meaning similarity between two texts using sentence transformers."""
    model = load_sentence_transformer()
    try:
        embedding1 = model.encode(text1, convert_to_tensor=True)
        embedding2 = model.encode(text2, convert_to_tensor=True)
        cosine_score = util.pytorch_cos_sim(embedding1, embedding2)
        return cosine_score.item() * 100 # Return as a percentage
    except Exception:
        return 0.0

def render_text_diff(text1, text2):
    """Generates an HTML view of the differences between two texts."""
    d = difflib.HtmlDiff(wrapcolumn=60)
    diff_html = d.make_table(text1.splitlines(), text2.splitlines(), "Original", "Paraphrase")
    # Basic styling to make it fit dark/light themes better
    diff_html = diff_html.replace(' nowrap="nowrap"', '')
    diff_html = diff_html.replace('<table', '<table style="width:100%; font-size: 0.9em;"')
    return diff_html

def create_gauge_chart(value, title, max_value, color):
    """Creates a Plotly bullet gauge chart."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        title = {'text': title, 'font': {'size': 16}},
        gauge = {
            'axis': {'range': [None, max_value], 'tickwidth': 1},
            'bar': {'color': color},
            'steps' : [
                {'range': [0, max_value * 0.5], 'color': 'lightgray'},
                {'range': [max_value * 0.5, max_value], 'color': 'darkgray'}],
        }
    ))
    fig.update_layout(height=150, margin=dict(l=20, r=20, t=40, b=10), paper_bgcolor='rgba(0,0,0,0)', font_color="#FAFAFA" if st.session_state.theme == 'dark' else '#31333F')
    return fig

def run_comparison_task(model_id, task_type, text, result_dict):
    """
    Target function for threading to run AI tasks and a comprehensive suite of analytics.
    """
    try:
        start_time = time.time()
        output_text = ""

        if task_type == "Summarization":
            pipeline, tokenizer = load_summarizer(model_id)
            if pipeline and tokenizer:
                output_text = summarize_text(pipeline, tokenizer, text)
        elif task_type == "Paraphrasing":
            pipeline, tokenizer = load_paraphraser(model_id)
            if pipeline and tokenizer:
                paraphrases = paraphrase_text(pipeline, tokenizer, text, num_return_sequences=1)
                output_text = paraphrases[0] if paraphrases else "No paraphrase generated."

        end_time = time.time()

        # --- GATHER ALL METRICS ---
        metrics = {}
        metrics['execution_time'] = end_time - start_time

        # Basic Text Stats
        metrics['word_count'] = len(output_text.split())
        metrics['char_count'] = len(output_text)

        # Readability Scores
        try:
            metrics['flesch_ease'] = textstat.flesch_reading_ease(output_text)
            metrics['flesch_grade'] = textstat.flesch_kincaid_grade(output_text)
        except:
            metrics['flesch_ease'], metrics['flesch_grade'] = 0, 0

        # ROUGE Scores (vs. Original)
        try:
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(text, output_text)
            metrics['rouge1'] = scores['rouge1'].fmeasure * 100
            metrics['rouge2'] = scores['rouge2'].fmeasure * 100
            metrics['rougeL'] = scores['rougeL'].fmeasure * 100
        except:
            metrics['rouge1'], metrics['rouge2'], metrics['rougeL'] = 0, 0, 0

        # Store results
        result_dict[model_id] = {
            "output": output_text,
            "metrics": metrics
        }
    except Exception as e:
        result_dict[model_id] = {"error": str(e)}

def answer_question(qa_pipeline, context, question):
    """Generates an answer from a context."""
    if not qa_pipeline:
        return "Q&A model not loaded."
    try:
        result = qa_pipeline(question=question, context=context)
        return result['answer']
    except Exception as e:
        st.error(f"Q&A failed: {e}")
        return "Error generating answer."


def generate_text(text_gen_pipeline, prompt, max_length=50):
    """Generates text from a prompt."""
    if not text_gen_pipeline:
        return "Text Generation model not loaded."
    try:
        result = text_gen_pipeline(prompt, max_length=max_length, num_return_sequences=1)
        return result[0]['generated_text']
    except Exception as e:
        st.error(f"Text generation failed: {e}")
        return "Error generating text."

def summarize_text(summarizer_pipeline, tokenizer, text, min_length=30, max_length=150):
    """Generates a summary for the given text using the model's tokenizer."""
    if not summarizer_pipeline or not tokenizer:
        return "Summarization model not loaded."

    max_model_input_length = 1000

    if len(text) > max_model_input_length:
        st.warning(f"Input text is too long ({len(text)} characters). Truncating to {max_model_input_length} characters for summarization.")
        text = text[:max_model_input_length]

    inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt", padding=True)

    if device != -1:
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

    try:
        summary_ids = summarizer_pipeline.model.generate(inputs["input_ids"], num_beams=5, max_length=max_length, min_length=min_length, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        st.error(f"Summarization failed: {e}")
        return "Error generating summary."


def paraphrase_text(paraphraser_pipeline, tokenizer, text, num_return_sequences=3):
    """Generates paraphrased versions of the given text using the model's tokenizer."""
    if not paraphraser_pipeline or not tokenizer:
        return ["Paraphrasing model not loaded."]

    max_model_input_length = 1000

    if len(text) > max_model_input_length:
         st.warning(f"Input text is too long ({len(text)} characters). Truncating to {max_model_input_length} characters for paraphrasing.")
         text = text[:max_model_input_length]

    inputs = tokenizer(f"paraphrase: {text}", max_length=1024, truncation=True, return_tensors="pt", padding=True)

    if device != -1:
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

    try:
        paraphrased_ids = paraphraser_pipeline.model.generate(
            inputs["input_ids"],
            num_beams=5,
            num_return_sequences=num_return_sequences,
            max_length=128,
            early_stopping=True
        )
        paraphrased_texts = [tokenizer.decode(g, skip_special_tokens=True) for g in paraphrased_ids]
        return paraphrased_texts
    except Exception as e:
        st.error(f"Paraphrasing failed: {e}")
        return ["Error generating paraphrases."]

# -----------------------------------------------
# VOICE AND LANGUAGE FUNCTIONS
# -----------------------------------------------

def audio_bytes_to_text(audio_data):
    if not audio_data or not isinstance(audio_data, dict) or 'bytes' not in audio_data:
        st.warning("No valid audio data received from microphone.")
        return ""

    audio_bytes = audio_data['bytes']
    sample_rate = audio_data.get('sample_rate', 16000)
    sample_width = audio_data.get('sample_width', 2)

    if not audio_bytes:
        st.warning("No audio bytes found in the recorded data.")
        return ""

    try:
        r = sr.Recognizer()
        audio_data_sr = sr.AudioData(audio_bytes, sample_rate, sample_width)

        text = r.recognize_google(audio_data_sr)
        return text
    except sr.UnknownValueError:
        st.warning("Speech Recognition could not understand audio")
        return ""
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service; {e}")
        return ""
    except Exception as e:
        st.error(f"An error occurred during speech processing: {e}")
        return ""


def text_to_speech(text):
    try:
        engine = pyttsx3.init()

        # Get available voices
        voices = engine.getProperty('voices')

        if not voices:
            st.warning("No text-to-speech voices found on this system.")
            return

        # Try to find an English voice
        selected_voice = None
        for voice in voices:
            # Check if voice has languages attribute and contains 'en'
            if hasattr(voice, 'languages') and voice.languages:
                if any('en' in lang for lang in voice.languages):
                    selected_voice = voice
                    break
            # Fallback: check voice ID for English indicators
            elif hasattr(voice, 'id') and ('en' in voice.id.lower() or 'english' in voice.id.lower()):
                selected_voice = voice
                break

        # If no English voice found, use the first available voice
        if selected_voice is None and voices:
            selected_voice = voices[0]
            st.info(f"Using available voice: {selected_voice.id}")

        if selected_voice:
            try:
                engine.setProperty('voice', selected_voice.id)
            except Exception as voice_error:
                st.warning(f"Could not set voice '{selected_voice.id}': {voice_error}")
                # Continue with default voice

        # Set speech properties
        engine.setProperty('rate', 150)  # Speed percent
        engine.setProperty('volume', 0.9)  # Volume 0-1

        engine.say(text)
        engine.runAndWait()

    except Exception as e:
        st.warning(f"Text-to-speech failed: {e}")
        st.info("Text-to-speech might not be fully supported in this environment. You may need to install additional voice packages on your system.")

def get_user_language(username):
    """Fetches the user's preferred language from the database."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT language FROM users WHERE username=?", (username,))
    result = cur.fetchone()
    conn.close()
    # Default to English if language is not set in the profile
    return result[0] if result and result[0] else "English"

# The detect_language function can remain as is, but it's not used by this new feature.
def detect_language(text):
    try:
        translator = Translator()
        detection = translator.detect(text)
        return detection.lang
    except:
        return "en"

def translate_text(text, dest_lang_name="English"):
    """
    Translates text for the UI, handling language names and codes.
    """
    if not text or not text.strip():
        return "No text provided to translate."
    try:
        # Get the language code (e.g., 'en') from the full name (e.g., 'English')
        dest_code = LANGUAGE_NAME_TO_CODE.get(dest_lang_name)
        if not dest_code:
            return f"Error: Could not find language code for '{dest_lang_name}'."

        translator = Translator()
        translation = translator.translate(text, dest=dest_code)
        return translation.text
    except Exception as e:
        st.error(f"Translation service failed: {e}")
        return "Error during translation."

# -----------------------------------------------
# FILE PROCESSING FUNCTIONS
# -----------------------------------------------

def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def extract_text_from_docx(file):
    try:
        doc = docx.Document(file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading Word document: {e}")
        return ""

# --- VVV ADD THESE NEW HELPER FUNCTIONS VVV ---

def augment_data_via_backtranslation(data, lang='de'):
    """Augments training data using back-translation."""
    augmented_data = []
    translator = Translator()
    st.info(f"Augmenting {len(data)} samples via back-translation. This may take a while...")
    progress_bar = st.progress(0)

    for i, (input_text, output_text) in enumerate(data):
        try:
            # Augment the input text
            translated_to = translator.translate(input_text, dest=lang).text
            back_translated_input = translator.translate(translated_to, dest='en').text
            augmented_data.append((back_translated_input, output_text))
        except Exception as e:
            st.warning(f"Skipping augmentation for one sample due to translation error: {e}")
            augmented_data.append((input_text, output_text)) # Add original if augmentation fails
        progress_bar.progress((i + 1) / len(data))

    st.success("Data augmentation complete!")
    return augmented_data

def compute_metrics_for_trainer(eval_preds):
    import numpy as np
    predictions, labels = eval_preds

    # If predictions are logits, convert them into token IDs
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    pred_ids = np.argmax(predictions, axis=-1)

    # Replace ignored index (-100) with pad token for decoding
    pred_ids = np.where(pred_ids != -100, pred_ids, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute ROUGE
    from rouge_score import rouge_scorer
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    result = {
        "rougeL": np.mean(
            [rouge.score(p, l)["rougeL"].fmeasure for p, l in zip(decoded_preds, decoded_labels)]
        )
    }
    return {k: round(v, 4) for k, v in result.items()}


# --- VVV ADD THESE NEW HELPER FUNCTIONS VVV ---

def log_agent_training_data(agent_model_id, training_data, source_info="User-provided"):
    """Hashes and logs the input text of training samples to prevent re-use."""
    conn = get_conn()
    cur = conn.cursor()
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    for input_text, _ in training_data:
        # Create a SHA256 hash of the input text to uniquely identify it
        data_hash = hashlib.sha256(str(input_text).encode()).hexdigest()
        cur.execute("""
            INSERT OR IGNORE INTO agent_training_data (agent_model_id, data_hash, source_info, timestamp)
            VALUES (?, ?, ?, ?)
        """, (agent_model_id, data_hash, source_info, now))
    conn.commit()
    conn.close()

def get_used_data_hashes(agent_model_id):
    """Fetches a set of all data hashes for a given agent for fast lookups."""
    conn = get_conn()
    df = pd.read_sql_query("SELECT data_hash FROM agent_training_data WHERE agent_model_id = ?", conn, params=(agent_model_id,))
    conn.close()
    return set(df['data_hash'])

# -----------------------------------------------
# DATABASE AND USER MANAGEMENT
# -----------------------------------------------

# REFINEMENT: Consolidate all table creation into a single function for cleaner startup.
def init_db():
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    # Updated 'users' table with new preference columns
    cur.execute("""CREATE TABLE IF NOT EXISTS agent_training_data (
                     agent_model_id TEXT NOT NULL,
                     data_hash TEXT NOT NULL,
                     source_info TEXT,
                     timestamp TEXT NOT NULL,
                     PRIMARY KEY (agent_model_id, data_hash))""")

    cur.execute("""CREATE TABLE IF NOT EXISTS users (
                     username TEXT PRIMARY KEY, name TEXT, email TEXT UNIQUE,
                     age_category TEXT, language TEXT DEFAULT 'English',
                     profile_pic BLOB, password TEXT, theme TEXT DEFAULT 'light',
                     is_active INTEGER DEFAULT 1, default_model TEXT,
                     reading_preferences TEXT, content_type TEXT, is_verified INTEGER DEFAULT 0, gemini_api_key TEXT, last_seen TEXT)""")

    # Updated 'submissions' table with new readability score columns
    cur.execute("""CREATE TABLE IF NOT EXISTS submissions (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     username TEXT,
                     timestamp TEXT,
                     model_id_used TEXT,
                     task_type TEXT,
                     input_text TEXT,
                     output_text TEXT,
                     sentiment TEXT,
                     flesch_reading_ease REAL,
                     flesch_kincaid_grade REAL,
                     gunning_fog REAL,
                     smog_index REAL,
                     rouge1 REAL,
                     rouge2 REAL,
                     rougeL REAL,
                     FOREIGN KEY(username) REFERENCES users(username))""")

    # 'models' table creation is now part of the main init function.
    cur.execute("""CREATE TABLE IF NOT EXISTS models (
                     model_id TEXT PRIMARY KEY,
                     display_name TEXT,
                     remarks TEXT,
                     status TEXT DEFAULT 'active',
                     task_type TEXT NOT NULL DEFAULT 'multi_task',
                     created_by_username TEXT,
                     is_public INTEGER DEFAULT 1,
                     eval_loss REAL,
                     eval_rougeL REAL,
                     eval_bleu REAL,
                     eval_bertscore REAL,
                     eval_edit_distance REAL,
                     training_progress REAL DEFAULT 0.0,
                     training_status_text TEXT,
                     agent_group_id TEXT,
                     version_number INTEGER,
                     is_active_version INTEGER DEFAULT 0)""")

    cur.execute("""CREATE TABLE IF NOT EXISTS smtp_settings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT,
                    password TEXT,
                    registration_msg TEXT,
                    reset_msg TEXT)""")

    cur.execute("""CREATE TABLE IF NOT EXISTS training_log (
                    submission_id INTEGER,
                    model_id TEXT,
                    timestamp TEXT,
                    PRIMARY KEY (submission_id, model_id),
                    FOREIGN KEY (submission_id) REFERENCES submissions(id) ON DELETE CASCADE)""")

    # --- VVV ADD THIS NEW TABLE DEFINITION VVV ---
    cur.execute("""CREATE TABLE IF NOT EXISTS broadcast_logs (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     timestamp TEXT NOT NULL,
                     sender_email TEXT NOT NULL,
                     recipient_username TEXT,
                     recipient_email TEXT NOT NULL,
                     subject TEXT,
                     message_body TEXT,
                     status TEXT NOT NULL)""")

    cur.execute("""CREATE TABLE IF NOT EXISTS tickets (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     timestamp TEXT NOT NULL,
                     username TEXT NOT NULL,
                     user_email TEXT NOT NULL,
                     user_description TEXT,
                     session_data TEXT,
                     status TEXT DEFAULT 'Open',
                     resolved_timestamp TEXT,
                     admin_reply_subject TEXT,
                     admin_reply_message TEXT,
                     FOREIGN KEY(username) REFERENCES users(username))""")

    cur.execute("""CREATE TABLE IF NOT EXISTS teams (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     team_name TEXT NOT NULL,
                     admin_username TEXT NOT NULL,
                     join_code TEXT UNIQUE NOT NULL,
                     created_at TEXT NOT NULL,
                     gemini_api_key TEXT,
                     default_model_id TEXT,
                     active_document_id INTEGER,
                     FOREIGN KEY(admin_username) REFERENCES users(username))""")

    cur.execute("""CREATE TABLE IF NOT EXISTS team_members (
                     team_id INTEGER NOT NULL,
                     username TEXT NOT NULL,
                     role TEXT NOT NULL DEFAULT 'member', -- 'admin' or 'member'
                     joined_at TEXT NOT NULL,
                     status TEXT NOT NULL DEFAULT 'active', -- 'active' or 'blocked'
                     PRIMARY KEY (team_id, username),
                     FOREIGN KEY(team_id) REFERENCES teams(id) ON DELETE CASCADE,
                     FOREIGN KEY(username) REFERENCES users(username) ON DELETE CASCADE)""")

    cur.execute("""CREATE TABLE IF NOT EXISTS team_chats (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     team_id INTEGER NOT NULL,
                     username TEXT NOT NULL,
                     message TEXT NOT NULL,
                     timestamp TEXT NOT NULL,
                     message_type TEXT DEFAULT 'user', -- 'user' or 'ai'
                     FOREIGN KEY(team_id) REFERENCES teams(id) ON DELETE CASCADE,
                     FOREIGN KEY(username) REFERENCES users(username))""")

    cur.execute("""CREATE TABLE IF NOT EXISTS team_documents (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     team_id INTEGER NOT NULL,
                     uploader_username TEXT NOT NULL,
                     file_name TEXT NOT NULL,
                     file_path TEXT NOT NULL,
                     uploaded_at TEXT NOT NULL,
                     FOREIGN KEY(team_id) REFERENCES teams(id) ON DELETE CASCADE,
                     FOREIGN KEY(uploader_username) REFERENCES users(username))""")

    cur.execute("""CREATE TABLE IF NOT EXISTS friends (
                     user1_username TEXT NOT NULL,
                     user2_username TEXT NOT NULL,
                     status TEXT NOT NULL, -- 'pending', 'accepted', 'declined', 'blocked'
                     action_user_username TEXT NOT NULL,
                     initial_message TEXT,
                     shared_key_by_user1 TEXT,
                     shared_key_by_user2 TEXT,
                     PRIMARY KEY (user1_username, user2_username),
                     FOREIGN KEY(user1_username) REFERENCES users(username),
                     FOREIGN KEY(user2_username) REFERENCES users(username))""")

    cur.execute("""CREATE TABLE IF NOT EXISTS direct_messages (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     sender_username TEXT NOT NULL,
                     recipient_username TEXT NOT NULL,
                     message TEXT NOT NULL,
                     timestamp TEXT NOT NULL,
                     message_type TEXT DEFAULT 'user', -- 'user' or 'ai'
                     FOREIGN KEY(sender_username) REFERENCES users(username),
                     FOREIGN KEY(recipient_username) REFERENCES users(username))""")

    try:
        cur.execute("ALTER TABLE friends ADD COLUMN shared_key_by_user1 TEXT")
        cur.execute("ALTER TABLE friends ADD COLUMN shared_key_by_user2 TEXT")
        cur.execute("ALTER TABLE models ADD COLUMN training_progress REAL DEFAULT 0.0")
        cur.execute("ALTER TABLE models ADD COLUMN training_status_text TEXT")
        cur.execute("ALTER TABLE models ADD COLUMN agent_group_id TEXT")
        cur.execute("ALTER TABLE models ADD COLUMN version_number INTEGER")
        cur.execute("ALTER TABLE models ADD COLUMN is_active_version INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass

    conn.commit()
    conn.close()
    print("Checked/Created all necessary tables in users.db")

# --- VVV ADD THESE NEW HELPER FUNCTIONS VVV ---

def get_user_agent_groups(username):
    """Fetches the distinct agent groups for a user."""
    conn = get_conn()
    # An agent group is defined by having the same agent_group_id
    df = pd.read_sql_query("""
        SELECT agent_group_id, display_name, task_type
        FROM models
        WHERE created_by_username = ? AND version_number = 1
        ORDER BY display_name
    """, conn, params=(username,))
    conn.close()
    return df

def get_agent_versions(agent_group_id):
    """Fetches all versions for a specific agent group."""
    conn = get_conn()
    df = pd.read_sql_query("""
        SELECT * FROM models
        WHERE agent_group_id = ?
        ORDER BY version_number DESC
    """, conn, params=(agent_group_id,))
    conn.close()
    return df

def activate_model_version(agent_group_id, version_model_id):
    """Activates a specific model version and deactivates all others in the same agent group."""
    conn = get_conn()
    cur = conn.cursor()
    # Deactivate all versions in this group
    cur.execute("UPDATE models SET is_active_version = 0 WHERE agent_group_id = ?", (agent_group_id,))
    # Activate the selected version
    cur.execute("UPDATE models SET is_active_version = 1 WHERE model_id = ?", (version_model_id,))
    conn.commit()
    conn.close()
    st.success("Agent version activated!")

def delete_model_version(model_id_to_delete):
    """Deletes a single version of a model, including its files."""
    conn = get_conn()
    cur = conn.cursor()
    # Prevent deleting the last version of an agent
    cur.execute("SELECT agent_group_id FROM models WHERE model_id = ?", (model_id_to_delete,))
    result = cur.fetchone()
    if not result:
        st.error("Model not found.")
        conn.close()
        return

    agent_group_id = result[0]
    cur.execute("SELECT COUNT(*) FROM models WHERE agent_group_id = ?", (agent_group_id,))
    version_count = cur.fetchone()[0]

    if version_count <= 1:
        st.warning("Cannot delete the only version of an agent. To remove it completely, use the main 'Delete Agent' button.")
        conn.close()
        return

    cur.execute("DELETE FROM models WHERE model_id = ?", (model_id_to_delete,))
    conn.commit()
    conn.close()

    # Also delete the model files from the server
    if os.path.isdir(model_id_to_delete):
        shutil.rmtree(model_id_to_delete)
    st.success(f"Version {os.path.basename(model_id_to_delete)} deleted successfully.")

def delete_agent_group(agent_group_id):
    """Deletes all versions of an agent and all associated files."""
    versions_df = get_agent_versions(agent_group_id)
    for _, version in versions_df.iterrows():
        model_path = version['model_id']
        if os.path.isdir(model_path):
            # The model path is .../<agent_name>/v<version>, so we delete the parent <agent_name> dir
            agent_dir = os.path.dirname(os.path.dirname(model_path))
            if os.path.exists(agent_dir):
                shutil.rmtree(agent_dir)
            break # Only need to do this once

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM models WHERE agent_group_id = ?", (agent_group_id,))
    conn.commit()
    conn.close()
    st.success("Agent and all its versions have been deleted.")

# --- VVV ADD THIS NEW GLOBAL FUNCTION HERE VVV ---
def get_conn(db_path="users.db"):
    """Establishes a connection to the SQLite database."""
    return sqlite3.connect(db_path, check_same_thread=False)

# -----------------------------------------------
# GOOGLE GEMINI FUNCTIONS
# -----------------------------------------------
import google.generativeai as genai

# Helper function to configure and get the Gemini model
def get_gemini_model(api_key):
    """Configures the Gemini API and returns a model object."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        return model
    except Exception as e:
        st.error(f"Error configuring Gemini: {e}")
        return None

def summarize_with_gemini(api_key, text, length_instruction="concise"):
    """Generates a summary using Gemini with a specific length instruction."""
    model = get_gemini_model(api_key)
    if not model:
        return "Failed to load Gemini model. Please check your API key."

    prompt = f"Please provide a {length_instruction} summary of the following text:\n\n{text}"
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini summarization failed: {e}"

def answer_question_with_gemini(api_key, context, question):
    """Answers a question based on a context using Gemini."""
    model = get_gemini_model(api_key)
    if not model:
        return "Failed to load Gemini model. Please check your API key."

    prompt = f"Based on the context below, answer the question.\n\nContext: {context}\n\nQuestion: {question}"
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini Q&A failed: {e}"

def generate_text_with_gemini(api_key, prompt_text, max_length=150):
    """Generates text from a prompt using Gemini."""
    model = get_gemini_model(api_key)
    if not model:
        return "Failed to load Gemini model. Please check your API key."

    try:
        # Note: Gemini's max_length is controlled differently, but we can instruct it in the prompt.
        response = model.generate_content(f"{prompt_text} (Please keep the response under {max_length} words)")
        return response.text
    except Exception as e:
        return f"Gemini text generation failed: {e}"

# Helper functions for database interaction
def get_user_gemini_key(username):
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("SELECT gemini_api_key FROM users WHERE username=?", (username,))
    result = cur.fetchone()
    conn.close()
    return result[0] if result and result[0] else ""

def update_user_gemini_key(username, api_key):
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("UPDATE users SET gemini_api_key=? WHERE username=?", (api_key, username))
    conn.commit()
    conn.close()

def clear_form_states():
    """Resets the states for the main input form."""
    st.session_state.manual_text = ""
    st.session_state.qa_question = ""
    st.session_state.readability_scores = None
    st.toast("Inputs cleared!")

# Add this line with the other session state initializations
if "comparison_scores" not in st.session_state:
    st.session_state.comparison_scores = None

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# Initialize database schema
init_db()

# Insert a test user and default models
def insert_initial_data():
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()

    # Insert test user
    cur.execute("SELECT * FROM users WHERE username=?", ("test",))
    if not cur.fetchone():
        cur.execute("INSERT INTO users (username, name, email, age_category, language, password, theme, default_model) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    ("test", "Test User", "test@test.com", "15 - 20", "English", hash_password("test"), "light", "google/pegasus-cnn_dailymail"))
        conn.commit()

    # Insert default models if not exist
    default_models = [
        # --- Existing Models ---
        ('gemini-1.5-flash', 'Google Gemini', 'Multi-task model powered by Google Gemini Pro', 'multi_task',),
        ("google/pegasus-cnn_dailymail", "Pegasus CNN/DailyMail (Summarization)", "Good for news articles", "summarization"),
        ("humarin/chatgpt_paraphraser_on_T5_base", "ChatGPT Paraphraser (Paraphrasing)", "Effective for rephrasing text", "paraphrasing"),
        ("distilbert-base-cased-distilled-squad", "DistilBERT (Q&A)", "Fast and accurate for question answering", "question_answering"),
        ("gpt2", "GPT-2 (Text Generation)", "Generates creative and coherent text", "text_generation"),

        # --- VVV NEW PARAPHRASING MODELS VVV ---
        ("Vamsi/T5_Paraphrase_Paws", "T5 PAWS (Paraphrasing)", "T5 fine-tuned on the PAWS dataset for paraphrase generation.", "paraphrasing"),
        ("ramsrigouthamg/t5_paraphraser", "T5 Paraphraser (Rewrite)", "T5 model for rewriting and paraphrasing sentences.", "paraphrasing"),
        ("agentlans/flan-t5-small-simplifier", "FLAN-T5-Small (Simplifier)", "FLAN-T5-Small for both simplification & paraphrasing.", "paraphrasing"),

        # --- VVV NEW SUMMARIZATION MODELS VVV ---
        ("sshleifer/distilbart-cnn-12-6", "DistilBART CNN (Summarization)", "A distilled version of BART for fast news summarization.", "summarization"),
        ("weijiahaha/t5-small-summarization", "T5-Small CNN (Summarization)", "A T5-small model fine-tuned on CNN/DailyMail.", "summarization"),
        ("cnicu/t5-small-booksum", "T5-Small BookSum (Summarization)", "T5-small fine-tuned on the BookSum dataset for longer text.", "summarization"),
    ]
    for model_id, display_name, remarks, task_type in default_models:
        cur.execute("SELECT * FROM models WHERE model_id=?", (model_id,))
        if not cur.fetchone():
            cur.execute("INSERT INTO models (model_id, display_name, remarks, status, task_type) VALUES (?, ?, ?, ?, ?)",
                        (model_id, display_name, remarks, 'active', task_type))
            conn.commit()

    conn.close()

insert_initial_data()


def valid_username(username):
    return re.match(r"^[A-Za-z][A-Za-z0-9_]*$", username)

def valid_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)

def export_pdf(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    for i, row in df.iterrows():
        # Safely access all the columns from the full DataFrame
        ts = pd.to_datetime(row['timestamp']).strftime('%Y-%m-%d %H:%M')
        task = row.get('task_type', 'N/A')
        sentiment = row.get('sentiment', 'N/A')
        grade = row.get('flesch_kincaid_grade', 0.0)

        input_text = row['input_text'][:200] + '...' if len(row['input_text']) > 200 else row['input_text']
        output_text = row['output_text'][:200] + '...' if len(row['output_text']) > 200 else row['output_text']

        # Create the text block for the PDF entry
        text_content = (
            f"Timestamp: {ts} | Task: {task}\n"
            f"Sentiment: {sentiment} | Readability Grade: {grade:.2f}\n"
            f"------------------------------------------------------------------\n"
            f"Input Text:\n{input_text}\n\n"
            f"Output Text:\n{output_text}"
        )

        pdf.multi_cell(0, 6, text_content.encode('latin-1', 'replace').decode('latin-1'), border=1)
        pdf.cell(0, 5, ln=True) # Add a space between entries

    return pdf.output(dest="S").encode("latin-1")

@st.cache_resource
def get_rouge():
    return load("rouge")

def compute_rouge(original_text, generated_text):
    rouge = get_rouge()
    results = rouge.compute(predictions=[generated_text], references=[original_text])
    return results

def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1: return "Positive"
    elif polarity < -0.1: return "Negative"
    else: return "Neutral"

def image_to_binary(image):
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    return buf.getvalue()

def binary_to_image(binary_data):
    return Image.open(io.BytesIO(binary_data))

def get_user_theme(username):
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("SELECT theme FROM users WHERE username=?", (username,))
    result = cur.fetchone()
    conn.close()
    return result[0] if result else "light"

def update_user_theme(username, theme):
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("UPDATE users SET theme=? WHERE username=?", (theme, username))
    conn.commit()
    conn.close()

def is_user_active(username):
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("SELECT is_active FROM users WHERE username=?", (username,))
    result = cur.fetchone()
    conn.close()
    return result[0] if result else 1

def toggle_user_status(username, status):
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("UPDATE users SET is_active=? WHERE username=?", (status, username))
    conn.commit()
    conn.close()

def delete_user(username):
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("DELETE FROM submissions WHERE username=?", (username,))
    cur.execute("DELETE FROM users WHERE username=?", (username,))
    conn.commit()
    conn.close()

# --- VVV ADD THIS NEW FUNCTION HERE VVV ---
def get_all_users_for_broadcast():
    """Fetches all user data as a DataFrame for the broadcast feature."""
    conn = sqlite3.connect("users.db")
    df = pd.read_sql_query("SELECT * FROM users", conn)
    conn.close()
    return df

def log_broadcast(sender, recipient, subject, message, status):
    """Logs a broadcast email event to the database."""
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO broadcast_logs
        (timestamp, sender_email, recipient_username, recipient_email, subject, message_body, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        sender,
        recipient.get('username', 'N/A'),
        recipient.get('email', 'N/A'),
        subject,
        message,
        status
    ))
    conn.commit()
    conn.close()

# --- VVV ADD THESE NEW FRIEND HELPER FUNCTIONS VVV ---

def get_all_other_users(current_username):
    """Fetches all users except the current one."""
    conn = sqlite3.connect("users.db")
    df = pd.read_sql_query("SELECT username, name, email FROM users WHERE username != ?", conn, params=(current_username,))
    conn.close()
    return df

def get_friendship(user1, user2):
    """Fetches the friendship status between two users."""
    # Ensure consistent order for querying
    u1, u2 = sorted([user1, user2])
    conn = sqlite3.connect("users.db")
    df = pd.read_sql_query("SELECT * FROM friends WHERE user1_username=? AND user2_username=?", conn, params=(u1, u2))
    conn.close()
    return df.iloc[0] if not df.empty else None

def send_friend_request(sender, recipient, initial_message):
    """Creates a new 'pending' friendship request."""
    u1, u2 = sorted([sender, recipient])
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO friends (user1_username, user2_username, status, action_user_username, initial_message)
            VALUES (?, ?, 'pending', ?, ?)
        """, (u1, u2, sender, initial_message))
        conn.commit()
    except sqlite3.IntegrityError:
        pass # Request already exists
    finally:
        conn.close()

def update_friend_request(user1, user2, status):
    """Accepts or declines a friend request."""
    u1, u2 = sorted([user1, user2])
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("UPDATE friends SET status=? WHERE user1_username=? AND user2_username=?", (status, u1, u2))
    conn.commit()
    conn.close()

def get_friend_requests(username):
    """Fetches pending friend requests for a user."""
    conn = sqlite3.connect("users.db")
    query = "SELECT * FROM friends WHERE (user1_username=? OR user2_username=?) AND status='pending' AND action_user_username != ?"
    df = pd.read_sql_query(query, conn, params=(username, username, username))
    conn.close()
    return df

def get_friends_list(username):
    """Fetches all accepted friends for a user."""
    conn = sqlite3.connect("users.db")
    query = """
        SELECT CASE
                   WHEN user1_username = ? THEN user2_username
                   ELSE user1_username
               END as friend_username
        FROM friends
        WHERE (user1_username = ? OR user2_username = ?) AND status = 'accepted'
    """
    df_friends = pd.read_sql_query(query, conn, params=(username, username, username))

    if df_friends.empty:
        conn.close()
        return pd.DataFrame()

    # Get details for the friends
    friend_usernames = tuple(df_friends['friend_username'].tolist())
    query_details = f"SELECT username, name, profile_pic, last_seen FROM users WHERE username IN ({','.join(['?']*len(friend_usernames))})"
    df_details = pd.read_sql_query(query_details, conn, params=friend_usernames)
    conn.close()
    return df_details

def get_direct_messages(user1, user2):
    """Fetches the chat history between two users."""
    conn = get_conn()
    # CORRECTED: Changed 's.username' to 's.sender_username' in the SELECT statement
    query = """
        SELECT s.sender_username,
               COALESCE(u.name, s.sender_username) as display_name,
               s.message, s.timestamp, s.message_type
        FROM direct_messages s
        LEFT JOIN users u ON s.sender_username = u.username
        WHERE (s.sender_username = ? AND s.recipient_username = ?)
           OR (s.sender_username = ? AND s.recipient_username = ?)
        ORDER BY s.timestamp ASC
    """
    df = pd.read_sql_query(query, conn, params=(user1, user2, user2, user1))
    conn.close()
    return df

def post_direct_message(sender, recipient, message, message_type='user'):
    """Saves a new plain-text direct message to the database."""
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # ✅ Store only raw text
    cur.execute("""
        INSERT INTO direct_messages (sender_username, recipient_username, message, timestamp, message_type)
        VALUES (?, ?, ?, ?, ?)
    """, (sender, recipient, message, now, message_type))

    conn.commit()
    conn.close()

def update_api_key_sharing(sharing_user, other_user, should_share):
    """Updates the API key sharing status in the friends table."""
    u1, u2 = sorted([sharing_user, other_user])
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()

    key_to_share = get_user_gemini_key(sharing_user) if should_share else None

    if sharing_user == u1:
        # The sharing user is user1, so update user1's shared key column
        update_col = "shared_key_by_user1"
    else:
        # The sharing user is user2, so update user2's shared key column
        update_col = "shared_key_by_user2"

    cur.execute(f"UPDATE friends SET {update_col}=? WHERE user1_username=? AND user2_username=?",
                (key_to_share, u1, u2))
    conn.commit()
    conn.close()

def get_effective_api_key(current_user, chat_partner):
    """Determines which API key to use: the partner's shared key or the user's own."""
    friendship = get_friendship(current_user, chat_partner)
    if friendship is None:
        return get_user_gemini_key(current_user)

    u1, u2 = sorted([current_user, chat_partner])

    # Check if the chat partner has shared a key with the current user
    if current_user == u1: # Current user is user1, so partner is user2
        partner_shared_key = friendship.get('shared_key_by_user2')
        if partner_shared_key:
            return partner_shared_key
    else: # Current user is user2, so partner is user1
        partner_shared_key = friendship.get('shared_key_by_user1')
        if partner_shared_key:
            return partner_shared_key

    # If no key is shared by the partner, fall back to the current user's own key
    return get_user_gemini_key(current_user)

def update_friendship_status(user1, user2, status, action_user):
    """Updates a friendship status (accept, decline, block)."""
    u1, u2 = sorted([user1, user2])
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("UPDATE friends SET status=?, action_user_username=? WHERE user1_username=? AND user2_username=?",
                (status, action_user, u1, u2))
    conn.commit()
    conn.close()

def remove_friend(user1, user2):
    """Removes a friendship entry entirely."""
    u1, u2 = sorted([user1, user2])
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("DELETE FROM friends WHERE user1_username=? AND user2_username=?", (u1, u2))
    conn.commit()
    conn.close()

def get_profile_avatar(user_row):
    """Generates an HTML avatar for a user."""
    # Check for a profile picture
    if user_row.get('profile_pic') and isinstance(user_row['profile_pic'], bytes):
        b64_img = base64.b64encode(user_row['profile_pic']).decode()
        return f'<img class="chat-avatar" src="data:image/png;base64,{b64_img}">'
    else:
        # Create a default avatar with user's initials
        initials = "".join([name[0] for name in user_row.get('name', '..').split()[:2]]).upper()
        return f'<div class="chat-avatar default-avatar">{initials}</div>'

def get_user_email_by_username(username):
    """Fetches a user's email from their username."""
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("SELECT email FROM users WHERE username=?", (username,))
    result = cur.fetchone()
    conn.close()
    return result[0] if result else None

def create_ticket(username, description, session_data_dict):
    """Creates a new support ticket in the database."""
    user_email = get_user_email_by_username(username)
    if not user_email:
        return False

    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO tickets (timestamp, username, user_email, user_description, session_data, status)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        username,
        user_email,
        description,
        json.dumps(session_data_dict, indent=2), # Store session data as a JSON string
        'Open'
    ))
    conn.commit()
    conn.close()
    return True

def resolve_ticket(ticket_id, reply_subject, reply_message):
    """Marks a ticket as resolved and logs the admin's reply."""
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("""
        UPDATE tickets
        SET status = 'Resolved',
            resolved_timestamp = ?,
            admin_reply_subject = ?,
            admin_reply_message = ?
        WHERE id = ?
    """, (
        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        reply_subject,
        reply_message,
        ticket_id
    ))
    conn.commit()
    conn.close()

def update_password(username, new_password):
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("UPDATE users SET password=? WHERE username=?", (hash_password(new_password), username))
    conn.commit()
    conn.close()

def get_active_models():
    conn = sqlite3.connect("users.db")
    df = pd.read_sql_query("SELECT model_id, display_name, remarks FROM models WHERE status='active'", conn)
    conn.close()
    return df

def get_all_models():
    conn = sqlite3.connect("users.db")
    df = pd.read_sql_query("SELECT model_id, display_name, remarks, status FROM models", conn)
    conn.close()
    return df

def add_model(model_id, display_name, remarks, status):
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO models (model_id, display_name, remarks, status) VALUES (?, ?, ?, ?)",
                    (model_id, display_name, remarks, status))
        conn.commit()
        st.success(f"Model '{display_name}' added successfully.")
    except sqlite3.IntegrityError:
        st.error(f"Model ID '{model_id}' already exists.")
    conn.close()

def update_model(model_id, display_name, remarks, status):
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("UPDATE models SET display_name=?, remarks=?, status=? WHERE model_id=?",
                (display_name, remarks, status, model_id))
    conn.commit()
    st.success(f"Model '{display_name}' updated successfully.")
    conn.close()

def delete_model(model_id):
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("DELETE FROM models WHERE model_id=?", (model_id,))
    conn.commit()
    st.success(f"Model '{model_id}' deleted successfully.")
    conn.close()

# -----------------------------------------------
# UI ENHANCEMENTS
# -----------------------------------------------

# OPTIMIZATION: Cache the Lottie file loading to prevent re-downloading.
@st.cache_data
def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

lottie_welcome = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_vyLwnL.json")
lottie_login = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_kcs1arba.json")
lottie_analytics = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_vybwn7df.json")
lottie_voice = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_ttvteyse.json")
lottie_models = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_yisb3l0x.json")

def custom_header():
    """Creates a custom, consistent header for all pages."""
    # Ensure user is logged in to fetch data
    if st.session_state.user:
        conn = sqlite3.connect("users.db")
        user_data = pd.read_sql_query("SELECT name, profile_pic FROM users WHERE username=?", conn, params=(st.session_state.user,))
        conn.close()

        user_name = user_data.iloc[0]['name']
        profile_pic_binary = user_data.iloc[0]['profile_pic']

        # Custom CSS for the header
        st.markdown("""
            <style>
            .header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 0.5rem 1rem;
                border-bottom: 1px solid #333;
                width: 100%;
            }
            .header-left {
                font-size: 1.5rem;
                font-weight: bold;
            }
            .header-right {
                display: flex;
                align-items: center;
                gap: 1rem;
            }
            .profile-avatar-header {
                width: 40px;
                height: 40px;
                border-radius: 50%;
                object-fit: cover;
                border: 2px solid #555;
            }
            </style>
        """, unsafe_allow_html=True)

        # Header Layout
        title_col, nav_col = st.columns([3, 2])

        with title_col:
            st.markdown(f'<span class="header-left">Welcome back, {user_name}!</span>', unsafe_allow_html=True)

        with nav_col:
            # Adjusted column ratios for better spacing
            cols = st.columns([1.2, 1.2, 1.2, 1.2, 0.5])
            with cols[0]:
                # Renamed "Dashboard" to "Profile"
                if st.button("👤 Profile", use_container_width=True):
                    st.session_state.page = "dashboard"
                    st.rerun()
            with cols[1]:
                if st.button("🤝 Friends", use_container_width=True):
                    st.session_state.page = "friends_hub"
                    st.rerun()
            with cols[2]:
                if st.button("🏢 Teams", use_container_width=True):
                    st.session_state.page = "teams"
                    st.rerun()
            with cols[3]:
                if st.button("Logout", use_container_width=True):
                    st.session_state.user = None
                    st.session_state.page = "login"
                    st.rerun()
            with cols[4]:
                 if profile_pic_binary:
                    image_b64 = base64.b64encode(profile_pic_binary).decode()
                    st.markdown(f'<img src="data:image/png;base64,{image_b64}" class="profile-avatar-header">', unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)


def apply_theme(theme):
    if theme == "dark":
        st.markdown(f"""
        <style>
        .main {{
            background-color: #0E1117;
            color: #FAFAFA;
        }}
        .main-header {{
            font-size: 2.5rem;
            color: #FF4B4B;
            text-align: center;
            margin-bottom: 1.5rem;
        }}
        .sub-header {{
            font-size: 1.5rem;
            color: #FF4B4B;
            border-bottom: 2px solid #FF4B4B;
            padding-bottom: 0.4rem;
            margin-top: 1.5rem;
            margin-bottom: 0.8rem;
        }}
        .feature-card {{
            background: rgba(30, 30, 30, 0.7);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 1.2rem;
            box-shadow: 0 6px 24px rgba(0, 0, 0, 0.3);
            margin-bottom: 1.2rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .stButton>button {{
            width: 100%;
            border-radius: 18px;
            border: 2px solid #FF4B4B;
            background-color: #FF4B4B;
            color: white;
            padding: 8px 20px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.3s;
        }}
        .stButton>button:hover {{
            background-color: #0E1117;
            color: #FF4B4B;
        }}
        .secondary-button {{
            background-color: #262730 !important;
            color: #FF4B4B !important;
            border: 2px solid #FF4B4B !important;
        }}
        .secondary-button:hover {{
            background-color: #FF4B4B !important;
            color: white !important;
        }}
        .text-input {{
            border-radius: 10px;
            background-color: #262730;
            color: white;
        }}
        .metric-card {{
            background: rgba(30, 30, 30, 0.7);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 1.2rem;
            box-shadow: 0 6px 24px rgba(0, 0, 0, 0.3);
            text-align: center;
            margin-bottom: 1.2rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .metric-value {{
            font-size: 2rem;
            font-weight: bold;
            color: #FF4B4B;
        }}
        .metric-label {{
            font-size: 0.9rem;
            color: #CCCCCC;
        }}
        .profile-avatar {{
            border-radius: 50%;
            border: 2px solid #FF4B4B;
            box_shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            object-fit: cover;
        }}
        .section-spacing {{
            margin-top: 0.8rem;
            margin-bottom: 0.8rem;
        }}
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <style>
        .main {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            color: #31333F;
        }}
        .main-header {{
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1.5rem;
        }}
        .sub-header {{
            font-size: 1.5rem;
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 0.4rem;
            margin-top: 1.5rem;
            margin-bottom: 0.8rem;
        }}
        .feature-card {{
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 1.2rem;
            box-shadow: 0 6px 24px rgba(0, 0, 0, 0.1);
            margin-bottom: 1.2rem;
            border: 1px solid rgba(255, 255, 255, 0.5);
        }}
        .stButton>button {{
            width: 100%;
            border-radius: 18px;
            border: 2px solid #4CAF50;
            background-color: #4CAF50;
            color: white;
            padding: 8px 20px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.3s;
        }}
        .stButton>button:hover {{
            background-color: white;
            color: #4CAF50;
        }}
        .secondary-button {{
            background-color: #f0f2f6 !important;
            color: #1f77b4 !important;
            border: 2px solid #1f77b4 !important;
        }}
        .secondary-button:hover {{
            background-color: #1f77b4 !important;
            color: white !important;
        }}
        .text-input {{
            border-radius: 10px;
        }}
        .metric-card {{
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 1.2rem;
            box_shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            text-align: center;
            margin-bottom: 1.2rem;
            border: 1px solid rgba(255, 255, 255, 0.5);
        }}
        .metric-value {{
            font-size: 2rem;
            font-weight: bold;
            color: #1f77b4;
        }}
        .metric-label {{
            font-size: 0.9rem;
            color: #7f8c8d;
        }}
        .profile-avatar {{
            border-radius: 50%;
            border: 2px solid #1f77b4;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            object-fit: cover;
        }}
        .section-spacing {{
            margin-top: 0.8rem;
            margin-bottom: 0.8rem;
        }}
        </style>
        """, unsafe_allow_html=True)

def animate_transition():
    with st.spinner("Loading..."):
        time.sleep(0.5)

# -----------------------------------------------
# STREAMLIT APPLICATION UI
# -----------------------------------------------
st.set_page_config(page_title="TextMorph App", layout="wide", page_icon="📝")

# REFINEMENT: Centralize all session state initializations here for clarity and better management.
if "page" not in st.session_state:
    st.session_state.page = "login"
if "user" not in st.session_state:
    st.session_state.user = None
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "voice_input_text" not in st.session_state:
    st.session_state.voice_input_text = ""
if "generated_summary" not in st.session_state:
    st.session_state.generated_summary = ""
if "generated_paraphrases" not in st.session_state:
    st.session_state.generated_paraphrases = []
if "current_summary_model_id" not in st.session_state:
    st.session_state.current_summary_model_id = None
if "current_paraphrase_model_id" not in st.session_state:
    st.session_state.current_paraphrase_model_id = None
if "summarizer_pipeline" not in st.session_state:
    st.session_state.summarizer_pipeline = None
if "summarizer_tokenizer" not in st.session_state:
    st.session_state.summarizer_tokenizer = None
if "paraphraser_pipeline" not in st.session_state:
    st.session_state.paraphraser_pipeline = None
if "paraphraser_tokenizer" not in st.session_state:
    st.session_state.paraphraser_tokenizer = None
if "readability_scores" not in st.session_state:
    st.session_state.readability_scores = None
if "manual_text" not in st.session_state:
    st.session_state.manual_text = ""
if "qa_question" not in st.session_state:
    st.session_state.qa_question = ""
if "otp_verified" not in st.session_state:
    st.session_state.otp_verified = False
if "comparison_results" not in st.session_state:
    st.session_state.comparison_results = None
if st.session_state.page != "dashboard" and "comparison_results" in st.session_state:
    st.session_state.comparison_results = None

if st.session_state.user:
    user_theme = get_user_theme(st.session_state.user)
    st.session_state.theme = user_theme

apply_theme(st.session_state.theme)

AGE_CATEGORIES = [f"{i} - {i+5}" for i in range(15, 76, 5)]

LANGUAGE_NAME_TO_CODE = {name.capitalize(): code for code, name in LANGUAGES.items()}
LANGUAGE_NAMES = sorted(list(LANGUAGE_NAME_TO_CODE.keys()))

# ---------- Login Page ----------
if st.session_state.page == "login":
    # NEW: A modern, centered layout
    st.markdown('<h1 class="main-header" style="text-align: center;">TextMorph</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.1rem; color: #CCCCCC;">Intelligent Text Transformation at Your Fingertips.</p>', unsafe_allow_html=True)

    # Use columns to center the login form
    _, login_col, _ = st.columns([1, 1.5, 1])

    with login_col:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)

        # Display Lottie animation inside the card for visual appeal
        if lottie_login:
            st_lottie(lottie_login, height=200, key="login_lottie")

        st.header("User Login")
        username = st.text_input("Username", help="Enter your username", label_visibility="collapsed", placeholder="Username", key="login_username")
        password = st.text_input("Password", type="password", help="Enter your password", label_visibility="collapsed", placeholder="Password", key="login_password")

        if st.button("Login", key="login_btn", use_container_width=True, type="primary"):
            # IMPORTANT SECURITY NOTE: Hardcoded admin credentials are not recommended for production.
            # Consider using a 'role' column in the database instead.
            if (username == "shashi" and password == "26092004") or (username == "pranavi" and password == "20112003"):
                st.session_state.user = username
                st.session_state.page = "admin"
                st.success("Admin login successful")
                animate_transition()
                st.rerun()
            else:
                if not is_user_active(username):
                    st.error("This account has been blocked. Please contact an administrator.")
                else:
                    conn = sqlite3.connect("users.db")
                    cur = conn.cursor()
                    cur.execute("SELECT * FROM users WHERE username=? AND password=?",
                                (username, hash_password(password)))
                    user = cur.fetchone()
                    conn.close()
                    if user:
                        st.session_state.user = username
                        st.session_state.page = "home"
                        st.success("Login successful")
                        animate_transition()
                        st.rerun()
                    else:
                        st.error("Invalid username or password")

        st.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)

        # Secondary action buttons
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Register", key="register_btn", use_container_width=True):
                st.session_state.page = "register"
                animate_transition()
                st.rerun()
        with col_b:
            if st.button("Reset Password", key="forgot_btn", use_container_width=True):
                st.session_state.page = "forgot"
                animate_transition()
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ---------- Register Page ----------
elif st.session_state.page == "register":
    st.markdown('<h1 class="main-header" style="text-align: center;">Create an Account</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.1rem; color: #CCCCCC;">Join us and start transforming your text.</p>', unsafe_allow_html=True)

    _, register_col, _ = st.columns([1, 1.8, 1])

    with register_col:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)

        with st.form("registration_form"):
            st.subheader("Personal Details")
            name = st.text_input("Full Name", placeholder="Enter your full name")
            email = st.text_input("Email", placeholder="Enter your email address")
            age_category = st.selectbox("Select Age Category", AGE_CATEGORIES)

            st.markdown("---")
            st.subheader("Account Credentials")
            username = st.text_input("Username", placeholder="Choose a unique username")
            password = st.text_input("Password", type="password", placeholder="Create a strong password")
            verify_password = st.text_input("Verify Password", type="password", placeholder="Re-enter your password")

            st.markdown("---")
            st.subheader("Profile Picture (Optional)")
            profile_pic = st.file_uploader("Upload a profile picture", type=['png', 'jpg', 'jpeg'])

            submitted = st.form_submit_button("Create Account", use_container_width=True, type="primary")

            if submitted:
                if not valid_username(username):
                    st.error("Invalid username format. Must start with a letter and contain only letters, numbers, and underscores.")
                elif not valid_email(email):
                    st.error("Invalid email format. Please enter a valid email address.")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters long")
                elif password != verify_password:
                    st.error("Passwords do not match")
                else:
                    conn = sqlite3.connect("users.db")
                    cur = conn.cursor()
                    try:
                        profile_pic_binary = None
                        if profile_pic:
                            image = Image.open(profile_pic)
                            profile_pic_binary = image_to_binary(image)

                        active_models = get_active_models()
                        default_model_id = active_models['model_id'].iloc[0] if not active_models.empty else None

                        # Save account as inactive until verification
                        cur.execute("""
                            INSERT INTO users (username, name, email, age_category, language, password, profile_pic, theme, default_model, is_verified)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                        """, (username, name, email, age_category, "English", hash_password(password),
                              profile_pic_binary, st.session_state.theme, default_model_id))
                        conn.commit()

                        # --- Send OTP and activation link ---
                        import random, string
                        otp = ''.join(random.choices(string.digits, k=6))
                        st.session_state.register_otp = otp
                        st.session_state.register_user = username
                        st.session_state.register_email = email

                        reset_link = f"http://localhost:8501/?activate_user={username}&otp={otp}"

                        send_registration_otp(email, username, otp)   # 🔥 send OTP
                        # send_password_reset(email, username, reset_link)  # 🔥 send activation link

                        st.success("✅ Account created! Please verify OTP from your email before logging in.")

                    except sqlite3.IntegrityError:
                        st.error("Username or Email already exists")
                    finally:
                        conn.close()

        # --- OTP Verification ---
        if "register_otp" in st.session_state:
            st.markdown("### 🔑 Verify Your Email")
            entered_otp = st.text_input("Enter the OTP sent to your email", key="reg_otp")

            if st.button("Verify & Activate", use_container_width=True, type="primary"):
                if entered_otp == st.session_state.register_otp:
                    conn = sqlite3.connect("users.db")
                    cur = conn.cursor()
                    cur.execute("UPDATE users SET is_verified=1 WHERE username=?", (st.session_state.register_user,))
                    conn.commit()
                    conn.close()

                    st.success("🎉 Account verified successfully! You can now login.")
                    st.session_state.pop("register_otp")
                    st.session_state.page = "login"
                    st.rerun()
                else:
                    st.error("❌ Invalid OTP. Please check your email again.")

        # "Back to Login" button
        if st.button("⬅️ Back to Login", key="back_login", use_container_width=True):
            st.session_state.page = "login"
            animate_transition()
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)


# ---------- Home Page (AI Tasks + Readability Analysis) ----------
elif st.session_state.page == "home":
    custom_header()

    # The home page layout and logic remains the same. The performance gain comes from the cached functions.
    st.set_page_config(page_title="Home", layout="wide")

    if "submitted" not in st.session_state:
        st.session_state.submitted = False
    if "ai_output" not in st.session_state:
        st.session_state.ai_output = ""
    if "__latest_source_text" not in st.session_state:
        st.session_state.__latest_source_text = ""

    theme = st.session_state.get("theme", "dark")

    # ---------- CSS ----------
    st.markdown("""
    <style>
    .feature-card {
        background-color: rgba(255, 255, 255, 0.03);
        padding: 1.2rem;
        border-radius: 14px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .sub-header {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.9rem;
        display: flex;
        align-items: center;
        gap: .5rem;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 12px;
        background: linear-gradient(145deg, rgba(128,128,128,0.10), rgba(128,128,128,0.05));
        text-align: center;
        transition: transform .2s ease;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-value { font-size: 2rem; font-weight: 800; margin-bottom: .25rem; }
    .metric-label { font-size: .85rem; opacity: .85; margin-bottom: .5rem; }
    .section-divider {
        height: 1px;
        background: linear-gradient(to right, transparent, rgba(128,128,128,0.25), transparent);
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 40px; background-color: rgba(255,255,255,0.05);
        border-radius: 8px 8px 0 0; padding-top: 8px; padding-bottom: 8px;
    }
    .stTabs [aria-selected="true"] { background-color: rgba(255,255,255,0.10); }
    .hint {
        font-size: .85rem; opacity: .8; padding: .35rem .6rem; border-radius: 6px;
        background: rgba(125,125,125,.12); display: inline-block;
    }
    </style>
    """, unsafe_allow_html=True)

    # ---------- DB: Active Models ----------
    conn = sqlite3.connect("users.db")
    active_models_df = pd.read_sql_query("SELECT * FROM models WHERE status='active'", conn)
    conn.close()

    # ---------- Split Screen ----------
    left, right = st.columns([1.15, 1.85], gap="large")

    # ======================================================================================
    # LEFT: Inputs (Model selection, text input, question, process)
    # ======================================================================================
    with left:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">🤖 Select AI Task & Model</p>', unsafe_allow_html=True)

        selected_task_type = None
        selected_model_id = None
        selected_display_name = None
        user_gemini_key = "" # Initialize key variable
        gemini_task = ""     # Initialize Gemini task variable

        if not active_models_df.empty:
            model_options = {
                row['display_name']: {'id': row['model_id'], 'task': row['task_type']}
                for _, row in active_models_df.iterrows()
            }

            # 1. Prepare lists for finding the index
            model_display_names = list(model_options.keys())
            model_ids = [v['id'] for v in model_options.values()]

            # 2. Get the user's saved default model ID from the database
            conn_home = sqlite3.connect(DB_PATH)
            user_df = pd.read_sql_query("SELECT default_model FROM users WHERE username=?", conn_home, params=(st.session_state.user,))
            user_default_model_id = user_df['default_model'].iloc[0] if not user_df.empty else None
            conn_home.close()

            # 3. Calculate the correct index for the selectbox
            default_index = 0  # Fallback to the first item
            if user_default_model_id and user_default_model_id in model_ids:
                try:
                    default_index = model_ids.index(user_default_model_id)
                except ValueError:
                    # This handles the case where a user's saved model might no longer be active.
                    # It will safely fall back to the first model in the list.
                    pass

            # 4. Create the selectbox with the dynamic index
            selected_display_name = st.selectbox(
                "Choose an AI Model",
                model_display_names,
                index=default_index, # <-- THE FIX IS APPLIED HERE
                help="Pick the model you want to run on your text"
            )

            if selected_display_name:
                selected_model_id = model_options[selected_display_name]['id']
                selected_task_type = model_options[selected_display_name]['task']
        else:
            st.warning("No active models available. Please contact an administrator.")

        # --- NEW: Gemini Multi-Task UI ---
        # This block appears only if the selected model has the 'multi_task' type
        if selected_task_type == 'multi_task':
            st.markdown("---")
            st.subheader("⚙️ Gemini Configuration")

            # 1. Ask for the specific task
            gemini_task = st.selectbox(
                "Select a Task for Gemini",
                ["Summarization", "Question Answering", "Text Generation"]
            )

            # 2. Ask for the API Key with a help button
            current_key = get_user_gemini_key(st.session_state.user)

            # --- START: MODIFIED SECTION WITH HELP BUTTON ---
            api_key_col, help_col = st.columns([4, 1])

            with api_key_col:
                new_key = st.text_input(
                    "Enter your Google Gemini API Key",
                    value=current_key,
                    type="password",
                    help="Get your key from Google AI Studio.",
                    label_visibility="collapsed",
                    placeholder="Enter your Google Gemini API Key"
                )

            with help_col:
                with st.popover("Instructions"):
                    st.markdown("""
                        #### How to get your Gemini API Key:
                        1.  **Go to Google AI Studio**:
                            [aistudio.google.com](https://aistudio.google.com/)
                        2.  **Sign in** with your Google Account.
                        3.  Click the **"< Get API key"** button, usually found in the top-left menu.
                        4.  In the new window, click **"Create API key in new project"**.
                        5.  **Copy** the generated key that appears.
                        6.  Come back here and **paste** it into the input box.
                    """)
            # --- END: MODIFIED SECTION WITH HELP BUTTON ---

            if st.button("Save Key"):
                update_user_gemini_key(st.session_state.user, new_key)
                st.success("API Key saved successfully!")
                st.rerun()

            if current_key:
                st.success("Gemini API Key is set and ready to use.")
                user_gemini_key = current_key # Set the key for processing
            else:
                st.warning("Please enter and save your Gemini API key to proceed.")

        st.markdown('</div>', unsafe_allow_html=True)

        # ------------- Input Form -------------
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">📥 Input Your Text</p>', unsafe_allow_html=True)

                # THE POPOVER IS MOVED OUTSIDE OF THE FORM
        with st.popover("🌐 Translate Input Text"):
            user_lang = get_user_language(st.session_state.user)
            try:
                # Set default language from user's profile
                default_lang_index = LANGUAGE_NAMES.index(user_lang.capitalize())
            except ValueError:
                default_lang_index = LANGUAGE_NAMES.index("English") # Fallback

            target_lang = st.selectbox(
                "Translate to:",
                LANGUAGE_NAMES,
                index=default_lang_index,
                key="translate_input_lang"
            )
            if st.button("Translate", key="translate_input_btn"):
                # We check session state directly since the popover is outside the form
                if st.session_state.manual_text:
                    translated_input = translate_text(st.session_state.manual_text, target_lang)
                    st.success("Translated Text:")
                    st.text_area("Translated", value=translated_input, height=150, disabled=True, label_visibility="collapsed")
                else:
                    st.warning("Please enter some text in the main input box first.")

        output_result = ""
        input_data_for_db = ""
        source_text = ""

        with st.form("home_input_form", clear_on_submit=False):
            tab1, tab2 = st.tabs(["✏️ Manual", "📁 Upload"])
            with tab1:
                source_text = st.text_area(
                    "Enter text",
                    value=st.session_state.get("manual_text", ""),
                    key="manual_text",
                    height=200,
                    placeholder="Paste or type your text here...",
                    label_visibility="collapsed",
                )
                st.caption("Tip: You can paste long content; output will show in the right panel.")

            with tab2:
                uploaded_file = st.file_uploader(
                    "Choose a file (.txt, .pdf, .docx)",
                    type=['txt', 'pdf', 'docx']
                )
                if uploaded_file is not None:
                    with st.spinner("Extracting text..."):
                        if uploaded_file.type == "text/plain":
                            source_text = uploaded_file.getvalue().decode("utf-8")
                        elif uploaded_file.type == "application/pdf":
                            source_text = extract_text_from_pdf(uploaded_file)
                        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                            source_text = extract_text_from_docx(uploaded_file)
                    if source_text:
                        st.text_area("📝 Extracted Text (preview)", source_text, height=150, disabled=True)
                    else:
                        st.error("Could not extract text from the uploaded file.")

            # Conditionally show the question input field
            is_qa_task = (selected_task_type == "question_answering") or \
                         (selected_task_type == 'multi_task' and gemini_task == 'Question Answering')

            if is_qa_task:
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                st.text_input(
                    "❓ Your Question",
                    key="qa_question",
                    placeholder="What would you like to ask about this text?"
                )
                process_button_label = "❓ Find Answer & Analyze Readability"
            else:
                process_button_label = "✨ Process Text & Analyze Readability"
            # --- NEW: Conditionally show Summarization Length Options ---
            is_summarization_task = (selected_task_type == "summarization") or \
                                    (selected_task_type == 'multi_task' and gemini_task == 'Summarization')


            if is_summarization_task:
                st.markdown("---")
                summary_length_option = st.radio(
                    "Select Summary Length",
                    ["Small", "Medium", "Large"],
                    index=1,  # Default to Medium
                    horizontal=True,
                    key="summary_length"
                )
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            c1, c2 = st.columns([1,1])
            with c1:
                detailed_view = st.toggle("Detailed Charts", value=True, help="Turn off for a compact summary")
            with c2:
                st.markdown('<span class="hint">Press Ctrl/Cmd + Enter to submit</span>', unsafe_allow_html=True)

            col_go, col_clear = st.columns([3,1])
            with col_go:
                process_clicked = st.form_submit_button(
                    process_button_label,
                    use_container_width=True,
                    type="primary",
                    disabled=(selected_model_id is None)
                )
            with col_clear:
                st.form_submit_button(
                    "Clear",
                    use_container_width=True,
                    on_click=clear_form_states  # ✅ Use the callback function here
                )

        st.markdown('</div>', unsafe_allow_html=True) # The form ends here

        # ------------- MODIFIED Processing Logic -------------
        if process_clicked:
            output_result = ""
            st.session_state.readability_scores = None

            # --- Define length parameters based on UI selection ---
            min_len, max_len = 50, 150 # Default Medium
            gemini_prompt_instruction = "medium-length (about 3-4 sentences)"
            if st.session_state.get("summary_length") == "Small":
                min_len, max_len = 20, 60
                gemini_prompt_instruction = "short, one-sentence"
            elif st.session_state.get("summary_length") == "Large":
                min_len, max_len = 150, 300
                gemini_prompt_instruction = "detailed, multi-paragraph"

            # --- Check if Gemini (multi_task) was selected ---
            if selected_task_type == 'multi_task':
                if not user_gemini_key:
                    st.error("Please enter and save your Gemini API key before processing.")
                else:
                    with st.status(f"Asking Gemini to perform {gemini_task}...", expanded=True) as status:
                        if gemini_task == "Summarization":
                            output_result = summarize_with_gemini(user_gemini_key, source_text, length_instruction=gemini_prompt_instruction)
                            input_data_for_db = source_text
                        elif gemini_task == "Question Answering":
                            output_result = answer_question_with_gemini(user_gemini_key, source_text, st.session_state.qa_question)
                            input_data_for_db = f"Context: {source_text[:500]}...\n\nQuestion: {st.session_state.qa_question}"
                        elif gemini_task == "Text Generation":
                            output_result = generate_text_with_gemini(user_gemini_key, source_text)
                            input_data_for_db = source_text
                        status.update(label="Gemini processing complete!", state="complete")

            # --- Existing Hugging Face Logic ---
            else:
                has_text = bool(source_text.strip())
                if selected_task_type == "question_answering":
                    has_text = has_text and bool(st.session_state.qa_question.strip())

                if not has_text:
                    st.warning("Please provide the required text input(s).")
                else:
                    with st.status("Running Hugging Face model...", expanded=True) as status:
                        if selected_task_type == "question_answering":
                            model_pipeline = load_qa_model(selected_model_id)
                            output_result = answer_question(model_pipeline, source_text, st.session_state.qa_question)
                            input_data_for_db = f"Context: {source_text[:500]}...\n\nQuestion: {st.session_state.qa_question}"
                        elif selected_task_type == "summarization":
                            model_pipeline, tokenizer = load_summarizer(selected_model_id)
                            output_result = summarize_text(model_pipeline, tokenizer, source_text, min_length=min_len, max_length=max_len)
                            input_data_for_db = source_text
                        elif selected_task_type == "paraphrasing":
                            model_pipeline, tokenizer = load_paraphraser(selected_model_id)
                            paraphrases = paraphrase_text(model_pipeline, tokenizer, source_text)
                            output_result = "\n\n".join([f"Option {i+1}: {p}" for i, p in enumerate(paraphrases)])
                            input_data_for_db = source_text
                        elif selected_task_type == "text_generation":
                            model_pipeline = load_text_generation_model(selected_model_id)
                            output_result = generate_text(model_pipeline, source_text, max_length=150)
                            input_data_for_db = source_text
                        status.update(label="Hugging Face processing complete!", state="complete")

            # --- Common Logic for saving results (for both Gemini and Hugging Face) ---
            if output_result and source_text:
                # --- NEW: Calculate scores for both input and output ---
                try:
                    input_scores = {
                        "Flesch Reading Ease": textstat.flesch_reading_ease(source_text),
                        "Flesch-Kincaid Grade": textstat.flesch_kincaid_grade(source_text),
                        "Gunning Fog": textstat.gunning_fog(source_text),
                        "SMOG Index": textstat.smog_index(source_text),
                    }
                except Exception: # Fallback for very short text
                    input_scores = {k: 0 for k in ["Flesch Reading Ease", "Flesch-Kincaid Grade", "Gunning Fog", "SMOG Index"]}

                try:
                    output_scores = {
                        "Flesch Reading Ease": textstat.flesch_reading_ease(output_result),
                        "Flesch-Kincaid Grade": textstat.flesch_kincaid_grade(output_result),
                        "Gunning Fog": textstat.gunning_fog(output_result),
                        "SMOG Index": textstat.smog_index(output_result),
                    }
                except Exception: # Fallback for very short AI answers
                    output_scores = {k: 0 for k in ["Flesch Reading Ease", "Flesch-Kincaid Grade", "Gunning Fog", "SMOG Index"]}

                st.session_state.comparison_scores = {
                    'input': input_scores,
                    'output': output_scores
                }
                # Keep the old variable for DB saving to avoid breaking the schema
                st.session_state.readability_scores = input_scores
                # --- END NEW ---
                try:
                    conn = sqlite3.connect("users.db")
                    cur = conn.cursor()
                    s = st.session_state.readability_scores # Save input scores to DB
                    sentiment = analyze_sentiment(input_data_for_db)
                    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
                    scores = scorer.score(source_text, output_result)
                    rouge1, rouge2, rougeL = scores['rouge1'].fmeasure, scores['rouge2'].fmeasure, scores['rougeL'].fmeasure

                    final_task_type = gemini_task if selected_task_type == 'multi_task' else selected_task_type

                    cur.execute("""
                        INSERT INTO submissions
                        (username, timestamp, model_id_used, task_type, input_text, output_text, sentiment,
                         flesch_reading_ease, flesch_kincaid_grade, gunning_fog, smog_index,
                         rouge1, rouge2, rougeL)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        st.session_state.user, str(datetime.datetime.now()), selected_model_id,
                        final_task_type, input_data_for_db, output_result, sentiment,
                        s['Flesch Reading Ease'], s['Flesch-Kincaid Grade'], s['Gunning Fog'], s['SMOG Index'],
                        rouge1, rouge2, rougeL
                    ))
                    conn.commit()
                    st.toast("✅ Result saved to your dashboard!")
                except sqlite3.Error as e:
                    st.error(f"Database error: {e}")
                finally:
                    if conn: conn.close()

        if output_result:
            st.session_state["__latest_output_result"] = output_result
        if source_text:
            st.session_state["__latest_source_text"] = source_text
        if process_clicked:
            st.session_state["__latest_detailed_view"] = detailed_view

    # ======================================================================================
    # RIGHT: Results (AI Output, Readability, Visualizations)
    # ======================================================================================
    with right:
        if st.session_state.get("__latest_output_result"):
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            header_cols = st.columns([3, 1])
            with header_cols[0]:
                st.markdown('<p class="sub-header">📊 AI Model Output</p>', unsafe_allow_html=True)
            with header_cols[1]:
                 with st.popover("🌐 Translate Output"):
                    user_lang = get_user_language(st.session_state.user)
                    try:
                        # Set default language from user's profile
                        default_lang_index = LANGUAGE_NAMES.index(user_lang.capitalize())
                    except ValueError:
                        default_lang_index = LANGUAGE_NAMES.index("English") # Fallback

                    target_lang_output = st.selectbox(
                        "Translate to:",
                        LANGUAGE_NAMES,
                        index=default_lang_index,
                        key="translate_output_lang"
                    )
                    if st.button("Translate", key="translate_output_btn"):
                        translated_output = translate_text(st.session_state["__latest_output_result"], target_lang_output)
                        st.success("Translated Text:")
                        st.text_area("Translated", value=translated_output, height=150, disabled=True, label_visibility="collapsed")
            with st.expander("View AI Output", expanded=True):
                st.success(st.session_state["__latest_output_result"])
            st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.readability_scores:
            s = st.session_state.readability_scores
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown('<p class="sub-header">📈 Readability Analysis</p>', unsafe_allow_html=True)
            st.caption("Analysis of your input text's readability metrics")

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                ease = s['Flesch Reading Ease']
                color = "#63d471" if ease > 70 else ("#fbd46d" if ease > 50 else "#f06a6a")
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:{color}">{ease:.1f}</div>
                    <div class="metric-label">Flesch Reading Ease</div>
                </div>
                """, unsafe_allow_html=True)
                st.caption("Higher = easier to read")
            with m2:
                fk = s['Flesch-Kincaid Grade']
                color = "#63d471" if fk <= 8 else ("#fbd46d" if fk <= 12 else "#f06a6a")
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:{color}">{fk:.1f}</div>
                    <div class="metric-label">Flesch-Kincaid Grade</div>
                </div>
                """, unsafe_allow_html=True)
                st.caption("U.S. grade level")
            with m3:
                fog = s['Gunning Fog']
                color = "#63d471" if fog < 10 else ("#fbd46d" if fog < 15 else "#f06a6a")
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:{color}">{fog:.1f}</div>
                    <div class="metric-label">Gunning Fog</div>
                </div>
                """, unsafe_allow_html=True)
                st.caption("Years of education")
            with m4:
                smog = s['SMOG Index']
                color = "#63d471" if smog < 10 else ("#fbd46d" if smog < 15 else "#f06a6a")
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:{color}">{smog:.1f}</div>
                    <div class="metric-label">SMOG Index</div>
                </div>
                """, unsafe_allow_html=True)
                st.caption("Years of education")

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            detailed_view = st.session_state.get("__latest_detailed_view", True)

            if detailed_view:
                viz_tab1, viz_tab2, viz_tab3, viz_tab4, viz_tab5 = st.tabs(
                    ["📊 Score Comparison", "📈 Grade Level", "🎯 Radar Chart", "📋 Text Stats", "🔍 Compare Original vs Generated"]
                )
                with viz_tab1:
                    # --- Get the comparison scores from session state ---
                    scores = st.session_state.comparison_scores
                    input_scores = scores['input']
                    output_scores = scores['output']

                    st.markdown("#### Readability Scores Comparison")
                    score_names = list(input_scores.keys())
                    input_values = list(input_scores.values())
                    output_values = list(output_scores.values())

                    # --- Create a grouped bar chart ---
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=score_names,
                        y=input_values,
                        name='Input',
                        marker_color='#636efa',
                        text=[f"{v:.1f}" for v in input_values]
                    ))
                    fig.add_trace(go.Bar(
                        x=score_names,
                        y=output_values,
                        name='Output',
                        marker_color='#ef553b',
                        text=[f"{v:.1f}" for v in output_values]
                    ))
                    fig.update_traces(textposition='auto')
                    fig.update_layout(
                        barmode='group',
                        height=400,
                        yaxis_title="Score",
                        xaxis_tickangle=-45,
                        legend_title_text='Text Type'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # --- Create a side-by-side metric comparison ---
                    st.markdown("##### Score Interpretation")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**📝 Input Text**")
                        st.metric("Flesch Reading Ease", f"{input_scores['Flesch Reading Ease']:.1f}",
                                  help="90-100: Very Easy, 60-70: Standard, 0-30: Very Confusing")
                        st.metric("Flesch-Kincaid Grade", f"Grade {input_scores['Flesch-Kincaid Grade']:.1f}",
                                  help="Approximate U.S. school grade level")
                    with col2:
                        st.markdown("**🤖 Output Text**")
                        st.metric("Flesch Reading Ease", f"{output_scores['Flesch Reading Ease']:.1f}",
                                  help="90-100: Very Easy, 60-70: Standard, 0-30: Very Confusing")
                        st.metric("Flesch-Kincaid Grade", f"Grade {output_scores['Flesch-Kincaid Grade']:.1f}",
                                  help="Approximate U.S. school grade level")

                with viz_tab2:
                    st.markdown("#### Grade Level Comparison")
                    grade_metrics = {
                        "Flesch-Kincaid": s['Flesch-Kincaid Grade'],
                        "Gunning Fog": s['Gunning Fog'],
                        "SMOG Index": s['SMOG Index'],
                        "Coleman-Liau": s.get('Coleman-Liau Index', 0),
                        "ARI": s.get('Automated Readability Index', 0)
                    }
                    fig = go.Figure(data=[go.Bar(
                        y=list(grade_metrics.keys()),
                        x=list(grade_metrics.values()),
                        orientation='h',
                        marker_color='#636efa'
                    )])
                    fig.update_layout(height=400, xaxis_title="Grade Level", yaxis_title="Metric")
                    st.plotly_chart(fig, use_container_width=True)

                    avg_grade = sum(grade_metrics.values()) / len(grade_metrics)
                    st.metric("Average Grade Level", f"{avg_grade:.1f}")

                with viz_tab3:
                    st.markdown("#### Readability Radar Chart")
                    categories = ['Flesch Ease', 'F-K Grade', 'Gunning Fog', 'SMOG Index']
                    max_ease, max_grade = 100, 20
                    normalized = [
                        s['Flesch Reading Ease'] / max_ease * 10,
                        (max_grade - s['Flesch-Kincaid Grade']) / max_grade * 10,
                        (max_grade - s['Gunning Fog']) / max_grade * 10,
                        (max_grade - s['SMOG Index']) / max_grade * 10,
                    ]
                    categories = categories + [categories[0]]
                    normalized = normalized + [normalized[0]]

                    fig = go.Figure(go.Scatterpolar(
                        r=normalized, theta=categories, fill='toself',
                        fillcolor='rgba(99,110,250,0.3)',
                        line=dict(color='rgb(99,110,250)'),
                        name="Readability Scores"
                    ))
                    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,10])),
                                      showlegend=False, height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Higher values indicate better readability across                    multiple dimensions.")

                with viz_tab4:
                    st.markdown("#### Text Statistics")
                    source_text_for_stats = st.session_state.get("__latest_source_text", "")
                    word_count = len(source_text_for_stats.split())
                    char_count = len(source_text_for_stats)
                    sentence_count = source_text_for_stats.count('.') + source_text_for_stats.count('!') + source_text_for_stats.count('?')
                    avg_sentence_len = word_count / max(sentence_count, 1)
                    avg_word_len = char_count / max(word_count, 1)

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Words", f"{word_count}")
                        st.metric("Characters", f"{char_count}")
                    with c2:
                        st.metric("Sentences", f"{sentence_count}")
                        st.metric("Avg Sentence Length", f"{avg_sentence_len:.1f} words")
                    with c3:
                        st.metric("Avg Word Length", f"{avg_word_len:.1f} chars")

                with viz_tab5:
                    st.markdown("#### Original vs Generated Text")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("📝 Original Text")
                        st.text_area("Original", st.session_state.get("__latest_source_text", ""), height=250, disabled=True)

                    with col2:
                        st.subheader("🤖 Generated Text")
                        st.text_area("Generated", st.session_state.get("__latest_output_result", ""), height=250, disabled=True)

                    # --- ROUGE Evaluation ---
                    orig = st.session_state.get("__latest_source_text", "")
                    gen = st.session_state.get("__latest_output_result", "")

                    if orig and gen:
                        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                        st.markdown("### 📏 ROUGE Evaluation")

                        from rouge_score import rouge_scorer
                        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
                        scores = scorer.score(orig, gen)

                        rouge_scores = {
                            "ROUGE-1 (Unigrams)": scores['rouge1'].fmeasure,
                            "ROUGE-2 (Bigrams)": scores['rouge2'].fmeasure,
                            "ROUGE-L (LCS)": scores['rougeL'].fmeasure,
                        }

                        # --- Bar Chart ---
                        fig = go.Figure(data=[go.Bar(
                            x=list(rouge_scores.keys()),
                            y=list(rouge_scores.values()),
                            marker_color=["#63d471", "#fbd46d", "#4c9aff"],
                            text=[f"{v:.3f}" for v in rouge_scores.values()],
                            textposition="auto"
                        )])
                        fig.update_layout(
                            height=400,
                            showlegend=False,
                            yaxis=dict(title="ROUGE Score (0–1)", range=[0,1]),
                            xaxis_tickangle=-20
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # --- Metrics ---
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.metric("ROUGE-1", f"{rouge_scores['ROUGE-1 (Unigrams)']:.3f}", help="Measures unigram (word-level) overlap")
                        with c2:
                            st.metric("ROUGE-2", f"{rouge_scores['ROUGE-2 (Bigrams)']:.3f}", help="Measures bigram (two-word sequence) overlap")
                        with c3:
                            st.metric("ROUGE-L", f"{rouge_scores['ROUGE-L (LCS)']:.3f}", help="Measures longest common subsequence overlap")




            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("---")
            with st.popover("🐞 Report a Problem", use_container_width=False):
                with st.form("ticket_form", clear_on_submit=True):
                    st.markdown("**Having an issue?** Describe the problem below and we'll take a look.")
                    user_desc = st.text_area("What went wrong? (Optional)", key="user_ticket_desc")

                    if st.form_submit_button("Submit Report", type="primary"):
                        # Gather all relevant session data
                        session_context = {
                            "username": st.session_state.user,
                            "input_text": st.session_state.get("__latest_source_text", "Not available"),
                            "output_text": st.session_state.get("__latest_output_result", "Not available"),
                            "selected_model": st.session_state.get("selected_model_id", "Not available"),
                            "task_type": st.session_state.get("gemini_task", st.session_state.get("selected_task_type", "Not available")),
                            "comparison_scores": st.session_state.get("comparison_scores", "Not available")
                        }
                        if create_ticket(st.session_state.user, user_desc, session_context):
                            st.success("✅ Your report has been submitted. We'll get back to you soon!")
                        else:
                            st.error("❌ Could not submit your report. Please try again.")

        else:
            st.info("Submit text on the left to see AI output and readability analytics here.")

# ---------- User Dashboard Page ----------
elif st.session_state.page == "dashboard":
    st.markdown(f'<h1 class="main-header" style="text-align: center;">Your Personal Dashboard</h1>', unsafe_allow_html=True)

    top_cols = st.columns([1, 4, 1])
    with top_cols[0]:
        if st.button("⬅️ Back to Home", use_container_width=True, key="back_home_top"):
            st.session_state.page = "home"
            animate_transition()
            st.rerun()
    with top_cols[2]:
        if st.button("Logout 🚪", use_container_width=True, help="Logout from your account"):
            st.session_state.user = None
            st.session_state.page = "login"
            animate_transition()
            st.rerun()

    st.markdown("---")

    user_data_row = pd.read_sql_query("SELECT * FROM users WHERE username=?", get_conn(), params=(st.session_state.user,)).iloc[0]

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["👤 Profile & Security", "🤖 AI Preferences", "🧠 Compare Models", "📊 My Analytics", "🤝 My Teams"])

    # --- TAB 1: Profile & Security ---
    with tab1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Profile Settings</p>', unsafe_allow_html=True)

        profile_cols = st.columns([1, 2])
        with profile_cols[0]:
            st.subheader("Profile Picture")
            if user_data_row['profile_pic']:
                try:
                    image_b64 = base64.b64encode(user_data_row['profile_pic']).decode()
                    st.markdown(f'''<div style="width: 150px; height: 150px; border-radius: 50%; overflow: hidden; border: 3px solid #444;"><img src="data:image/png;base64,{image_b64}" style="width: 100%; height: 100%; object-fit: cover;"></div>''', unsafe_allow_html=True)
                except:
                    st.error("Could not load image.")
            else:
                st.info("No profile picture uploaded.")
            new_profile_pic = st.file_uploader("Update Profile Picture", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")

        with profile_cols[1]:
            name = st.text_input("Name", value=user_data_row['name'] or "")
            age_category = st.selectbox("Age Category", AGE_CATEGORIES, index=AGE_CATEGORIES.index(user_data_row['age_category'] or "15 - 20"))
            lang = st.radio("Preferred Language", ["English", "Telugu", "Hindi"], index=["English", "Telugu", "Hindi"].index(user_data_row['language'] or "English"))
            # NEW: Reading Preferences and Content Type fields
            reading_prefs = st.text_input("Reading Preferences (e.g., simple, academic)", value=user_data_row['reading_preferences'] or "")
            content_type = st.text_input("Preferred Content Type (e.g., news, technical)", value=user_data_row['content_type'] or "")

        if st.button("Update Profile", use_container_width=True, type="primary"):
            conn_profile = get_conn()
            cur = conn_profile.cursor()
            profile_pic_binary = user_data_row['profile_pic']
            if new_profile_pic:
                image = Image.open(new_profile_pic)
                profile_pic_binary = image_to_binary(image)
            cur.execute("""UPDATE users SET name=?, age_category=?, language=?, profile_pic=?,
                          reading_preferences=?, content_type=? WHERE username=?""",
                          (name, age_category, lang, profile_pic_binary,
                          reading_prefs, content_type, st.session_state.user))
            conn_profile.commit()
            conn_profile.close()
            st.success("Profile updated successfully!")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="feature-card" style="margin-top: 1rem;">', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Change Password</p>', unsafe_allow_html=True)
        with st.form("password_form"):
            current_password = st.text_input("Current Password", type="password")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm New Password", type="password")
            submitted = st.form_submit_button("Update Password", use_container_width=True)
            if submitted:
                if hash_password(current_password) != user_data_row['password']:
                    st.error("Current password is incorrect")
                elif new_password != confirm_password:
                    st.error("New passwords do not match")
                elif len(new_password) < 6:
                    st.error("New password must be at least 6 characters long")
                else:
                    update_password(st.session_state.user, new_password)
                    st.success("Password updated successfully!")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- TAB 2: AI Preferences ---
    with tab2:
        pref_tab1, pref_tab2 = st.tabs(["⚙️ Default AI Model", "🤖 My AI Agents"])

        with pref_tab1:
            # This tab's code remains unchanged
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown('<p class="sub-header">Default AI Model</p>', unsafe_allow_html=True)
            st.info("Select the AI model you'd like to use by default on the home page.")
            active_models_df = get_active_models()
            if not active_models_df.empty:
                model_options = {f"{row['display_name']}": row['model_id'] for _, row in active_models_df.iterrows()}
                current_default_model_id = user_data_row['default_model']
                model_display_names = list(model_options.keys())
                default_index = 0
                for i, model_id in enumerate(model_options.values()):
                    if model_id == current_default_model_id:
                        default_index = i
                        break
                selected_default_model_display = st.selectbox(
                    "Choose your default AI Model:",
                    model_display_names,
                    index=default_index,
                    key="default_model_selection"
                )
                if st.button("Set as Default Model", use_container_width=True, type="primary"):
                    selected_model_id = model_options[selected_default_model_display]
                    cur = conn.cursor()
                    cur.execute("UPDATE users SET default_model=? WHERE username=?", (selected_model_id, st.session_state.user))
                    conn.commit()
                    st.success("Default model updated successfully!")
                    st.rerun()
            else:
                st.warning("No active models are available to set as default. Please contact an administrator.")
            st.markdown('</div>', unsafe_allow_html=True)


        # --- VVV THIS IS THE FULL, CORRECTLY INDENTED CODE FOR THE "MY AI AGENTS" TAB VVV ---
        with pref_tab2:
            # Check if we are in re-training mode based on session state
            if 'retrain_agent_group' in st.session_state:
                agent_info = st.session_state.retrain_agent_group
                st.markdown(f"### 🚀 Train New Version for: {agent_info['display_name']}")
                st.caption("Provide new training data below. Any data used in previous versions will be automatically excluded.")

                # The data sourcing and training form for re-training
                with st.form("retrain_agent_form"):
                    # (The data source tabs are identical to the 'Create New Agent' form)
                    st.markdown("##### Provide New Training Data")
                    data_tab1, data_tab2, data_tab3 = st.tabs(["Manual Entry", "File Upload (CSV/JSON)", "Kaggle Dataset"])

                    with data_tab1:
                        manual_input = st.text_area("Input Text (Example)", key="retrain_manual_in")
                        manual_output = st.text_area("Ideal Output (Target)", key="retrain_manual_out")

                    with data_tab2:
                        uploaded_file = st.file_uploader("Upload a CSV or JSON file", type=["csv", "json"], key="retrain_file")
                        if uploaded_file:
                            try:
                                if uploaded_file.name.endswith('.csv'):
                                    df_preview = pd.read_csv(uploaded_file, nrows=5)
                                else:
                                    df_preview = pd.read_json(uploaded_file, orient='records', lines=True if '.jsonl' in uploaded_file.name else False).head(5)
                                st.write("File Preview:"); st.dataframe(df_preview)
                                file_cols = df_preview.columns.tolist()
                                file_input_col = st.selectbox("Select Input Column", file_cols, key="retrain_file_in")
                                file_output_col = st.selectbox("Select Output Column", file_cols, key="retrain_file_out")
                            except Exception as e:
                                st.error(f"Error reading file: {e}")

                    with data_tab3:
                        kaggle_id = st.text_input("Kaggle Dataset URL or Identifier", placeholder="owner/dataset-name", key="retrain_kaggle_id")
                        if st.form_submit_button("Inspect Kaggle Dataset"):
                            with st.spinner("Fetching dataset from Kaggle..."):
                                df, error = fetch_and_inspect_kaggle_dataset(kaggle_id)
                                if error:
                                    st.error(error)
                                    if 'kaggle_info_retrain' in st.session_state: del st.session_state['kaggle_info_retrain']
                                else:
                                    st.session_state.kaggle_info_retrain = {'df': df, 'total_rows': len(df), 'columns': df.columns.tolist()}
                                    st.success("Dataset inspected successfully! Configure below.")
                        if 'kaggle_info_retrain' in st.session_state:
                            info = st.session_state.kaggle_info_retrain
                            st.metric("Total Rows Found", f"{info['total_rows']:,}")
                            kaggle_input_col = st.selectbox("Select Input Column", info['columns'], key="retrain_kg_in")
                            kaggle_output_col = st.selectbox("Select Output Column", info['columns'], key="retrain_kg_out")
                            row_count = st.number_input(
                                "Number of rows to use for training",
                                min_value=1,
                                max_value=info['total_rows'],
                                value=min(1000, info['total_rows']),
                                step=100
                            )

                    submitted = st.form_submit_button("🚀 Start Re-training", use_container_width=True, type="primary")
                    if submitted:
                        new_training_data = []
                        if manual_input and manual_output: new_training_data.append((manual_input, manual_output))
                        if 'uploaded_file' in locals() and uploaded_file:
                            if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file)
                            else: df = pd.read_json(uploaded_file, orient='records', lines=True if '.jsonl' in uploaded_file.name else False)
                            new_training_data.extend(list(zip(df[file_input_col], df[file_output_col])))
                        if 'kaggle_info_retrain' in st.session_state:
                            info = st.session_state.kaggle_info_retrain
                            df = info['df'].head(row_count)
                            new_training_data.extend(list(zip(df[kaggle_input_col], df[kaggle_output_col])))
                            del st.session_state['kaggle_info_retrain']

                        if not new_training_data:
                            st.warning("Please provide new training data to add to the agent.")
                        else:
                            # --- KEY FEATURE: Filter out all previously used data for this agent group ---
                            all_versions = get_agent_versions(agent_info['agent_group_id'])
                            used_hashes = set()
                            for _, version in all_versions.iterrows():
                                used_hashes.update(get_used_data_hashes(version['model_id']))

                            fresh_training_data = [(inp, out) for inp, out in new_training_data if hashlib.sha256(str(inp).encode()).hexdigest() not in used_hashes]

                            if not fresh_training_data:
                                st.warning("The provided data has already been used to train this agent. Please provide new, unseen data.")
                            else:
                                st.info(f"Found {len(fresh_training_data)} new samples to train on.")

                                # --- FIX 1: Correctly determine the agent's main directory ---
                            agent_dir = os.path.dirname(all_versions.iloc[0]['model_id'])
                            new_version_number = all_versions['version_number'].max() + 1
                            new_model_path = os.path.join(agent_dir, f"v{new_version_number}")
                            os.makedirs(new_model_path, exist_ok=True)

                            conn = get_conn(); cur = conn.cursor()
                            # --- FIX 2: Use INSERT OR IGNORE to prevent unique constraint errors ---
                            cur.execute("""
                                INSERT OR IGNORE INTO models (model_id, display_name, remarks, status, task_type,
                                                    created_by_username, is_public, version_number, agent_group_id, is_active_version)
                                VALUES (?, ?, ?, 'training', ?, ?, 0, ?, ?, 0)
                            """, (new_model_path, agent_info['display_name'], "Awaiting training completion...", agent_info['task_type'],
                                  st.session_state.user, new_version_number, agent_info['agent_group_id']))
                            conn.commit(); conn.close()

                            log_agent_training_data(new_model_path, fresh_training_data)

                            train_model(base_model_id=agent_info['base_model_id'], training_data=fresh_training_data,
                                        base_task_type=agent_info['task_type'], output_model_dir=new_model_path)

                            st.success(f"Training started for v{new_version_number} of '{agent_info['display_name']}'!")
                            del st.session_state.retrain_agent_group
                            st.rerun()

            if st.button("Cancel"):
                del st.session_state.retrain_agent_group
                st.rerun()

            else: # Default view: Manage agents and create new ones
                st.markdown('<div class="feature-card">', unsafe_allow_html=True)
                st.markdown('<p class="sub-header">Manage Your Custom AI Agents</p>', unsafe_allow_html=True)

                agent_groups = get_user_agent_groups(st.session_state.user)

                if agent_groups.empty:
                    st.info("You haven't created any AI agents yet. Create one below!")
                else:
                    for _, group in agent_groups.iterrows():
                        agent_display_name = group['display_name'].split(' (')[0]
                        expander_title = f"{agent_display_name} ({group['task_type'].capitalize()})"

                        with st.expander(expander_title):
                            versions_df = get_agent_versions(group['agent_group_id'])

                            st.markdown("##### 📜 Versions")

                            # --- NEW: Clean table-like layout for versions ---
                            header_cols = st.columns([1, 1, 2, 2, 2, 1])
                            header_cols[0].markdown("**Version**")
                            header_cols[1].markdown("**Status**")
                            header_cols[2].markdown("**Test Loss**")
                            header_cols[3].markdown("**Primary Score**")
                            header_cols[4].markdown("**Remarks**")
                            header_cols[5].markdown("**Actions**")
                            st.markdown("<hr style='margin-top: 0; margin-bottom: 10px;'>", unsafe_allow_html=True)

                            for _, version in versions_df.iterrows():
                                v_cols = st.columns([1, 1, 2, 2, 2, 1])
                                v_cols[0].markdown(f"**v{version['version_number']}**")

                                # Status and Activation
                                if version['is_active_version']:
                                    v_cols[1].success("Active")
                                else:
                                    if v_cols[1].button("Activate", key=f"act_{version['model_id']}", use_container_width=True):
                                        activate_model_version(group['agent_group_id'], version['model_id'])
                                        st.rerun()

                                # Evaluation Metrics
                                if version['status'] == 'training':
                                    st_autorefresh(interval=5000, key=f"refresh_{version['model_id']}")
                                    progress_pct = version.get('training_progress', 0.0) or 0.0
                                    v_cols[2].progress(progress_pct, text=f"{progress_pct:.0%}")
                                    v_cols[3].caption(version.get('training_status_text', 'Initializing...'))
                                elif version['status'] == 'active':
                                    loss = version.get('eval_loss')
                                    v_cols[2].markdown(f"`{loss:.4f}`" if loss is not None else "N/A")
                                    if version['task_type'] == 'paraphrasing':
                                        bert = version.get('eval_bertscore')
                                        v_cols[3].markdown(f"`{bert:.2f}` (BERTScore)" if bert is not None else "N/A")
                                    else:
                                        rouge = version.get('eval_rougeL')
                                        v_cols[3].markdown(f"`{rouge:.2f}%` (ROUGE-L)" if rouge is not None else "N/A")
                                else:
                                    v_cols[2].error(f"{version['status'].capitalize()}")

                                v_cols[4].caption(version['remarks'])

                                # Delete Version Button
                                if not version['is_active_version']:
                                    if v_cols[5].button("🗑️", key=f"del_v_{version['model_id']}", help="Delete this version"):
                                        delete_model_version(version['model_id'])
                                        st.rerun()

                            st.markdown("---")
                            # Actions for the entire agent group
                            group_action_cols = st.columns(2)
                            if group_action_cols[0].button("🚀 Train New Version", key=f"retrain_{group['agent_group_id']}", use_container_width=True):
                                # Set session state to switch to the re-training UI
                                st.session_state.retrain_agent_group = {
                                    "agent_group_id": group['agent_group_id'],
                                    "display_name": agent_display_name,
                                    "task_type": group['task_type'],
                                    "base_model_id": versions_df.iloc[0]['remarks'].split("Base: ")[-1].strip()
                                }
                                st.rerun()

                            if group_action_cols[1].button("❌ Delete Agent (All Versions)", key=f"del_group_{group['agent_group_id']}", type="primary", use_container_width=True):
                                delete_agent_group(group['agent_group_id'])
                                st.rerun()

                st.markdown('</div>', unsafe_allow_html=True)
                with st.expander("➕ Create a New AI Agent (v1)", expanded=True):
                    with st.form("create_agent_form"):
                        agent_name = st.text_input("Agent Display Name", placeholder="e.g., My News Article Summarizer")
                        base_models_df = pd.read_sql_query("SELECT model_id, display_name FROM models WHERE model_id NOT LIKE '%gemini%' AND (created_by_username IS NULL OR created_by_username = 'admin')", get_conn())
                        base_model_options = {row['display_name']: row['model_id'] for _, row in base_models_df.iterrows()}
                        selected_base_display = st.selectbox("Select a Base Model to Fine-Tune", base_model_options.keys())

                        st.markdown("---")
                        st.markdown("##### Training Configuration")
                        # --- NEW: UI for Split and Augmentation ---
                        train_test_ratio = st.slider("Train/Test Split Ratio (Training %)", 10, 90, 80, 5, help="Percentage of data to use for training. The rest will be used for testing.")
                        enable_augmentation = st.checkbox("Enable Data Augmentation (Back-Translation)", help="Creates more training data by translating text to German and back. Increases training time.")

                        st.markdown("---")
                        st.markdown("##### Provide Training Data (You can use multiple sources at once)")
                        st.markdown("---")
                        st.markdown("##### Provide Training Data")
                        data_tab1, data_tab2, data_tab3 = st.tabs(["Manual Entry", "CSV Upload", "Kaggle Dataset"])

                        with data_tab1: # Manual
                            manual_input = st.text_area("Input Text (Example)")
                            manual_output = st.text_area("Ideal Output (Target)")

                        with data_tab2: # CSV
                            csv_file = st.file_uploader("Upload a CSV", type="csv")
                            if csv_file:
                                df_preview = pd.read_csv(csv_file, nrows=5)
                                st.write("CSV Preview:"); st.dataframe(df_preview)
                                csv_cols = df_preview.columns.tolist()
                                csv_input_col = st.selectbox("Select Input Column", csv_cols, key="csv_in")
                                csv_output_col = st.selectbox("Select Output Column", csv_cols, key="csv_out")

                        with data_tab3: # Kaggle
                            kaggle_id = st.text_input("Kaggle Dataset URL or Identifier", placeholder="gowrishankarp/newspaper-text-summarization-cnn-dailymail")

                            if st.form_submit_button("Inspect Kaggle Dataset"):
                                with st.spinner("Fetching dataset from Kaggle..."):
                                    df, error = fetch_and_inspect_kaggle_dataset(kaggle_id)
                                    if error:
                                        st.error(error)
                                        if 'kaggle_info' in st.session_state: del st.session_state['kaggle_info']
                                    else:
                                        st.session_state.kaggle_info = {
                                            'df': df,
                                            'total_rows': len(df),
                                            'columns': df.columns.tolist()
                                        }
                                        st.success("Dataset inspected successfully! Configure below.")

                            if 'kaggle_info' in st.session_state:
                                info = st.session_state.kaggle_info
                                st.metric("Total Rows Found", f"{info['total_rows']:,}")
                                kaggle_input_col = st.selectbox("Select Input Column", info['columns'], key="kg_in")
                                kaggle_output_col = st.selectbox("Select Output Column", info['columns'], key="kg_out")
                                row_count = st.number_input(
                                    "Number of rows to use for training",
                                    min_value=1,
                                    max_value=info['total_rows'],
                                    value=min(1000, info['total_rows']),
                                    step=100
                                )

                        submitted = st.form_submit_button("🚀 Start Training Agent", use_container_width=True, type="primary")
                        if submitted:
                            training_data = []

                            # Logic to gather data based on which tab is presumed active
                            if manual_input and manual_output:
                                training_data.append((manual_input, manual_output))

                            elif 'csv_file' in locals() and csv_file:
                                df = pd.read_csv(csv_file)
                                training_data.extend(list(zip(df[csv_input_col], df[csv_output_col])))

                            elif 'kaggle_info' in st.session_state:
                                info = st.session_state.kaggle_info
                                df = info['df'].head(row_count)
                                training_data.extend(list(zip(df[kaggle_input_col], df[kaggle_output_col])))
                                del st.session_state['kaggle_info'] # Clean up state

                            if not agent_name or not selected_base_display:
                                st.warning("Please provide an Agent Name and select a Base Model.")
                            elif not training_data:
                                st.warning("Please provide training data via one of the methods.")
                            else:
                                # 1. Define paths and IDs
                                safe_agent_name = re.sub(r'[^a-zA-Z0-9_-]', '_', agent_name)
                                agent_group_id = f"{st.session_state.user}_{safe_agent_name}" # Unique group ID
                                version_number = 1

                                # New directory structure: .../user/agent_name/v1/
                                agent_dir = os.path.join("user_finetuned_models", st.session_state.user, safe_agent_name)
                                new_model_path = os.path.join(agent_dir, f"v{version_number}")
                                os.makedirs(new_model_path, exist_ok=True)

                                if enable_augmentation:
                                    training_data = augment_data_via_backtranslation(training_data)

                                # 2. Insert into database with versioning info
                                base_model_id = base_model_options[selected_base_display]
                                base_task_type = pd.read_sql_query("SELECT task_type FROM models WHERE model_id = ?", get_conn(), params=(base_model_id,)).iloc[0,0]
                                final_display_name = f"{agent_name} ({base_task_type.capitalize()})"

                                conn = get_conn(); cur = conn.cursor()
                                cur.execute("""
                                    INSERT INTO models (model_id, display_name, remarks, status, task_type,
                                                        created_by_username, is_public, version_number, agent_group_id, is_active_version)
                                    VALUES (?, ?, ?, 'training', ?, ?, 0, ?, ?, 1)
                                """, (new_model_path, final_display_name, f"Trained on 0 samples at {datetime.datetime.now()}", base_task_type,
                                      st.session_state.user, version_number, agent_group_id))
                                conn.commit(); conn.close()

                                # Log the initial data
                                log_agent_training_data(new_model_path, training_data)

                                # 3. Start training process
                                train_model(
                                    base_model_id=base_model_id,
                                    training_data=training_data,
                                    base_task_type=base_task_type,
                                    output_model_dir=new_model_path,
                                    test_size=(100 - train_test_ratio) / 100.0 # Convert percentage to ratio
                                )

                                st.success(f"Training started for '{agent_name}'! It will appear in your list with evaluation scores upon completion.")
                                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)



    # --- VVV THIS IS THE FULL, CORRECTED CODE FOR THE "COMPARE MODELS" TAB VVV ---

    with tab3: # Or tab4/tab5, whichever is "Compare Models"

        summarize_tab, paraphrase_tab = st.tabs(["📊 Summarization Playground", "🎨 Paraphrasing Studio"])

        # ------------------ SUMMARIZATION PLAYGROUND ------------------
        with summarize_tab:
            st.markdown("#### Test and Analyze Summarization Models")
            st.caption("Enter a long text and see how different models capture its essence. Select two models for a direct head-to-head comparison.")

            with st.form("summarize_form"):
                s_input_text = st.text_area("Enter text to summarize (long text recommended)", height=200, key="s_text")
                s_models_df = pd.read_sql_query("SELECT display_name, model_id FROM models WHERE status='active' AND task_type='summarization'", get_conn())
                s_model_options = {row['display_name']: row['model_id'] for _, row in s_models_df.iterrows()}
                s_selected_models = st.multiselect("Select models to compare", s_model_options.keys(), max_selections=3, default=list(s_model_options.keys())[:2] if len(s_model_options) >= 2 else list(s_model_options.keys()))
                s_submitted = st.form_submit_button("🔬 Analyze Summaries", use_container_width=True)

            if s_submitted and s_selected_models and s_input_text:
                with st.spinner("Running summarization and analysis... This might take a moment."):
                    s_results = {}
                    threads = []
                    for model_display_name in s_selected_models:
                        model_id = s_model_options[model_display_name]
                        thread = threading.Thread(target=run_comparison_task, args=(model_id, "Summarization", s_input_text, s_results))
                        threads.append(thread)
                        thread.start()
                    for thread in threads: thread.join()

                st.markdown("---")
                st.subheader("Results")

                if len(s_selected_models) == 2:
                    # --- HEAD-TO-HEAD VIEW ---
                    m1_display, m2_display = s_selected_models[0], s_selected_models[1]
                    m1_id, m2_id = s_model_options[m1_display], s_model_options[m2_display]
                    m1_res, m2_res = s_results.get(m1_id, {}), s_results.get(m2_id, {})

                    if "error" in m1_res or "error" in m2_res:
                        st.error(f"An error occurred. M1: {m1_res.get('error', 'OK')}, M2: {m2_res.get('error', 'OK')}")
                    else:
                        m1_metrics, m2_metrics = m1_res['metrics'], m2_res['metrics']
                        win_counts = {'m1': 0, 'm2': 0}

                        def style_metric(val1, val2, higher_is_better=True, is_time=False):
                            style_win = "color: #28a745; font-weight: bold;"
                            v1_str = f"{val1:.2f}" if not is_time else f"{val1:.2f} s"
                            v2_str = f"{val2:.2f}" if not is_time else f"{val2:.2f} s"
                            if (higher_is_better and val1 > val2) or (not higher_is_better and val1 < val2):
                                win_counts['m1'] += 1
                                return f'<span style="{style_win}">{v1_str}</span>', v2_str
                            elif (higher_is_better and val2 > val1) or (not higher_is_better and val2 < val1):
                                win_counts['m2'] += 1
                                return v1_str, f'<span style="{style_win}">{v2_str}</span>'
                            return v1_str, v2_str

                        time_s = style_metric(m1_metrics['execution_time'], m2_metrics['execution_time'], higher_is_better=False, is_time=True)
                        words_s = style_metric(m1_metrics['word_count'], m2_metrics['word_count'], higher_is_better=False) # Lower is more concise
                        ease_s = style_metric(m1_metrics['flesch_ease'], m2_metrics['flesch_ease'], higher_is_better=True)
                        grade_s = style_metric(m1_metrics['flesch_grade'], m2_metrics['flesch_grade'], higher_is_better=False)
                        r1_s = style_metric(m1_metrics['rouge1'], m2_metrics['rouge1'], higher_is_better=True)
                        rL_s = style_metric(m1_metrics['rougeL'], m2_metrics['rougeL'], higher_is_better=True)

                        col1, col2 = st.columns(2)
                        for i, (display, mid, res, metrics) in enumerate([(m1_display, m1_id, m1_res, m1_metrics), (m2_display, m2_id, m2_res, m2_metrics)]):
                            with (col1 if i == 0 else col2):
                                st.markdown(f"#### {display}")
                                if st.button("Set as My Default", key=f"s_set_default_{mid}", use_container_width=True):
                                    update_user_default_model(st.session_state.user, mid)
                                    st.success(f"'{display}' is now your default model!")
                                with st.expander("View Full Summary"): st.write(res['output'])
                                st.markdown(f"""
                                | Metric | Score |
                                |---|---|
                                | **Execution Time** | {time_s[i]} |
                                | **Word Count** | {words_s[i]} |
                                | **Flesch Reading Ease** | {ease_s[i]} |
                                | **Flesch-Kincaid Grade** | {grade_s[i]} |
                                | **ROUGE-1 Score** | {r1_s[i]} |
                                | **ROUGE-L Score** | {rL_s[i]} |
                                """, unsafe_allow_html=True)

                        st.markdown("---")
                        st.subheader("🏆 Head-to-Head Verdict")
                        if win_counts['m1'] > win_counts['m2']: winner = m1_display
                        elif win_counts['m2'] > win_counts['m1']: winner = m2_display
                        else: winner = "It's a Tie!"
                        st.success(f"**Best Overall Performance:** {winner} ({max(win_counts.values())} wins)")

                else: # --- DEFAULT VIEW FOR 1 OR 3 MODELS ---
                    cols = st.columns(len(s_selected_models))
                    for i, model_display_name in enumerate(s_selected_models):
                        with cols[i]:
                            model_id = s_model_options[model_display_name]
                            res = s_results.get(model_id, {})
                            if "error" in res: st.error(f"**{model_display_name}** failed."); continue
                            st.markdown(f"##### {model_display_name}")
                            if st.button("Set as My Default", key=f"s_set_default_{model_id}", use_container_width=True):
                                update_user_default_model(st.session_state.user, model_id)
                                st.success(f"'{model_display_name}' is now your default model!")

                            metrics = res['metrics']
                            inner_tabs = st.tabs(["📝 Output", "📊 Analytics"])
                            with inner_tabs[0]:
                                st.metric("Execution Time", f"{metrics['execution_time']:.2f} s")
                                st.write(res['output'])
                            with inner_tabs[1]:
                                st.markdown("**Readability**")
                                st.metric("Flesch Reading Ease", f"{metrics['flesch_ease']:.1f}")
                                st.metric("Flesch-Kincaid Grade", f"{metrics['flesch_grade']:.1f}")
                                st.markdown("**Quality Scores**")
                                st.metric("ROUGE-1 (Overlap)", f"{metrics['rouge1']:.1f}%")
                                st.metric("ROUGE-L (Sequence)", f"{metrics['rougeL']:.1f}%")

        # ------------------ PARAPHRASING STUDIO ------------------
        with paraphrase_tab:
            st.markdown("#### Test and Analyze Paraphrasing Models")
            st.caption("Enter a sentence and analyze how models rephrase it. Select two for a direct head-to-head comparison.")

            with st.form("paraphrase_form"):
                p_input_text = st.text_area("Enter text to paraphrase", height=100, key="p_text")
                p_models_df = pd.read_sql_query("SELECT display_name, model_id FROM models WHERE status='active' AND task_type='paraphrasing'", get_conn())
                p_model_options = {row['display_name']: row['model_id'] for _, row in p_models_df.iterrows()}
                p_selected_models = st.multiselect("Select models to compare", p_model_options.keys(), max_selections=3, default=list(p_model_options.keys())[:2] if len(p_model_options) >= 2 else list(p_model_options.keys()))
                p_submitted = st.form_submit_button("🎨 Analyze Paraphrases", use_container_width=True)

            if p_submitted and p_selected_models and p_input_text:
                with st.spinner("Running paraphrasing and analysis..."):
                    p_results = {}; threads = []
                    for model_display_name in p_selected_models:
                        model_id = p_model_options[model_display_name]
                        thread = threading.Thread(target=run_comparison_task, args=(model_id, "Paraphrasing", p_input_text, p_results))
                        threads.append(thread); thread.start()
                    for thread in threads: thread.join()

                st.markdown("---")
                st.subheader("Results")

                if len(p_selected_models) == 2:
                    # --- HEAD-TO-HEAD VIEW ---
                    m1_display, m2_display = p_selected_models[0], p_selected_models[1]
                    m1_id, m2_id = p_model_options[m1_display], p_model_options[m2_display]
                    m1_res, m2_res = p_results.get(m1_id, {}), p_results.get(m2_id, {})

                    if "error" in m1_res or "error" in m2_res:
                        st.error(f"An error occurred. M1: {m1_res.get('error', 'OK')}, M2: {m2_res.get('error', 'OK')}")
                    else:
                        m1_metrics, m2_metrics = m1_res['metrics'], m2_res['metrics']
                        m1_sim = calculate_semantic_similarity(p_input_text, m1_res['output'])
                        m2_sim = calculate_semantic_similarity(p_input_text, m2_res['output'])
                        win_counts = {'m1': 0, 'm2': 0}

                        def style_metric(val1, val2, higher_is_better=True, is_time=False):
                            # (Same helper function as above, defined locally for this scope)
                            style_win = "color: #28a745; font-weight: bold;"
                            v1_str = f"{val1:.2f}" if not is_time else f"{val1:.2f} s"
                            v2_str = f"{val2:.2f}" if not is_time else f"{val2:.2f} s"
                            if (higher_is_better and val1 > val2) or (not higher_is_better and val1 < val2):
                                win_counts['m1'] += 1
                                return f'<span style="{style_win}">{v1_str}</span>', v2_str
                            elif (higher_is_better and val2 > val1) or (not higher_is_better and val2 < val1):
                                win_counts['m2'] += 1
                                return v1_str, f'<span style="{style_win}">{v2_str}</span>'
                            return v1_str, v2_str

                        time_s = style_metric(m1_metrics['execution_time'], m2_metrics['execution_time'], higher_is_better=False, is_time=True)
                        sim_s = style_metric(m1_sim, m2_sim, higher_is_better=True)
                        ease_s = style_metric(m1_metrics['flesch_ease'], m2_metrics['flesch_ease'], higher_is_better=True)

                        col1, col2 = st.columns(2)
                        for i, (display, mid, res, sim) in enumerate([(m1_display, m1_id, m1_res, m1_sim), (m2_display, m2_id, m2_res, m2_sim)]):
                             with (col1 if i == 0 else col2):
                                st.markdown(f"#### {display}")
                                if st.button("Set as My Default", key=f"p_set_default_{mid}", use_container_width=True):
                                    update_user_default_model(st.session_state.user, mid)
                                    st.success(f"'{display}' is now your default model!")
                                with st.expander("Show Word-level Changes"):
                                    st.components.v1.html(render_text_diff(p_input_text, res['output']), height=200, scrolling=True)
                                st.markdown(f"""
                                | Metric | Score |
                                |---|---|
                                | **Execution Time** | {time_s[i]} |
                                | **Semantic Similarity** | {sim_s[i]} % |
                                | **Flesch Reading Ease** | {ease_s[i]} |
                                """, unsafe_allow_html=True)

                        st.markdown("---")
                        st.subheader("🏆 Head-to-Head Verdict")
                        if win_counts['m1'] > win_counts['m2']: winner = m1_display
                        elif win_counts['m2'] > win_counts['m1']: winner = m2_display
                        else: winner = "It's a Tie!"
                        st.success(f"**Best Overall Performance:** {winner} ({max(win_counts.values())} wins)")

                else: # --- DEFAULT VIEW FOR 1 OR 3 MODELS ---
                    cols = st.columns(len(p_selected_models))
                    for i, model_display_name in enumerate(p_selected_models):
                        with cols[i]:
                            model_id = p_model_options[model_display_name]
                            res = p_results.get(model_id, {})
                            if "error" in res: st.error(f"**{model_display_name}** failed."); continue
                            st.markdown(f"##### {model_display_name}")
                            if st.button("Set as My Default", key=f"p_set_default_{model_id}", use_container_width=True):
                                update_user_default_model(st.session_state.user, model_id)
                                st.success(f"'{model_display_name}' is now your default model!")

                            metrics = res['metrics']
                            inner_tabs = st.tabs(["📝 Output", "📊 Analytics"])
                            with inner_tabs[0]:
                                st.metric("Execution Time", f"{metrics['execution_time']:.2f} s")
                                st.write(res['output'])
                            with inner_tabs[1]:
                                similarity_score = calculate_semantic_similarity(p_input_text, res['output'])
                                st.plotly_chart(create_gauge_chart(similarity_score, "Semantic Similarity", 100, "#00CC96"), use_container_width=True)
                                with st.expander("Show Word-level Changes"):
                                    st.components.v1.html(render_text_diff(p_input_text, res['output']), height=300, scrolling=True)

    # --- TAB 4: My Analytics ---
    with tab4:
        # NEW: Updated query to fetch readability scores
        df = pd.read_sql_query("""
              SELECT timestamp, task_type, input_text, output_text, sentiment,
                    flesch_reading_ease, flesch_kincaid_grade, gunning_fog, smog_index,
                    rouge1, rouge2, rougeL
              FROM submissions
              WHERE username=? ORDER BY id DESC
          """, get_conn(), params=(st.session_state.user,))

        if not df.empty:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            metric_cols = st.columns(3)
            with metric_cols[0]:
                st.metric(label="Total Submissions", value=len(df))
            with metric_cols[1]:
                most_common_sentiment = df['sentiment'].mode()[0]
                st.metric(label="Most Common Sentiment", value=most_common_sentiment)
            with metric_cols[2]:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                latest_submission = df['timestamp'].max().strftime("%b %d, %Y")
                st.metric(label="Last Submission", value=latest_submission)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="feature-card" style="margin-top: 1rem;">', unsafe_allow_html=True)
            st.markdown('<p class="sub-header">Submissions History</p>', unsafe_allow_html=True)

            # NEW: Reordered columns to show scores in the dataframe
            df_display = df[['timestamp', 'task_type', 'input_text', 'output_text', 'flesch_kincaid_grade', 'gunning_fog', 'smog_index']]
            # Round the scores for cleaner display
            for col in ['flesch_kincaid_grade', 'gunning_fog', 'smog_index']:
                df_display[col] = df_display[col].round(2)

            st.dataframe(df_display, use_container_width=True)

            st.markdown("---")
            export_cols = st.columns(2)
            with export_cols[0]:
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download Data (CSV)", data=csv, file_name="my_submissions.csv", use_container_width=True)
            with export_cols[1]:
                pdf_data = export_pdf(df) # Pass original df to export function
                st.download_button("Download Data (PDF)", data=pdf_data, file_name="my_submissions.pdf", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("You haven't submitted any text yet. Go to the Home page to start analyzing!")

    with tab5:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Team Workspaces</p>', unsafe_allow_html=True)
        st.write("Collaborate with your colleagues, chat in real-time, and leverage AI as a team.")
        if st.button("Go to My Teams Hub", use_container_width=True, type="primary"):
            st.session_state.page = "teams"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ---------- Admin Dashboard Page ----------
elif st.session_state.page == "admin":
    st.markdown('<h1 class="main-header" style="text-align: center;">Admin Dashboard</h1>', unsafe_allow_html=True)

    # --- Admin Header ---
    admin_header_cols = st.columns([3, 1, 1])
    with admin_header_cols[0]:
        st.markdown(f"#### Welcome, {st.session_state.user}!")

    with admin_header_cols[1]:
        # --- THIS IS THE CORRECTED BUTTON ---
      if st.button("💾 Download All Data", use_container_width=True, type="primary", key="download_all_data_btn"):
          with st.spinner("📦 Zipping all database tables... This may take a moment."):
              zip_bytes = generate_full_database_backup_zip()
              st.session_state.download_zip = zip_bytes # Store in session state

    with admin_header_cols[2]:
        if st.button("🚪 Logout", use_container_width=True, help="Logout from admin panel"):
            st.session_state.user = None
            st.session_state.page = "login"
            st.rerun()

    if 'download_zip' in st.session_state and st.session_state.download_zip:
        st.download_button(
            label="📥 Click to Download .zip File",
            data=st.session_state.download_zip,
            file_name=f"textmorph_full_backup_{datetime.datetime.now().strftime('%Y%m%d')}.zip",
            mime="application/zip",
            use_container_width=True,
            key="final_download_zip" # Good practice to key this too
        )
        # Clear the state after the button is shown
        del st.session_state.download_zip

    st.markdown("---")

    conn = sqlite3.connect("users.db")

    # --- NEW: Tabbed Interface for Admin sections ---
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["📊 Platform Analytics", "👥 User Management", "🤖 AI Model Management", "📜 All Submissions", "SMTP and OTP Settings", "🗄️ Database Management", "📢 Broadcast", "🎫 Tickets"])

    # --- TAB 1: Platform Analytics ---
    with tab1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Key Metrics</p>', unsafe_allow_html=True)
        metric_cols = st.columns(2)
        with metric_cols[0]:
            total_users = pd.read_sql_query("SELECT COUNT(*) FROM users", conn).iloc[0, 0]
            st.metric(label="Total Registered Users", value=total_users)
        with metric_cols[1]:
            total_submissions = pd.read_sql_query("SELECT COUNT(*) FROM submissions", conn).iloc[0, 0]
            st.metric(label="Total Text Submissions", value=total_submissions)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="feature-card" style="margin-top: 1rem;">', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">User Demographics</p>', unsafe_allow_html=True)
        chart_cols = st.columns(2)
        with chart_cols[0]:
            lang_counts = pd.read_sql_query("SELECT language, COUNT(*) as count FROM users GROUP BY language", conn)
            if not lang_counts.empty:
                fig = px.pie(lang_counts, values="count", names="language", title="User Language Distribution", hole=0.3)
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white" if st.session_state.theme == 'dark' else 'black')
                st.plotly_chart(fig, use_container_width=True)
        with chart_cols[1]:
            age_counts = pd.read_sql_query("SELECT age_category, COUNT(*) as count FROM users GROUP BY age_category", conn)
            if not age_counts.empty:
                fig = px.bar(age_counts, x="age_category", y="count", title="User Age Distribution")
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white" if st.session_state.theme == 'dark' else 'black')
                st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # NEW: Model Usage Pie Chart
        st.markdown('<div class="feature-card" style="margin-top: 1rem;">', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">AI Model Usage Distribution</p>', unsafe_allow_html=True)
        model_usage_df = pd.read_sql_query("SELECT model_id_used, COUNT(*) as usage_count FROM submissions GROUP BY model_id_used", conn)
        if not model_usage_df.empty:
            fig = px.pie(model_usage_df, values="usage_count", names="model_id_used", title="Model Popularity", hole=0.3)
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white" if st.session_state.theme == 'dark' else 'black')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No model usage data has been recorded yet.")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- TAB 2: User Management ---
    with tab2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Manage Users</p>', unsafe_allow_html=True)
        users_df = pd.read_sql_query("SELECT username, name, email, age_category, language, is_active FROM users WHERE username NOT IN ('shashi', 'pranavi')", conn)

        if not users_df.empty:
            for _, user in users_df.iterrows():
                with st.expander(f"{user['name']} ({user['username']}) - Status: {'Active' if user['is_active'] else 'Blocked'}"):
                    st.write(f"**Email:** {user['email']}")
                    st.write(f"**Age Category:** {user['age_category']}")

                    btn_cols = st.columns(2)
                    with btn_cols[0]:
                        if user['is_active']:
                            if st.button(f"Block User", key=f"block_{user['username']}", use_container_width=True):
                                toggle_user_status(user['username'], 0)
                                st.success(f"User {user['username']} has been blocked.")
                                st.rerun()
                        else:
                            if st.button(f"Unblock User", key=f"unblock_{user['username']}", use_container_width=True):
                                toggle_user_status(user['username'], 1)
                                st.success(f"User {user['username']} has been unblocked.")
                                st.rerun()
                    with btn_cols[1]:
                        if st.button(f"Delete User", key=f"delete_{user['username']}", use_container_width=True, type="primary"):
                            delete_user(user['username'])
                            st.success(f"User {user['username']} has been deleted.")
                            st.rerun()
        else:
            st.info("No users found.")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- TAB 3: AI Model Management ---
    with tab3:
        TASK_TYPES = ["summarization", "paraphrasing", "question_answering", "text_generation", "multi_task"]

        # --- SECTION 1: ADD NEW MODEL ---
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Add New AI Model</p>', unsafe_allow_html=True)
        with st.form("add_model_form", clear_on_submit=True):
            new_model_id = st.text_input("Model ID (from Hugging Face)", placeholder="e.g., google/pegasus-cnn_dailymail")
            new_display_name = st.text_input("Display Name", placeholder="e.g., Pegasus CNN (Summarization)")
            new_task_type = st.selectbox("Task Type", TASK_TYPES, index=0)
            new_remarks = st.text_area("Remarks/Description", placeholder="Good for news articles")
            new_status = st.selectbox("Status", ["active", "inactive"])

            if st.form_submit_button("Add Model", use_container_width=True, type="primary"):
                if new_model_id and new_display_name:
                    conn_add = get_conn()
                    cur = conn_add.cursor()
                    cur.execute("INSERT INTO models (model_id, display_name, remarks, status, task_type, is_public) VALUES (?, ?, ?, ?, ?, 1)",
                                (new_model_id, new_display_name, new_remarks, new_status, new_task_type))
                    conn_add.commit()
                    conn_add.close()
                    st.success("Model added successfully!")
                    st.rerun()
                else:
                    st.warning("Model ID and Display Name are required.")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="feature-card" style="margin-top: 1rem;">', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Manage Existing Models</p>', unsafe_allow_html=True)

        models_df = pd.read_sql_query("SELECT * FROM models", get_conn())
        models_df['Owner'] = models_df['created_by_username'].fillna('System')

        if not models_df.empty:
            for index, model in models_df.iterrows():
                expander_title = f"{model['display_name']} (Owner: {model['Owner']}) - Status: {model['status']}"
                with st.expander(expander_title):
                    # Each model gets its own form, which is NOT nested.
                    with st.form(f"edit_model_{model['model_id']}"):
                        edited_display_name = st.text_input("Display Name", value=model['display_name'], key=f"dname_{model['model_id']}")
                        edited_task_type = st.selectbox("Task Type", TASK_TYPES, index=TASK_TYPES.index(model['task_type']), key=f"ttype_{model['model_id']}")
                        edited_remarks = st.text_area("Remarks", value=model['remarks'], key=f"rem_{model['model_id']}")
                        edited_status = st.selectbox("Status", ["active", "inactive"], index=["active", "inactive"].index(model['status']), key=f"stat_{model['model_id']}")

                        # Button columns
                        btn_cols = st.columns(4)
                        with btn_cols[0]:
                            if st.form_submit_button("Save Changes", use_container_width=True):
                                conn_update = get_conn()
                                cur = conn_update.cursor()
                                cur.execute("UPDATE models SET display_name=?, remarks=?, status=?, task_type=? WHERE model_id=?",
                                            (edited_display_name, edited_remarks, edited_status, edited_task_type, model['model_id']))
                                conn_update.commit()
                                conn_update.close()
                                st.success("Model updated successfully!")
                                st.rerun()
                        with btn_cols[1]:
                             if st.form_submit_button("Delete Model", use_container_width=True, type="primary"):
                                delete_model(model['model_id'])
                                st.rerun()
                        with btn_cols[2]:
                            if model['task_type'] in ["summarization", "paraphrasing"]:
                                if st.form_submit_button("🎯 Tune Model", use_container_width=True):
                                    st.session_state.selected_model_for_tuning = model['model_id']
                                    st.rerun()
                        with btn_cols[3]:
                            if model['Owner'] != 'System':
                                if st.form_submit_button("🚀 Clone to Public", use_container_width=True):
                                    clone_agent_to_public(model['model_id'])
                                    st.rerun()

                # ---------------- Tuning UI ----------------
                if st.session_state.get("selected_model_for_tuning") == model['model_id']:
                    st.markdown("---")
                    st.subheader(f"Fine-tune Model: {model['display_name']}")

                    # Fetch submissions not yet used for THIS specific model
                    train_df = pd.read_sql_query("""
                        SELECT id, input_text, output_text
                        FROM submissions
                        WHERE id NOT IN (
                            SELECT submission_id FROM training_log WHERE model_id = ?
                        )
                    """, conn, params=(model['model_id'],))

                    if train_df.empty:
                        st.info("✅ No new training data available.")
                    else:
                        st.markdown("### 📂 Available Training Data")
                        selected_ids = []
                        if not train_df.empty:
                            train_df_display = train_df.copy()
                            train_df_display["Select"] = False

                            edited_df = st.data_editor(
                                train_df_display,
                                column_config={
                                    "id": st.column_config.TextColumn("ID"),
                                    "input_text": st.column_config.TextColumn("Input Text", width="medium"),
                                    "output_text": st.column_config.TextColumn("Output Text", width="medium"),
                                    "Select": st.column_config.CheckboxColumn("Use?", default=False),
                                },
                                disabled=["id"],
                                hide_index=True,
                                use_container_width=True,
                            )

                            selected_ids = edited_df.loc[edited_df["Select"] == True, "id"].tolist()
                        else:
                            st.info("✅ No new training data available.")

                        # --- Manual data input ---
                        st.markdown("### ✍️ Add Manual Training Example")
                        manual_input = st.text_area("Manual Input Text")
                        manual_output = st.text_area("Manual Output Text")

                        # --- Build training_data before button ---
                        training_data = []
                        if selected_ids:
                            training_data.extend([
                                (row["input_text"], row["output_text"])
                                for _, row in train_df[train_df["id"].isin(selected_ids)].iterrows()
                            ])

                        if manual_input.strip() and manual_output.strip():
                            training_data.append((manual_input.strip(), manual_output.strip()))

                        # --- Train button ---
                        if st.button("🚀 Train Now", key=f"train_btn_{model['model_id']}"):
                            if not training_data:
                                st.warning("Please select some data or add manual data.")
                            else:
                                st.success("🚀 Training started in background. You can keep using the app.")

                                # kick off training (non-blocking)
                                train_model(
                                    model['model_id'],
                                    training_data,
                                    base_task_type=model['task_type'],
                                    db_path="users.db",
                                    output_dir="fine_tuned_models",
                                    selected_ids=selected_ids  # 👈 pass selected IDs so we can mark them later
                                )


        else:
            st.info("No AI models found in the database.")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- TAB 4: All Submissions ---
    with tab4:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">View All User Submissions</p>', unsafe_allow_html=True)
        df_sub = pd.read_sql_query("SELECT username, timestamp, task_type, model_id_used, input_text, output_text, sentiment FROM submissions ORDER BY id DESC", conn)
        st.dataframe(df_sub, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- TAB 5: SMTP & OTP Settings ---
    with tab5:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">📧 SMTP & OTP Settings</p>', unsafe_allow_html=True)
        cur = conn.cursor()

        # Ensure table exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS smtp_settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT,
                password TEXT,
                registration_msg TEXT,
                reset_msg TEXT
            )
        """)
        conn.commit()

        # Fetch existing settings
        cur.execute("SELECT * FROM smtp_settings LIMIT 1")
        smtp_row = cur.fetchone()

        if smtp_row:
            current_email, current_pass, current_reg_msg, current_reset_msg = smtp_row[1:]
        else:
            # Default placeholders
            current_email, current_pass = "", ""
            current_reg_msg = "Hello {username}, your OTP is {otp}."
            current_reset_msg = "Hello {username}, click here to reset your password: {link}"

        with st.form("smtp_settings_form"):
            smtp_email = st.text_input("SMTP Email (Sender)", value=current_email)
            smtp_pass = st.text_input("SMTP Password", value=current_pass, type="password")
            st.markdown("**✉️ Message Templates (placeholders available: {username}, {otp}, {link})**")
            reg_msg = st.text_area("Registration OTP Message", value=current_reg_msg, height=100)
            reset_msg = st.text_area("Password Reset Message", value=current_reset_msg, height=100)

            if st.form_submit_button("💾 Save Settings", use_container_width=True, type="primary"):
                if smtp_row:
                    cur.execute("""
                        UPDATE smtp_settings
                        SET email=?, password=?, registration_msg=?, reset_msg=?
                        WHERE id=?
                    """, (smtp_email, smtp_pass, reg_msg, reset_msg, smtp_row[0]))
                else:
                    cur.execute("""
                        INSERT INTO smtp_settings (email, password, registration_msg, reset_msg)
                        VALUES (?, ?, ?, ?)
                    """, (smtp_email, smtp_pass, reg_msg, reset_msg))
                conn.commit()
                st.success("✅ SMTP & OTP settings updated successfully!")
                st.rerun()
        st.markdown("### 🛠 Test SMTP Settings")
        test_email = st.text_input("Test Email Address")
        if st.button("📨 Send Test OTP"):
            if test_email:
                otp = "123456"
                success = send_registration_otp(test_email, "Admin", otp)
                if success:
                    st.success("✅ Test OTP sent successfully!")
                else:
                    st.error("❌ Failed to send test email. Check your credentials.")


        st.markdown('</div>', unsafe_allow_html=True)


    # --- TAB 6: Database Management ---
    with tab6:

        # --- Utility functions from the provided script ---

        def db_list_tables(conn):
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
            return [r[0] for r in cur.fetchall()]

        def db_read_table(conn, table_name, limit=None, where_clause=None):
            q = f"SELECT rowid as __rowid__, * FROM \"{table_name}\""
            if where_clause:
                q += f" WHERE {where_clause}"
            if limit:
                q += f" LIMIT {limit}"
            return pd.read_sql_query(q, conn)

        def db_get_schema(conn, table_name):
            cur = conn.cursor()
            cur.execute(f"PRAGMA table_info('{table_name}')")
            rows = cur.fetchall()
            cols = [{"cid": r[0], "name": r[1], "type": r[2], "notnull": r[3], "dflt_value": r[4], "pk": r[5]} for r in rows]
            return pd.DataFrame(cols)

        def db_backup(src_path=DB_PATH):
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            dst = f"{src_path}.backup.{ts}"
            with open(src_path, "rb") as fsrc, open(dst, "wb") as fdst:
                fdst.write(fsrc.read())
            return dst

        def db_drop_column_sqlite(conn, table, col_to_drop):
            cur = conn.cursor()
            cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,))
            row = cur.fetchone()
            if not row:
                raise ValueError("Table not found")
            create_sql = row[0]
            cur.execute(f"PRAGMA table_info('{table}')")
            cols_info = cur.fetchall()
            cols = [c[1] for c in cols_info if c[1] != col_to_drop]
            if not cols:
                raise ValueError("Cannot drop the only column")
            cols_list = ", ".join([f'"{c}"' for c in cols])
            tmp_table = f"{table}__tmp__{int(time.time())}"
            cur.execute(f"CREATE TABLE \"{tmp_table}\" AS SELECT {cols_list} FROM \"{table}\";")
            conn.commit()
            cur.execute(f"DROP TABLE \"{table}\";")
            cur.execute(f"ALTER TABLE \"{tmp_table}\" RENAME TO \"{table}\";")
            conn.commit()
            return True

        # --- UI for the Database Workbench ---
        st.subheader("🗄️ SQLite Workbench")
        st.caption("Perform CRUD operations, edit schemas, and run SQL on `users.db`. Always back up before destructive actions.")

        toolbar_col1, toolbar_col2, toolbar_col3 = st.columns([1, 3, 1])
        with toolbar_col2:
            st.info(f"DB file: `{DB_PATH}` — last modified: {time.ctime(os.path.getmtime(DB_PATH)) if os.path.exists(DB_PATH) else 'n/a'}")
        with toolbar_col3:
            if st.button("💾 Backup DB", key="db_backup_btn"):
                b = db_backup(DB_PATH)
                st.success(f"Backup created: {b}")

        db_conn = get_conn()
        tables = db_list_tables(db_conn)

        left, right = st.columns([1, 3], gap="large")
        with left:
            st.markdown("##### Tables")
            if not tables:
                st.info("No tables found in DB.")
            selected_table = st.selectbox("Select a table to manage", options=[""] + tables, index=0, key="db_table_select")
            st.markdown("---")
            st.markdown("##### Table Actions")
            with st.form("create_table_form"):
                new_table_name = st.text_input("Create table name")
                new_table_cols = st.text_area("Columns (SQL syntax)", placeholder='e.g. id INTEGER PRIMARY KEY, name TEXT')
                if st.form_submit_button("Create Table"):
                    if not new_table_name or not new_table_cols.strip():
                        st.warning("Provide table name and column definitions.")
                    else:
                        try:
                            db_conn.execute(f"CREATE TABLE \"{new_table_name}\" ({new_table_cols});")
                            db_conn.commit()
                            st.success(f"Table '{new_table_name}' created.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")

            st.markdown("##### Import / Export")
            csv_file = st.file_uploader("Upload CSV to append to selected table", type=["csv"], key="db_csv_uploader")
            if st.button("Import CSV"):
                if not selected_table or not csv_file:
                    st.warning("Please select a table and upload a CSV file.")
                else:
                    try:
                        df_csv = pd.read_csv(csv_file)
                        df_csv.to_sql(selected_table, db_conn, if_exists="append", index=False)
                        st.success("CSV data appended successfully.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Import failed: {e}")

            if selected_table:
                df_export = db_read_table(db_conn, selected_table)
                csv_export = df_export.to_csv(index=False).encode("utf-8")
                st.download_button("Export selected table to CSV", data=csv_export, file_name=f"{selected_table}.csv")

        with right:
            st.markdown("##### Table / Data Explorer")
            if selected_table:
                st.markdown(f"**Schema for `{selected_table}`**")
                schema_df = db_get_schema(db_conn, selected_table)
                st.dataframe(schema_df, use_container_width=True)

                st.markdown("**Filter & Edit Data**")
                where_clause = st.text_input("WHERE clause (optional)", key=f"where_{selected_table}")
                limit = st.number_input("Limit rows", min_value=10, value=100, step=10, key=f"limit_{selected_table}")

                df = db_read_table(db_conn, selected_table, limit=limit, where_clause=where_clause.strip() or None)
                st.markdown(f"**Editing {len(df)} rows** (includes `__rowid__` for tracking)")
                df_display = df.copy()
                if "__delete__" not in df_display.columns:
                    df_display["__delete__"] = False

                edited_df = st.data_editor(df_display, use_container_width=True, num_rows="dynamic", key=f"editor_{selected_table}")

                if st.button("Apply Changes to DB", key=f"apply_{selected_table}"):
                    try:
                        orig_df = df.set_index("__rowid__")
                        to_delete = edited_df[edited_df["__delete__"] == True]["__rowid__"].tolist()

                        # Logic to find updates and inserts
                        inserts = []
                        updates = []
                        for idx, row in edited_df.iterrows():
                            rid = row["__rowid__"]
                            if pd.isna(rid):
                                inserts.append({c: row[c] for c in edited_df.columns if c not in ["__rowid__", "__delete__"]})
                            elif rid in orig_df.index and not row.equals(orig_df.loc[rid]):
                                if rid not in to_delete:
                                    updates.append(row)

                        cur = db_conn.cursor()
                        if to_delete:
                            placeholders = ",".join(["?"] * len(to_delete))
                            cur.execute(f'DELETE FROM "{selected_table}" WHERE rowid IN ({placeholders})', to_delete)

                        for row in updates:
                            rid = row["__rowid__"]
                            set_clauses = ", ".join([f'"{c}"=?' for c in orig_df.columns if c != "__rowid__"])
                            params = [row[c] for c in orig_df.columns if c != "__rowid__"]
                            cur.execute(f'UPDATE "{selected_table}" SET {set_clauses} WHERE rowid=?', params + [rid])

                        for row_dict in inserts:
                            cols = ", ".join([f'"{c}"' for c in row_dict.keys()])
                            placeholders = ", ".join(["?"] * len(row_dict))
                            cur.execute(f'INSERT INTO "{selected_table}" ({cols}) VALUES ({placeholders})', list(row_dict.values()))

                        db_conn.commit()
                        st.success(f"Applied changes: {len(to_delete)} deleted, {len(updates)} updated, {len(inserts)} inserted.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to apply changes: {e}")

                st.markdown("---")
                st.markdown(f"**Schema Operations on `{selected_table}`**")
                sc1, sc2 = st.columns(2)
                with sc1:
                    with st.form(f"add_col_{selected_table}"):
                        st.markdown("###### Add Column")
                        col_name = st.text_input("Column name")
                        col_type = st.selectbox("Type", ["TEXT", "INTEGER", "REAL", "BLOB"])
                        if st.form_submit_button("Add Column"):
                            if col_name:
                                try:
                                    db_conn.execute(f'ALTER TABLE "{selected_table}" ADD COLUMN "{col_name}" {col_type}')
                                    st.success(f"Column '{col_name}' added.")
                                    st.rerun()
                                except Exception as e: st.error(e)
                with sc2:
                     with st.form(f"drop_col_{selected_table}"):
                        st.markdown("###### Drop Column")
                        drop_col = st.selectbox("Select column to drop", options=schema_df['name'].tolist())
                        if st.form_submit_button("Drop Column", type="primary"):
                            try:
                                db_backup(DB_PATH)
                                db_drop_column_sqlite(db_conn, selected_table, drop_col)
                                st.success(f"Column '{drop_col}' dropped. A backup was created.")
                                st.rerun()
                            except Exception as e: st.error(e)

                st.markdown("---")
                st.markdown("**Quick SQL Runner**")
                sql_text = st.text_area("SQL Query", value=f'SELECT * FROM "{selected_table}" LIMIT 10;', key=f"sql_{selected_table}")
                if st.button("Run SQL", key=f"run_sql_{selected_table}"):
                    try:
                        if sql_text.strip().lower().startswith("select"):
                            st.dataframe(pd.read_sql_query(sql_text, db_conn), use_container_width=True)
                        else:
                            db_conn.execute(sql_text)
                            db_conn.commit()
                            st.success("SQL executed.")
                            st.rerun()
                    except Exception as e:
                        st.error(f"SQL Error: {e}")

            else:
                st.info("Select a table from the left to get started.")

        # --- TAB 7: Broadcast ---
    with tab7:
        st.subheader("📢 Broadcast a Message to All Users")

        # --- Section 1: Sender Credentials (remains the same) ---
        with st.expander("✉️ Configure Sender Email", expanded=False):
            with st.form("broadcast_smtp_form"):
                st.info("These are the same settings from the SMTP tab, shown here for convenience.")
                current_email, current_pass, _, _ = get_smtp_settings()

                broadcast_sender_email = st.text_input("Sender Email", value=current_email or "")
                broadcast_sender_pass = st.text_input("Sender Password", value=current_pass or "", type="password")

                if st.form_submit_button("Save Sender Credentials"):
                    conn = sqlite3.connect("users.db")
                    cur = conn.cursor()
                    if current_email:
                         cur.execute("UPDATE smtp_settings SET email=?, password=?", (broadcast_sender_email, broadcast_sender_pass))
                    else:
                         cur.execute("INSERT INTO smtp_settings (email, password) VALUES (?, ?)", (broadcast_sender_email, broadcast_sender_pass))
                    conn.commit()
                    conn.close()
                    st.success("Sender credentials updated successfully!")
                    st.rerun()

        # --- Section 2: Compose Message & Select Recipients ---
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)

        users_df = get_all_users_for_broadcast()

        placeholders = ", ".join([f"`{{{col}}}`" for col in users_df.columns if col not in ['password', 'profile_pic']])
        st.info(f"**Available Placeholders:** {placeholders}")

        broadcast_subject = st.text_input("Email Subject", key="broadcast_subject")
        broadcast_message = st.text_area("Email Message Body", height=200, key="broadcast_message")

        st.markdown("---")
        st.markdown("##### Select Recipients")

        # --- NEW: Table-based selection and manual entry ---
        # 1. Select from a table of registered users
        st.write("**Option 1: Select from Registered Users**")
        if not users_df.empty:
            users_df_display = users_df[['username', 'name', 'email']].copy()
            users_df_display.insert(0, "Select", False)
            edited_df = st.data_editor(
                users_df_display,
                hide_index=True,
                column_config={"Select": st.column_config.CheckboxColumn(required=True)},
                disabled=users_df.columns,
                key="user_selection_editor"
            )
            selected_from_table = edited_df[edited_df["Select"]]
        else:
            st.info("No registered users to display.")

        # 2. Manually add email addresses
        st.write("**Option 2: Manually Add Emails**")
        manual_emails_str = st.text_area("Enter single or multiple email addresses (separated by comma, space, or new line):", key="manual_emails")
        st.caption("Note: Personalization placeholders like `{name}` will not work for manually entered emails that are not in the database.")

        if st.button("🚀 Send Broadcast", type="primary", key="send_broadcast_btn"):
            sender_email, sender_pass, _, _ = get_smtp_settings()
            if not sender_email or not sender_pass:
                st.error("Sender credentials are not configured.")
            elif not broadcast_subject or not broadcast_message:
                st.warning("Please provide a subject and a message.")
            else:
                # --- NEW: Logic to build the final recipient list ---
                final_recipients = {} # Use a dictionary to auto-deduplicate by email

                # Add users selected from the table
                if not selected_from_table.empty:
                    for _, row in selected_from_table.iterrows():
                        user_details = users_df[users_df['email'] == row['email']].iloc[0].to_dict()
                        final_recipients[row['email']] = user_details

                # Add and process manually entered emails
                if manual_emails_str:
                    manual_emails = [email.strip() for email in re.split('[,\\s\\n]+', manual_emails_str) if email.strip()]
                    for email in manual_emails:
                        if email not in final_recipients:
                            # Check if the manual email exists in our database
                            matching_user = users_df[users_df['email'] == email]
                            if not matching_user.empty:
                                final_recipients[email] = matching_user.iloc[0].to_dict()
                            else:
                                # Create a generic entry for external emails
                                final_recipients[email] = {'email': email, 'name': 'Valued User', 'username': 'user'}

                recipients_list = list(final_recipients.values())

                if not recipients_list:
                    st.warning("No recipients selected or entered.")
                else:
                    total_recipients = len(recipients_list)
                    st.success(f"Preparing to send broadcast to {total_recipients} unique recipient(s)...")

                    progress_bar = st.progress(0, text="Starting broadcast...")
                    success_count, failure_count = 0, 0

                    for i, user in enumerate(recipients_list):
                        try:
                            p_subject = broadcast_subject.format(**user)
                            p_message = broadcast_message.format(**user)
                        except KeyError as e:
                            st.error(f"Invalid placeholder {e}. Broadcast stopped.")
                            break

                        if send_mail(user['email'], p_subject, p_message):
                            success_count += 1
                            log_broadcast(sender_email, user, p_subject, p_message, "Success")
                        else:
                            failure_count += 1
                            log_broadcast(sender_email, user, p_subject, p_message, "Failed")

                        progress_value = (i + 1) / total_recipients
                        progress_text = f"Sending... {i + 1}/{total_recipients} (✅ Sent: {success_count} | ❌ Failed: {failure_count})"
                        progress_bar.progress(progress_value, text=progress_text)
                    else:
                        st.success("Broadcast complete!")

        st.markdown('</div>', unsafe_allow_html=True)

    # --- TAB 8: Tickets ---
    with tab8:
        st.subheader("🎫 User Support Tickets")

        # Fetch tickets from DB
        conn = sqlite3.connect("users.db")
        tickets_df = pd.read_sql_query("SELECT * FROM tickets ORDER BY status ASC, timestamp DESC", conn)
        conn.close()

        if tickets_df.empty:
            st.info("🎉 No support tickets found. Great job!")
        else:
            # Display metrics
            open_tickets = tickets_df[tickets_df['status'] == 'Open'].shape[0]
            resolved_tickets = tickets_df[tickets_df['status'] == 'Resolved'].shape[0]
            t_col1, t_col2 = st.columns(2)
            t_col1.metric("Open Tickets", open_tickets)
            t_col2.metric("Resolved Tickets", resolved_tickets)
            st.markdown("---")

            for index, ticket in tickets_df.iterrows():
                status_color = "red" if ticket['status'] == 'Open' else "green"
                expander_title = f"Ticket #{ticket['id']} - **{ticket['username']}** - Status: :{status_color}[{ticket['status']}] - Reported on {ticket['timestamp']}"

                with st.expander(expander_title):
                    st.markdown(f"**User's Description:**")
                    if ticket['user_description']:
                        st.warning(ticket['user_description'])
                    else:
                        st.info("User did not provide a description.")

                    st.markdown("---")
                    st.markdown("**Session Snapshot**")

                    # Safely load and display the session data
                    try:
                        session_data = json.loads(ticket['session_data'])

                        s_col1, s_col2 = st.columns(2)
                        s_col1.info(f"**Model Used:** `{session_data.get('selected_model', 'N/A')}`")
                        s_col2.info(f"**Task Performed:** `{session_data.get('task_type', 'N/A')}`")

                        st.markdown("**Input Text:**")
                        st.text_area("Input", value=session_data.get('input_text', ''), height=150, disabled=True, key=f"input_{ticket['id']}")

                        st.markdown("**Output Text:**")
                        st.text_area("Output", value=session_data.get('output_text', ''), height=150, disabled=True, key=f"output_{ticket['id']}")

                        st.markdown("**Readability Scores:**")
                        scores = session_data.get('comparison_scores')
                        if scores and isinstance(scores, dict):
                           score_df = pd.DataFrame(scores).T # Transpose for better view
                           st.dataframe(score_df)
                        else:
                           st.write("Scores not available for this session.")

                    except (json.JSONDecodeError, TypeError):
                        st.error("Could not parse session data.")
                        st.code(ticket['session_data'])

                    # --- Admin Reply Section ---
                    st.markdown("---")
                    if ticket['status'] == 'Open':
                        st.markdown("##### 📧 Reply to User & Resolve Ticket")
                        with st.form(key=f"reply_form_{ticket['id']}"):
                            reply_subject = st.text_input("Subject", value="Message from TextMorph Admin")
                            reply_message = st.text_area("Your Message to the User")

                            if st.form_submit_button("Send Reply & Mark as Resolved", type="primary"):
                                if not reply_message:
                                    st.warning("Please enter a message to send.")
                                else:
                                    if send_mail(ticket['user_email'], reply_subject, reply_message):
                                        resolve_ticket(ticket['id'], reply_subject, reply_message)
                                        st.success("Reply sent and ticket resolved successfully!")
                                        st.rerun()
                                    else:
                                        st.error("Failed to send email. Check SMTP settings.")
                    else: # If ticket is resolved
                        st.success(f"This ticket was resolved on {ticket['resolved_timestamp']}.")
                        st.markdown("**Admin's Reply:**")
                        st.info(f"**Subject:** {ticket['admin_reply_subject']}")
                        st.code(ticket['admin_reply_message'])

    conn.close()

# ---------- Forgot Password Page ----------
elif st.session_state.page == "forgot":
    st.markdown('<h1 class="main-header" style="text-align: center;">Reset Password</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.1rem; color: #CCCCCC;">Enter your email or username to receive a reset link and OTP.</p>', unsafe_allow_html=True)

    _, reset_col, _ = st.columns([1, 1.5, 1])

    with reset_col:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)

        identifier = st.text_input("Email or Username", placeholder="Enter your email or username")

        if st.button("Request Reset", use_container_width=True, type="primary"):
            conn = sqlite3.connect("users.db")
            cur = conn.cursor()

            # Find user by email OR username
            cur.execute("SELECT username, email FROM users WHERE username=? OR email=?", (identifier, identifier))
            user = cur.fetchone()

            if user:
                username, email = user

                # Generate OTP
                import random, string
                otp = ''.join(random.choices(string.digits, k=6))

                # Generate reset link (just an example, you can make dynamic route)
                reset_link = f"http://localhost:8501/?reset_user={username}&otp={otp}"

                # Save OTP in session (or DB for persistence)
                st.session_state.reset_otp = otp
                st.session_state.reset_user = username

                # Send mail using Admin SMTP settings
                try:
                    send_password_reset(email, username, reset_link)  # 🔥 uses your mail function
                    # send_registration_otp(email, username, otp)       # 🔥 OTP mail
                    st.success("✅ Reset instructions have been sent to your email.")
                except Exception as e:
                    st.error(f"❌ Failed to send reset email: {e}")
            else:
                st.error("No account found with that email/username.")

            conn.close()

        # OTP Verification Section
        if "reset_otp" in st.session_state:
            st.markdown("### 🔑 Verify OTP")
            entered_otp = st.text_input("Enter the OTP sent to your email")

            if st.button("Verify OTP", use_container_width=True):
                if entered_otp == st.session_state.reset_otp:
                    st.success("✅ OTP verified successfully!")
                    # Set the session state to remember verification
                    st.session_state.otp_verified = True
                    st.rerun() # Rerun to immediately show the next step
                else:
                    st.error("❌ Invalid OTP.")

            # This block will now display persistently after successful OTP verification
            if st.session_state.get("otp_verified"):
                st.markdown("### 🔒 Set Your New Password")
                new_pass = st.text_input("New Password", type="password")
                confirm_pass = st.text_input("Confirm Password", type="password")

                if st.button("Update Password", use_container_width=True, type="primary"):
                    if new_pass == confirm_pass and new_pass.strip():
                        update_password(st.session_state.reset_user, new_pass)
                        st.success("✅ Password updated successfully! Please log in again.")
                        # Clean up session state after completion
                        st.session_state.pop("reset_otp", None)
                        st.session_state.pop("reset_user", None)
                        st.session_state.otp_verified = False # Reset for next time
                        st.session_state.page = "login"
                        st.rerun()
                    else:
                        st.error("❌ Passwords do not match or are empty.")

        # "Back to Login" button
        if st.button("⬅️ Back to Login", use_container_width=True):
            st.session_state.page = "login"
            animate_transition()
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)


# ---------- Teams Hub Page ----------
elif st.session_state.page == "teams":
    custom_header()
    # --- VVV ADD THIS BUTTON HERE VVV ---
    if st.button("⬅️ Back to Home"):
        st.session_state.page = "home"
        st.rerun()
    st.markdown('<h1 class="main-header" style="text-align: center;">Team Workspaces</h1>', unsafe_allow_html=True)

    user_teams = get_user_teams(st.session_state.user)

    if not user_teams.empty:
        st.subheader("Your Teams")
        for index, team in user_teams.iterrows():
            with st.container():
                st.markdown(f"#### {team['team_name']}")
                st.caption(f"Your Role: {team['role'].capitalize()}")
                if st.button("Open Workspace", key=f"open_{team['id']}", use_container_width=True):
                    st.session_state.page = "view_team"
                    st.session_state.current_team_id = team['id']
                    st.rerun()
                st.markdown("---")
    else:
        st.info("You are not a member of any team yet. Create a new team or join one below.")

    st.subheader("Get Started")
    col1, col2 = st.columns(2)
    with col1:
        with st.form("create_team_form"):
            st.markdown("##### 🏢 Create a New Team")
            new_team_name = st.text_input("Team Name")
            if st.form_submit_button("Create Team", type="primary"):
                if new_team_name:
                    create_team(new_team_name, st.session_state.user)
                    st.success(f"Team '{new_team_name}' created successfully!")
                    st.rerun()
                else:
                    st.warning("Please enter a team name.")
    with col2:
        with st.form("join_team_form"):
            st.markdown("##### 🤝 Join an Existing Team")
            join_code = st.text_input("Enter Join Code")
            if st.form_submit_button("Join Team"):
                if join_code:
                    result = join_team(st.session_state.user, join_code.strip().upper())
                    if result == "Success":
                        st.success("Successfully joined the team!")
                        st.rerun()
                    else:
                        st.error(f"Error: {result}")
                else:
                    st.warning("Please enter a join code.")

# ---------- Team View Page ----------
elif st.session_state.page == "view_team":
    team_id = st.session_state.current_team_id
    team_details = get_team_details(team_id)
    members_df = get_team_members(team_id)
    is_admin = members_df[members_df['username'] == st.session_state.user]['role'].iloc[0] == 'admin'

    update_user_last_seen(st.session_state.user)

    # --- UPDATED: Refined CSS for WhatsApp-style chat bubbles ---
    st.markdown(f"""
    <style>
    .chat-container {{
        display: flex;
        flex-direction: column;
        height: 60vh;
        overflow-y: auto;
        padding: 10px;
        background-color: { '#1e1e1e' if st.session_state.theme == 'dark' else '#ffffff' };
        border: 1px solid {'#444' if st.session_state.theme == 'dark' else '#ddd'};
        border-radius: 10px;
    }}
    .message-wrapper {{
        display: flex;
        flex-direction: column;
        margin-bottom: 10px;
    }}
    .user {{
        align-items: flex-end;
    }}
    .other {{
        align-items: flex-start;
    }}
    .ai {{
        align-items: flex-start;
    }}
    .chat-bubble {{
        display: inline-block;
        max-width: 75%;
        padding: 8px 15px;
        border-radius: 18px;
        word-wrap: break-word;
        width: fit-content;
    }}
    .user .chat-bubble {{
        background-color: #0b93f6;
        color: white;
        border-bottom-right-radius: 2px;
    }}
    .other .chat-bubble {{
        background-color: {'#333' if st.session_state.theme == 'dark' else '#e5e5ea'};
        color: {'white' if st.session_state.theme == 'dark' else 'black'};
        border-bottom-left-radius: 2px;
    }}
    .ai .chat-bubble {{
        background-color: #5d3fd3;
        color: white;
        border: 1px solid #7a5af8;
        border-bottom-left-radius: 2px;
    }}
    .sender-info {{
        font-size: 0.8rem;
        color: grey;
        margin-bottom: 2px;
    }}
    </style>
    """, unsafe_allow_html=True)

    # --- Header ---
    header_cols = st.columns([3, 1])
    with header_cols[0]:
        st.header(f"Workspace: {team_details['team_name']}")
    with header_cols[1]:
        if st.button("⬅️ Back to Teams Hub"):
            st.session_state.page = "teams"
            del st.session_state.current_team_id
            st.rerun()

    # --- Main Layout ---
    chat_col, dash_col = st.columns([2.5, 1.5])

    with chat_col:
        st.subheader("💬 Team Chat")

        active_models_df = get_active_models()
        model_options = {row['display_name']: row['model_id'] for _, row in active_models_df.iterrows()}
        selected_model_display = st.selectbox("Select AI Model for Chat", model_options.keys(), key="team_chat_model")
        selected_model_id = model_options[selected_model_display]

        # --- UPDATED: Build chat history as a single HTML block ---
        chat_history = get_team_messages(team_id)
        chat_html = ""
        for index, msg in chat_history.iterrows():
            if msg['message_type'] == 'ai':
                wrapper_class = "ai"
            elif msg['username'] == st.session_state.user:
                wrapper_class = "user"
            else:
                wrapper_class = "other"

            timestamp = pd.to_datetime(msg['timestamp']).strftime('%b %d, %H:%M')
            sender_info = f"<div class='sender-info'>{msg['display_name']} · {timestamp}</div>"
            message_bubble = f"<div class='chat-bubble'>{msg['message']}</div>"

            chat_html += f"<div class='message-wrapper {wrapper_class}'>{sender_info}{message_bubble}</div>"

        # Display the entire chat history inside the container
        st.markdown(f"<div class='chat-container'>{chat_html}</div>", unsafe_allow_html=True)


        # Chat Input logic follows...
        prompt = st.chat_input("Say something or use @AI or @DOC to ask a question...")
        if prompt:
            # --- UPDATED: Add @DOC logic ---
            if prompt.lower().startswith("@doc "):
                doc_question = prompt[5:].strip()
                post_team_message(team_id, st.session_state.user, prompt) # Post user's question

                doc_text, doc_name = get_active_document_text(team_id)
                if not doc_text:
                    ai_response = f"Error: {doc_name}" # doc_name contains the error message here
                else:
                    ai_response = f"Answering based on **{doc_name}**:\n\n"
                    # Use a Q&A model to answer
                    if 'gemini' in selected_model_id:
                        api_key = team_details.get('gemini_api_key') or get_user_gemini_key(st.session_state.user)
                        if not api_key:
                           ai_response += "Gemini API key is not set."
                        else:
                           ai_response += answer_question_with_gemini(api_key, doc_text, doc_question)
                    else: # Hugging Face Q&A
                        qa_pipeline = load_qa_model(selected_model_id)
                        ai_response += answer_question(qa_pipeline, doc_text, doc_question)

                post_team_message(team_id, selected_model_display, ai_response, message_type='ai')

            elif prompt.lower().startswith("@ai "):
                ai_query = prompt[4:].strip()
                post_team_message(team_id, st.session_state.user, prompt)

                ai_response = ""
                if ai_query.lower() in ["what is the date today?", "date", "today's date"]:
                    ai_response = f"Today's date is {datetime.datetime.now().strftime('%A, %B %d, %Y')}."
                else:
                    if 'gemini' in selected_model_id:
                        api_key = team_details.get('gemini_api_key') or get_user_gemini_key(st.session_state.user)
                        if not api_key:
                            ai_response = "Error: Gemini API key is not set for this team or your personal account."
                        else:
                            ai_response = generate_text_with_gemini(api_key, ai_query)
                    else:
                        pipeline = load_text_generation_model(selected_model_id)
                        ai_response = generate_text(pipeline, ai_query)

                post_team_message(team_id, selected_model_display, ai_response, message_type='ai')
            else:
                post_team_message(team_id, st.session_state.user, prompt)

            st.rerun()

    with dash_col:
        # ... (The dashboard column code remains the same) ...
        st.subheader("⚙️ Team Dashboard")
        # --- VVV ADD THIS NEW UPLOAD SECTION VVV ---
        with st.expander("📄 Document Hub", expanded=True):
            uploaded_file = st.file_uploader(
                "Upload a document for the team to discuss",
                type=['txt', 'pdf', 'docx']
            )
            if uploaded_file is not None:
                if st.button("Process Document"):
                    with st.spinner("Analyzing document, please wait..."):
                        doc_text = handle_document_upload(team_id, st.session_state.user, uploaded_file)

                        # --- VVV UPDATED: Use the correct model VVV ---
                        # Prioritize the team's default model, otherwise use the chat's selected model
                        summary_model_id = team_details.get('default_model_id') or selected_model_id

                        summary = ""
                        try:
                            # Attempt to load the selected model as a summarizer
                            summarizer, tokenizer = load_summarizer(summary_model_id)
                            if summarizer and tokenizer:
                                summary = summarize_text(summarizer, tokenizer, doc_text, max_length=150)
                            else:
                                # This handles cases where a non-summarization model was selected
                                summary = f"Error: The selected model (`{summary_model_id}`) is not a valid summarization model. An admin can set a default summarization model in the dashboard settings."
                        except Exception as e:
                            summary = f"An error occurred during summarization: {e}"
                        # --- END UPDATE ---

                        uploader_name = st.session_state.user
                        post_team_message(team_id, "System",
                                          f"**{uploader_name}** uploaded a new document: **{uploaded_file.name}**\n\n**AI Summary:**\n{summary}",
                                          message_type='ai')
                    st.success("Document processed and summary posted to chat!")
                    st.rerun()

        with st.container():
            st.markdown("**Team Info & Members**")
            online_count = 0
            for index, member in members_df.iterrows():
                status_icon = "🟢" # Online
                if member['last_seen']:
                    last_seen_time = pd.to_datetime(member['last_seen'])
                    if (pd.Timestamp.now() - last_seen_time).total_seconds() > 300: # 5 minutes
                        status_icon = "⚪️" # Offline
                else:
                    status_icon = "⚪️" # Offline

                if status_icon == "🟢":
                    online_count += 1

                role_label = " (Admin)" if member['role'] == 'admin' else ""
                st.write(f"{status_icon} **{member['name']}**{role_label}")

            st.metric("Active Members", f"{online_count} Online / {len(members_df)} Total")

            st.markdown("**Join Code**")
            st.code(team_details['join_code'])

            if is_admin:
                with st.expander("👑 Admin Settings"):
                    with st.form("team_settings_form"):
                        st.write("Edit Team Name")
                        new_team_name = st.text_input("New Name", value=team_details['team_name'], label_visibility="collapsed")

                        # --- VVV NEW: Model Selection Dropdown VVV ---
                        st.write("Default AI Model for this Workspace")
                        # Get all active models for the dropdown
                        all_models_df = get_active_models()
                        model_dict = {row['display_name']: row['model_id'] for _, row in all_models_df.iterrows()}
                        model_display_names = list(model_dict.keys())

                        # Find the index of the currently saved model
                        current_model_id = team_details.get('default_model_id')
                        try:
                            current_index = list(model_dict.values()).index(current_model_id)
                        except ValueError:
                            current_index = 0 # Default to the first model if not found

                        new_model_display = st.selectbox("Default Model", options=model_display_names, index=current_index, label_visibility="collapsed")
                        new_model_id = model_dict[new_model_display]
                        # --- END NEW ---

                        st.write("Team Gemini API Key (Optional)")
                        new_api_key = st.text_input("API Key", value=team_details.get('gemini_api_key', ''), type="password", label_visibility="collapsed")
                        st.caption("This key will be used for all `@AI` calls using Gemini in this workspace.")

                        if st.form_submit_button("Save Settings"):
                            update_team_settings(team_id, new_team_name, team_details['join_code'], new_model_id, new_api_key)
                            st.success("Settings updated!")
                            st.rerun()

                        if st.form_submit_button("Generate New Join Code"):
                            new_code = generate_join_code()
                            update_team_settings(team_id, new_team_name, new_code, new_model_id, new_api_key)
                            st.success("New join code generated!")
                            st.rerun()

# ---------- Friends Hub Page ----------
elif st.session_state.page == "friends_hub":
    custom_header()
    # --- VVV ADD THIS BUTTON HERE VVV ---
    if st.button("⬅️ Back to Home"):
        st.session_state.page = "home"
        st.rerun()
    st.markdown('<h1 class="main-header" style="text-align: center;">Friends Hub</h1>', unsafe_allow_html=True)
    update_user_last_seen(st.session_state.user)

    tab1, tab2, tab3 = st.tabs(["My Friends", "Friend Requests", "Find New Friends"])

    with tab1: # My Friends
        friends_list = get_friends_list(st.session_state.user)
        if friends_list.empty:
            st.info("You haven't added any friends yet. Go to the 'Find New Friends' tab to connect with others!")
        else:
            for _, friend in friends_list.iterrows():
                with st.container():
                    f_cols = st.columns([1, 3])
                    with f_cols[0]:
                        # Display avatar
                        st.markdown(get_profile_avatar(friend), unsafe_allow_html=True)
                    with f_cols[1]:
                        st.markdown(f"**{friend['name']}** (`{friend['username']}`)")

                        b_cols = st.columns(2)
                        if b_cols[0].button("Open Chat", key=f"chat_{friend['username']}", use_container_width=True):
                            st.session_state.page = "direct_message"
                            st.session_state.chatting_with = friend['username']
                            st.rerun()

                        with b_cols[1].popover("Manage", use_container_width=True):
                            st.write(f"Manage your friendship with {friend['name']}:")
                            if st.button("Remove Friend", key=f"remove_{friend['username']}", use_container_width=True):
                                remove_friend(st.session_state.user, friend['username'])
                                st.success(f"{friend['name']} has been removed from your friends list.")
                                time.sleep(1)
                                st.rerun()
                            if st.button("Block User", key=f"block_{friend['username']}", type="primary", use_container_width=True):
                                update_friendship_status(st.session_state.user, friend['username'], 'blocked', st.session_state.user)
                                st.error(f"{friend['name']} has been blocked.")
                                time.sleep(1)
                                st.rerun()

                    st.markdown("---")

    with tab2: # Friend Requests
        requests = get_friend_requests(st.session_state.user)
        if requests.empty:
            st.info("You have no pending friend requests.")
        else:
            for _, req in requests.iterrows():
                requester = req['action_user_username']
                with st.container():
                    st.markdown(f"**{requester}** wants to be your friend.")
                    st.info(f"Message: \"{req['initial_message']}\"")

                    req_cols = st.columns(2)
                    with req_cols[0]:
                        if st.button("Accept", key=f"accept_{requester}", type="primary", use_container_width=True):
                            update_friend_request(st.session_state.user, requester, 'accepted')
                            st.success(f"You are now friends with {requester}!")
                            st.rerun()
                    with req_cols[1]:
                        if st.button("Decline", key=f"decline_{requester}", use_container_width=True):
                            update_friend_request(st.session_state.user, requester, 'declined')
                            st.warning(f"Request from {requester} declined.")
                            st.rerun()
                    st.markdown("---")

    with tab3: # Find New Friends
        all_users = get_all_other_users(st.session_state.user)
        for _, user in all_users.iterrows():
            friendship = get_friendship(st.session_state.user, user['username'])
            with st.container():
                st.markdown(f"**{user['name']}** (`{user['username']}`)")

                if friendship is None:
                    with st.popover("Send Friend Request"):
                        with st.form(f"req_form_{user['username']}"):
                            initial_msg = st.text_input("Say hello! (1 message)")
                            if st.form_submit_button("Send"):
                                send_friend_request(st.session_state.user, user['username'], initial_msg)
                                st.success("Friend request sent!")
                                time.sleep(1)
                                st.rerun()
                elif friendship['status'] == 'pending':
                    st.button("Request Sent", disabled=True, key=f"sent_{user['username']}")
                elif friendship['status'] == 'accepted':
                    st.button("✔️ Friends", disabled=True, key=f"friends_{user['username']}")

                st.markdown("---")

# ---------- Direct Message Page ----------
elif st.session_state.page == "direct_message":
    chat_partner_username = st.session_state.chatting_with
    partner_details = pd.read_sql_query("SELECT * FROM users WHERE username=?", get_conn(), params=(chat_partner_username,)).iloc[0]

    update_user_last_seen(st.session_state.user)

    # --- VVV COMPLETE CSS BLOCK VVV ---
    st.markdown(f"""
    <style>
    .chat-avatar {{ width: 35px; height: 35px; border-radius: 50%; object-fit: cover; margin-right: 10px; }}
    .default-avatar {{ background-color: #555; color: white; display: flex; align-items: center; justify-content: center; font-weight: bold; }}
    .message-entry {{ display: flex; align-items: flex-start; margin-bottom: 10px; }}
    .message-content {{ display: flex; flex-direction: column; }}
    .user .message-entry {{ flex-direction: row-reverse; }}
    .user .message-content {{ align-items: flex-end; }}
    .user .chat-avatar {{ margin-left: 10px; margin-right: 0; }}
    .chat-container {{ height: 70vh; overflow-y: auto; padding: 10px; border-radius: 10px; background-color: { '#262730' if st.session_state.theme == 'dark' else '#f0f2f6' }; }}
    .chat-bubble {{ display: inline-block; max-width: 75%; padding: 8px 15px; border-radius: 18px; word-wrap: break-word; width: fit-content; }}
    .user .chat-bubble {{ background-color: #0b93f6; color: white; border-bottom-right-radius: 2px; }}
    .other .chat-bubble, .ai .chat-bubble {{ border-bottom-left-radius: 2px; }}
    .other .chat-bubble {{ background-color: {'#333' if st.session_state.theme == 'dark' else '#e5e5ea'}; color: {'white' if st.session_state.theme == 'dark' else 'black'}; }}
    .ai .chat-bubble {{ background-color: #5d3fd3; color: white; border: 1px solid #7a5af8; }}
    .sender-info {{ font-size: 0.8rem; color: grey; margin-bottom: 2px; }}
    </style>
    """, unsafe_allow_html=True)

    header_cols = st.columns([2, 2, 2])
    with header_cols[0]:
        st.header(f"Chat with {partner_details['name']}")
    with header_cols[1]:
        friendship = get_friendship(st.session_state.user, chat_partner_username)
        if friendship is not None:
            with st.popover("⚙️ Chat Settings"):
                st.markdown("**Gemini API Key Sharing**")
                u1, u2 = sorted([st.session_state.user, chat_partner_username])
                is_currently_sharing = False
                if st.session_state.user == u1:
                    if friendship.get('shared_key_by_user1'): is_currently_sharing = True
                else:
                    if friendship.get('shared_key_by_user2'): is_currently_sharing = True
                share_toggle = st.toggle(f"Share my API key with {partner_details['name']}", value=is_currently_sharing)
                if share_toggle != is_currently_sharing:
                    update_api_key_sharing(st.session_state.user, chat_partner_username, share_toggle)
                    st.success("Sharing settings updated!")
                    time.sleep(1)
                    st.rerun()
    with header_cols[2]:
        if st.button("⬅️ Back to Friends Hub"):
            st.session_state.page = "friends_hub"
            del st.session_state.chatting_with
            st.rerun()

    active_models_df = get_active_models()
    model_options = {row['display_name']: row['model_id'] for _, row in active_models_df.iterrows()}
    selected_model_display = st.selectbox("Select AI Model for Chat", model_options.keys(), key="dm_chat_model")
    selected_model_id = model_options[selected_model_display]

    # Chat History
    messages = get_direct_messages(st.session_state.user, chat_partner_username)
    with st.container() as chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for _, msg in messages.iloc[::-1].iterrows():
            msg_class = "user-message" if msg['sender_username'] == st.session_state.user else "other-message"
            if msg['message_type'] == 'ai': msg_class = 'ai-message'
            st.markdown(f"**{msg['display_name']}** <small>({pd.to_datetime(msg['timestamp']).strftime('%b %d, %H:%M')})</small>", unsafe_allow_html=True)
            st.markdown(f'<div class="chat-message {msg_class}">{msg["message"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Chat Input
    prompt = st.chat_input("Send a message...")
    if prompt:
        post_direct_message(st.session_state.user, chat_partner_username, prompt)
        if prompt.lower().startswith("@ai "):
            ai_query = prompt[4:].strip()
            if 'gemini' in selected_model_id:
                api_key = get_effective_api_key(st.session_state.user, chat_partner_username)
                if not api_key:
                    ai_response = f"Error: No Gemini API key is available."
                else:
                    ai_response = generate_text_with_gemini(api_key, ai_query)
            else:
                pipeline = load_text_generation_model(selected_model_id)
                ai_response = generate_text(pipeline, ai_query)
            post_direct_message(selected_model_display, chat_partner_username, ai_response, message_type='ai')
        st.rerun()

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
