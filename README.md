# User Feedback → Engineering Insights Pipeline

Convert raw user feedback (app reviews, support tickets, NPS comments) into structured engineering insights using a two-agent AI pipeline. The project evaluates baseline performance with DeepEval, fine-tunes both agents independently, and measures quality improvements after training.

Important Links:
https://github.com/aws-samples/amazon-sagemaker-generativeai
https://huggingface.co/docs/transformers/en/training
https://deepeval.com/docs/metrics-llm-evals

## Overview

### What We Are Building

This project implements a two-agent workflow in n8n:

```text
User Feedback
    ↓
Feedback Classifier (Agent 1)
    ↓
Engineering Insight Writer (Agent 2)
    ↓
Engineering Ticket
```

#### Agent 1 — Feedback Classifier

Reads raw user feedback and extracts:

- Category (bug, feature request, performance issue, unclear)
- Severity
- Affected system
- Platform
- Environment
- Exact key phrases
- Reproduction hints
- Missing technical context

#### Agent 2 — Engineering Insight Writer

Transforms the classification into a structured engineering ticket containing:

- Title
- Ticket type
- Technical summary
- Affected components
- Reproduction steps
- Investigation checklist
- Data gaps
- Priority recommendation

---

## Problem Statement

When the base model (`meta-llama/Llama-3.1-8B-Instruct`) receives vague feedback such as:

> "This app is garbage"

it frequently hallucinates technical details, including:

- Fabricated system names
- Imaginary reproduction steps
- Made-up root causes
- Unsupported engineering conclusions

This produces professional-looking but inaccurate engineering tickets.

---

## Solution

Fine-tuning teaches each agent the correct behavior.

### Feedback Classifier

Instead of inventing information, it learns to:

- Mark feedback as `UNCLEAR`
- Set `AFFECTED_SYSTEM=UNKNOWN`
- Identify missing context

### Engineering Insight Writer

Instead of fabricating tickets, it learns to:

- Output **"Insufficient feedback for engineering action"**
- Highlight missing information
- Recommend contacting the user for clarification

---

## Prerequisites

Before starting, ensure you have:

- Python 3.12 or higher
- pip

---

# Setup Guide

## 1. Install UV

### Windows

Run:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### macOS

Run:

```bash
brew install uv
```

### Linux / macOS

Run:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## 2. Create the Project Directory

Navigate to your project directory:

```bash
cd <project_directory>
```

Initialize the project:

```bash
uv venv
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
uv pip install huggingface-hub torch transformers fastapi typing-extensions typing-inspection peft trl datasets deepeval openai python-dotenv bitsandbytes uvicorn
```

---

## 4. Request Access to the Llama Model at HuggingFace

### Why Access Is Required

#### License Agreement

- Meta requires users to accept licensing terms before downloading or deploying the model.

#### Repository Unlocking

- Llama models are gated on Hugging Face and require approval before download.

#### Token Authorization

- Approval enables Hugging Face access tokens to authenticate model downloads and usage.

### Request Access

1. Visit the model page:

   https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

2. Review the license agreement.
3. Complete the required form.
4. Submit and wait for approval.

---

## 5. Configure the HuggingFace and OpenAI API Key Token

### Option A — Environment Variable

#### macOS / Linux

```bash
export OPENAI_API_KEY=<YOUR_OPENAI_TOKEN>
export HF_TOKEN=<YOUR_HF_TOKEN>
```

### Option B — In Code (e.g. in VSCode)

```python
import os

os.environ["OPENAI_API_KEY"] = "<YOUR_OPENAI_TOKEN>"
os.environ["HF_TOKEN"] = "<YOUR_HF_TOKEN>"
```

---

## 6. Download the Base Model

```bash
python download_hf_model.py
```

> Estimated download time: ~15 minutes

---

## 7. Run the Local Base Model

```bash
python load_run_local_model.py
```

---

## 8. Expose the Model Using ngrok

### Install ngrok

Reference:

https://dashboard.ngrok.com/get-started/setup/mac-os

Install:

```bash
brew install ngrok
```

Configure:

```bash
ngrok config add-authtoken $YOUR_AUTHTOKEN
```

Expose the API:

```bash
ngrok http --host-header=rewrite http://127.0.0.1:8005
```

---

## 9. Set Up n8n

1. Create a free hosted instance:

      https://app.n8n.cloud

      No local installation is required.

2. Import the n8n workflow from ```Fine Tuning n8n - using local LLM.json``` file

3. Set the credential that points to the local model for ```Classifier LLM``` & ```Insight LLM``` nodes. Use the ngrok exposed url. E.g.,

    ```text
    https://spinner-tighten-jeep.ngrok-free.dev
    ```

4. Test the n8n setup by posting "hello" message.

---

## 10. Run Test Feedback Items

1. Submit all 8 feedback examples through the n8n chat interface.
2. Wait for each run to finish (approximately 30–60 seconds).
3. Save every generated output for later evaluation.

---

## 11. Evaluate with DeepEval (Before Fine-Tuning)

### Update Pipeline Outputs

1. Open:

   ```text
   test_feedback_engineering.py
   ```

2. Locate the `PIPELINE_OUTPUTS` section.
3. Replace every placeholder with the corresponding n8n output.
4. Save the file.

### Run Evaluation

```bash
python test_feedback_engineering.py
```

---

## 12. Fine-Tune Both Models

Fine-tuning uses:

- LoRA
- 4-bit quantization

### Step 1 — Train LoRA Adapters

```bash
python fine_tune_training.py
```

> Estimated training time: ~2 hours for each model

### Step 2 — Merge Adapters into the Base Model

```bash
python merge_to_base_model.py
```

This step:

- Loads the base model
- Merges LoRA adapters
- Saves the merged model
- Optionally uploads the model to Hugging Face

---

## 13. Run the Fine-Tuned Models

### Load Fine-Tuned Models

```bash
python load_run_local_finetune_model.py
```

### Expose Endpoints

```bash
ngrok http --host-header=rewrite http://127.0.0.1:8006
```

---

## 14. Update n8n to Use Fine-Tuned Models

### Feedback Classifier Endpoint

Update the Classifier LLM credential to: E.g.,

```text
https://spinner-tighten-jeep.ngrok-free.dev/v2
```

### Engineering Insight Endpoint

Update the Insight LLM credential to:

```text
https://spinner-tighten-jeep.ngrok-free.dev/v3
```

---

## 15. Re-Run Test Feedback Items

1. Run feedback items F1–F8 again.
2. Save all outputs.
3. Open `test_feedback_engineering.py`.
4. Replace all entries in `PIPELINE_OUTPUTS` with the new responses.

---

## 16. Evaluate with DeepEval (After Fine-Tuning)

Run:

```bash
python test_feedback_engineering.py
```

---

## Expected Outcome

After fine-tuning:

- Fewer hallucinated engineering details
- Improved handling of ambiguous feedback
- Better identification of missing information
- More reliable engineering ticket generation
- Higher DeepEval evaluation scores

---

## Technology Stack

- Python
- Hugging Face Transformers
- Meta Llama 3.1
- PEFT
- LoRA
- TRL
- Datasets
- DeepEval
- FastAPI
- ngrok
- n8n

---

## Project Workflow

```text
Raw Feedback
      ↓
Feedback Classifier
      ↓
Structured Classification
      ↓
Engineering Insight Writer
      ↓
Engineering Ticket
      ↓
DeepEval Evaluation
      ↓
Fine-Tuning
      ↓
Re-Evaluation
```
