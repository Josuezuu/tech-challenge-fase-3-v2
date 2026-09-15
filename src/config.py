"""Configuracao central: paths, ids de modelo e a variavel que escolhe a LLM."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- paths -------------------------------------------------------------- #

RAIZ: Path = Path(__file__).resolve().parent.parent
DATA_RAW: Path = RAIZ / "data" / "raw"
DATA_PROCESSED: Path = RAIZ / "data" / "processed"
MODELS_DIR: Path = RAIZ / "models"
LOGS_DIR: Path = RAIZ / "logs"
DOCS_DIR: Path = RAIZ / "docs"

DATASET_TREINO: Path = DATA_PROCESSED / "train.jsonl"
DATASET_HOLDOUT: Path = DATA_PROCESSED / "holdout.jsonl"
RELATORIO_PREPARO: Path = DATA_PROCESSED / "relatorio_preparo.json"
FAISS_INDEX_DIR: Path = DATA_PROCESSED / "faiss_index"
HOSPITAL_DB: Path = DATA_PROCESSED / "hospital.db"
AUDITORIA_LOG: Path = LOGS_DIR / "auditoria.jsonl"
AVALIACAO_JSON: Path = DOCS_DIR / "evaluation_results.json"

# --- modelo base e fine-tuning ------------------------------------------ #

BASE_MODEL_ID: str = "microsoft/Phi-3-mini-4k-instruct"
EOS_TOKEN: str = "<|endoftext|>"

ADAPTER_DIR: Path = MODELS_DIR / "adapter"
CHECKPOINTS_DIR: Path = MODELS_DIR / "checkpoints"
GGUF_PATH: Path = MODELS_DIR / "gguf" / "phi3-medico-v2.q4_k_m.gguf"

# alvo real do Phi-3: Q/K/V fundidos num unico modulo. Nunca copiar
# q_proj/v_proj de tutorial de Llama aqui — roda, nao aprende nada.
LORA_TARGET_MODULES: tuple[str, ...] = ("qkv_proj", "o_proj")

HIPERPARAMETROS_LORA: dict[str, float | int] = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "learning_rate": 2e-4,
    "epochs": 3,
    "batch_size": 1,
    "gradient_accumulation_steps": 16,
    "max_seq_length": 768,
    "warmup_ratio": 0.03,
    "seed": 42,
}

SALVAR_CHECKPOINT_A_CADA_N_PASSOS: int = 20

# --- RAG ------------------------------------------------------------------ #

EMBEDDING_MODEL_ID: str = "intfloat/multilingual-e5-base"
RETRIEVER_K: int = 4

# --- provider de LLM (producao) ------------------------------------------ #

LLM_PROVIDER: str = os.environ.get("LLM_PROVIDER", "fake")
OPENAI_MODEL_ID: str = "gpt-4o-mini"

# --- dados ------------------------------------------------------------------ #

SEED: int = 42
PROPORCAO_HOLDOUT: float = 0.15
MIN_PROPORCAO_RECUSA: float = 0.08
MIN_PROPORCAO_FORA_ESCOPO: float = 0.08
LIMITE_CHARS_RESPOSTA_EXTERNA: int = 2000

# pastas do MedQuAD com <Answer></Answer> vazio por direito autoral do
# MedlinePlus — pular pelo nome, nao tentar limpar.
MEDQUAD_PASTAS_IGNORADAS: tuple[str, ...] = (
    "10_MPlus_ADAM_QA",
    "11_MPlusDrugs_QA",
    "12_MPlusHerbsSupplements_QA",
)
