"""QLoRA do Phi-3-mini sobre o dataset do HSA — RODE NO SEU TERMINAL, NAO AQUI.

Processo longo em GPU morre se disparado de dentro da sessao do Claude Code
(o antivirus mata o processo). Este modulo so define as funcoes; quem chama
`treinar()` e o script `scripts/run_finetune.py`, executado no terminal do
usuario.

`inspecionar_target_modules()` existe porque todo tutorial de LoRA para Llama
usa `q_proj`/`v_proj`. Copiar isso para o Phi-3 produz um treino que RODA, NAO
DA ERRO, E NAO APRENDE NADA — o Phi-3 funde Q/K/V num unico `qkv_proj`, e o
PEFT nao reclama de um alvo que nao existe. Por isso o codigo confere os
nomes reais dos modulos antes de treinar, em vez de confiar no nome do
tutorial.
"""

from __future__ import annotations

import json
from pathlib import Path

from src import config
from src.data import Exemplo
from src.prompt_template import formatar_treino


def montar_dataset_texto(exemplos: list[Exemplo]):
    """Uma linha de texto por exemplo, ja no formato final de treino."""
    from datasets import Dataset

    return Dataset.from_dict({"text": [formatar_treino(e) for e in exemplos]})


def inspecionar_target_modules(modelo) -> list[str]:
    """Lista os nomes reais dos modulos Linear do modelo carregado.

    Chame ANTES de montar o LoraConfig e confira que `config.LORA_TARGET_MODULES`
    e um subconjunto do que aparece aqui. Se `qkv_proj` nao aparecer, o
    modelo carregado nao e o Phi-3 que o projeto espera.
    """
    nomes = set()
    for nome, modulo in modelo.named_modules():
        if modulo.__class__.__name__ in ("Linear", "Linear4bit"):
            nomes.add(nome.split(".")[-1])

    faltando = set(config.LORA_TARGET_MODULES) - nomes
    if faltando:
        raise ValueError(
            f"target_modules {sorted(faltando)} nao existem no modelo carregado. "
            f"Modulos Linear encontrados: {sorted(nomes)}. "
            "Isso e o erro silencioso do LoRA: treino roda, nao aprende nada."
        )
    return sorted(nomes)


def carregar_modelo_base_4bit():
    """Phi-3-mini em NF4, double quant, compute bf16 — cabe em 8GB com folga."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    modelo = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL_ID,
        quantization_config=bnb_config,
        attn_implementation="sdpa",  # flash-attn nao compila bem no Windows
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_ID)
    return modelo, tokenizer


def montar_lora(modelo):
    from peft import LoraConfig, get_peft_model

    inspecionar_target_modules(modelo)
    hp = config.HIPERPARAMETROS_LORA
    lora_config = LoraConfig(
        r=hp["r"],
        lora_alpha=hp["lora_alpha"],
        lora_dropout=hp["lora_dropout"],
        target_modules=list(config.LORA_TARGET_MODULES),
        task_type="CAUSAL_LM",
        bias="none",
    )
    return get_peft_model(modelo, lora_config)


def treinar(
    caminho_dataset: Path,
    caminho_holdout: Path = config.DATASET_HOLDOUT,
    saida_dir: Path = config.ADAPTER_DIR,
) -> dict:
    """Treino QLoRA completo. Salva checkpoints intermediarios e as curvas de loss.

    O holdout entra so como `eval_dataset` (mede loss, nao recebe gradiente) —
    isso nao viola a garantia de que o holdout nunca alimenta o treino.
    """
    from transformers import Trainer, TrainingArguments

    from src.data import carregar

    exemplos = carregar(caminho_dataset)
    dataset = montar_dataset_texto(exemplos)

    exemplos_holdout = carregar(caminho_holdout)
    dataset_holdout = montar_dataset_texto(exemplos_holdout)

    modelo, tokenizer = carregar_modelo_base_4bit()
    modelo = montar_lora(modelo)

    hp = config.HIPERPARAMETROS_LORA

    def tokenizar(lote):
        # sem padding: batch_size=1 nao precisa de tamanho uniforme dentro do
        # batch, e forcar max_length aqui inflava o loss (padding vira maioria
        # dos tokens e entrava sem mascara em labels, ver commit que fixou isso).
        saida = tokenizer(
            lote["text"],
            truncation=True,
            max_length=hp["max_seq_length"],
        )
        saida["labels"] = [ids.copy() for ids in saida["input_ids"]]
        return saida

    dataset_tokenizado = dataset.map(tokenizar, batched=True, remove_columns=["text"])
    holdout_tokenizado = dataset_holdout.map(tokenizar, batched=True, remove_columns=["text"])

    args = TrainingArguments(
        output_dir=str(saida_dir),
        num_train_epochs=hp["epochs"],
        per_device_train_batch_size=hp["batch_size"],
        gradient_accumulation_steps=hp["gradient_accumulation_steps"],
        learning_rate=hp["learning_rate"],
        # transformers >=5 fundiu warmup_ratio em warmup_steps: um float < 1 aqui
        # e interpretado como razao, nao contagem de passos.
        warmup_steps=hp["warmup_ratio"],
        save_steps=config.SALVAR_CHECKPOINT_A_CADA_N_PASSOS,
        save_total_limit=10,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=config.SALVAR_CHECKPOINT_A_CADA_N_PASSOS,
        per_device_eval_batch_size=hp["batch_size"],
        bf16=True,
        seed=hp["seed"],
        report_to=[],
    )

    trainer = Trainer(
        model=modelo,
        args=args,
        train_dataset=dataset_tokenizado,
        eval_dataset=holdout_tokenizado,
    )
    resultado = trainer.train()

    saida_dir.mkdir(parents=True, exist_ok=True)
    modelo.save_pretrained(str(saida_dir))
    tokenizer.save_pretrained(str(saida_dir))

    curva_loss_treino = [
        {"step": log["step"], "loss": log["loss"]}
        for log in trainer.state.log_history
        if "loss" in log
    ]
    curva_loss_validacao = [
        {"step": log["step"], "eval_loss": log["eval_loss"]}
        for log in trainer.state.log_history
        if "eval_loss" in log
    ]
    (config.DOCS_DIR / "loss.json").parent.mkdir(parents=True, exist_ok=True)
    (config.DOCS_DIR / "loss.json").write_text(
        json.dumps({"treino": curva_loss_treino, "validacao": curva_loss_validacao}, indent=2),
        encoding="utf-8",
    )

    return {
        "loss_final": resultado.training_loss,
        "curva_loss": curva_loss_treino,
        "curva_loss_validacao": curva_loss_validacao,
    }


def mesclar_adapter_em_cpu(adapter_dir: Path, saida_dir: Path) -> None:
    """Funde o adapter LoRA no modelo base, em CPU — nao precisa de GPU pra isso."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL_ID, device_map="cpu")
    modelo = PeftModel.from_pretrained(base, str(adapter_dir))
    modelo = modelo.merge_and_unload()

    saida_dir.mkdir(parents=True, exist_ok=True)
    modelo.save_pretrained(str(saida_dir))
    AutoTokenizer.from_pretrained(config.BASE_MODEL_ID).save_pretrained(str(saida_dir))


def instrucoes_ollama() -> str:
    """Texto unico — reusado pelo erro do provider Ollama, para as duas versoes nao divergirem."""
    return f"""
O modelo fine-tunado nao vem de registro nenhum — e produzido a partir do adapter treinado:
    1. python -m scripts.run_finetune --etapa merge   (funde o adapter em CPU)
    2. converta o modelo mesclado para GGUF q4_K_M com o llama.cpp
       (convert_hf_to_gguf.py + llama-quantize, fora deste repositorio)
    3. escreva um Modelfile com TEMPLATE de passagem (Ollama nao deve reaplicar
       o chat template do Phi-3 por cima do nosso)
    4. ollama create assistente-medico-v2-finetuned -f models/gguf/Modelfile
    5. confira com: ollama list

O modelo base do comparativo vem pronto do registro publico:
    ollama pull phi3:3.8b-mini-4k-instruct-q4_0
E o MESMO modelo que o adapter treinou ({config.BASE_MODEL_ID}); trocar um sem o
outro faz a tabela de avaliacao medir diferenca de familia, nao aprendizado.
""".strip()
