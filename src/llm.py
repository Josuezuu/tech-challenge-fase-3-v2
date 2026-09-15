"""O unico arquivo que sabe que uma LLM existe.

Nenhum outro modulo importa um cliente de LLM. Trocar de modelo e trocar
`LLM_PROVIDER`; nenhum no do grafo sabe qual ponta esta ativa.

Motivo medido no projeto anterior para o Phi-3 nao ser o orquestrador: um
Phi-3-mini quantizado nao obedece JSON schema de forma confiavel — ele
devolveu `nota: 3` num campo `float` entre 0 e 1 e derrubou parte das
execucoes. Ele serve como gerador (ponta de comparacao), nao como quem decide
rota no grafo. Produção usa `openai`; `finetuned`/`base` existem para o
comparativo de avaliacao.

Se `openai` for pedido sem chave, cai para `fake` e avisa — nunca em
silencio. Provider local indisponivel levanta erro dizendo como resolver, e
nao gasta dinheiro na API por conta propria.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Protocol

from src import config

OLLAMA_URL = "http://localhost:11434/api/generate"


class LLMProvider(Protocol):
    nome: str

    def gerar(self, prompt: str, max_tokens: int = 256) -> str: ...


@dataclass
class ProviderFake:
    """Substituto deterministico — usado pelos testes, sem rede e sem GPU."""

    nome: str = "fake"

    def gerar(self, prompt: str, max_tokens: int = 256) -> str:
        primeira_linha = prompt.strip().splitlines()[-1][:80] if prompt.strip() else ""
        return f"[resposta fake] {primeira_linha}"


@dataclass
class ProviderOpenAI:
    nome: str = "openai"
    modelo: str = config.OPENAI_MODEL_ID

    def gerar(self, prompt: str, max_tokens: int = 256) -> str:
        from langchain_openai import ChatOpenAI

        chat = ChatOpenAI(model=self.modelo, max_tokens=max_tokens, temperature=0.2)
        return chat.invoke(prompt).content


@dataclass
class ProviderOllama:
    """`finetuned` ou `base`, ambos servidos localmente pelo Ollama."""

    nome: str
    modelo_ollama: str

    def gerar(self, prompt: str, max_tokens: int = 256) -> str:
        import requests

        try:
            resposta = requests.post(
                OLLAMA_URL,
                json={
                    "model": self.modelo_ollama,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": max_tokens},
                },
                timeout=120,
            )
            resposta.raise_for_status()
        except requests.RequestException as erro:
            raise RuntimeError(
                f"Ollama nao respondeu para o modelo '{self.modelo_ollama}'. "
                f"Confira 'ollama list' e 'ollama serve'. Erro original: {erro}"
            ) from erro
        return resposta.json()["response"]


@dataclass
class ProviderCheckpointLocal:
    """Carrega um checkpoint intermediario direto via transformers+peft.

    Existe para a avaliacao de checkpoints intermediarios (config.SALVAR_CHECKPOINT_A_CADA_N_PASSOS):
    reexportar cada checkpoint para GGUF/Ollama so para medir seria caro. So
    usado por scripts de avaliacao, roda em GPU local, nunca em producao.
    """

    nome: str
    adapter_dir: str
    _pipeline: object = None

    def _carregar(self):
        if self._pipeline is None:
            from peft import PeftModel

            from src.finetune import carregar_modelo_base_4bit

            base, tokenizer = carregar_modelo_base_4bit()
            modelo = PeftModel.from_pretrained(base, self.adapter_dir)
            self._pipeline = (modelo, tokenizer)
        return self._pipeline

    def gerar(self, prompt: str, max_tokens: int = 256) -> str:
        modelo, tokenizer = self._carregar()
        entradas = tokenizer(prompt, return_tensors="pt").to(modelo.device)
        saida = modelo.generate(**entradas, max_new_tokens=max_tokens, do_sample=False)
        texto = tokenizer.decode(saida[0][entradas["input_ids"].shape[1] :], skip_special_tokens=True)
        return texto


def obter_provider(nome: str | None = None) -> LLMProvider:
    """Fabrica o provider pedido; cai para `fake` com aviso se faltar chave."""
    escolhido = nome or config.LLM_PROVIDER

    if escolhido == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            warnings.warn(
                "LLM_PROVIDER=openai sem OPENAI_API_KEY — caindo para 'fake'.",
                stacklevel=2,
            )
            return ProviderFake()
        return ProviderOpenAI()

    if escolhido == "finetuned":
        return ProviderOllama(nome="finetuned", modelo_ollama="assistente-medico-v2-finetuned")

    if escolhido == "base":
        return ProviderOllama(nome="base", modelo_ollama="phi3:3.8b-mini-4k-instruct-q4_0")

    if escolhido == "fake":
        return ProviderFake()

    raise ValueError(f"LLM_PROVIDER desconhecido: {escolhido!r}")
