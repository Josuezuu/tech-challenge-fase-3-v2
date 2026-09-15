# Tech Challenge Fase 3 v2 — Assistente Médico com LLM Customizada

FIAP Pós Tech — 8IADT, Fase 3. Assistente virtual médico com fine-tuning (QLoRA sobre
`Phi-3-mini-4k-instruct`), pipeline LangChain com RAG e fluxo de decisão LangGraph.

**O assistente nunca prescreve.** Ele recomenda com fonte e encaminha para validação
humana.

Este é o projeto v2: um décimo do tamanho do
[anterior](../tech-challenge-fase-3) (12 arquivos em `src/`, não 72), com uma avaliação
de fine-tuning melhor — n≥50 do holdout, IC 95% por bootstrap, checkpoints
intermediários comparados, loss de treino **e** validação.

## O enunciado, item por item

| Requisito | Onde está |
| --- | --- |
| Fine-tuning de LLM com dados médicos | `src/finetune.py`, `scripts/run_finetune.py` |
| Preprocessing, anonimização e curadoria | `src/data.py` |
| Pipeline LangChain integrando a LLM customizada | `src/llm.py`, `src/rag.py` |
| Consulta a base estruturada de prontuários | `src/db.py` |
| Contextualizar a resposta com dados do paciente | `src/graph.py` (nó `carregar_paciente`) |
| Nunca prescrever sem validação humana | `src/safety.py`, guardrails no `src/graph.py` |
| Logging detalhado para auditoria | `src/audit.py` |
| Explainability — fonte em toda resposta | `src/rag.py` (`gerar_resposta_com_fonte`) |
| Fluxos em LangGraph | `src/graph.py` |
| Dataset anonimizado ou sintético | `src/data.py`, `data/raw/FONTES.md` |
| Avaliação e análise dos resultados | `src/evaluation.py`, `docs/evaluation_results.json` |

## Preparar o ambiente

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -e ".[dev]"
python -m scripts.check_env
```

GPU e o extra `[train]` só são necessários para treinar:

```bash
pip install -e ".[train]"
```

## Rodar, na ordem

```bash
python -m scripts.run_prepare_data --pubmedqa data/raw/pubmedqa.jsonl --medquad data/raw/medquad
python -m scripts.run_ingest                      # constroi o indice FAISS (baixa ~1GB na 1a vez)

# treino roda no SEU terminal, nunca disparado pela sessao do Claude Code
python -m scripts.run_finetune --etapa train
python -m scripts.run_finetune --etapa merge

LLM_PROVIDER=fake python -m scripts.run_assistant --pergunta "Qual a conduta para sepse em adulto?"

python -m scripts.run_evaluation --n-geracao 50
```

## Trocar de LLM

`LLM_PROVIDER` tem quatro valores — `openai` (produção), `finetuned` e `base` (via
Ollama, para o comparativo), `fake` (testes, sem rede). Nenhum nó do grafo sabe qual
está ativo; só `src/llm.py` sabe que uma LLM existe.

## Testes

```bash
pytest -q
ruff check src tests scripts
```

## Estrutura

```
src/
├── config.py          # paths, ids de modelo, hiperparametros
├── canon.py            # vocabulario unico do hospital fictício (HSA)
├── data.py             # geradores sinteticos + ingestao externa + pipeline de preparo
├── prompt_template.py  # o UNICO lugar que monta prompt de treino/inferencia
├── db.py               # prontuario sintetico — schema, seed, queries parametrizadas
├── safety.py           # sanitizacao, deteccao de prescricao, disclaimer
├── finetune.py          # QLoRA do Phi-3-mini — rode no seu terminal
├── rag.py               # loader, chunking, FAISS, retriever MMR, geracao com fonte
├── llm.py               # o unico arquivo que sabe que uma LLM existe
├── graph.py             # o grafo LangGraph
├── audit.py             # log auditavel, sem PII
└── evaluation.py        # metricas, bootstrap IC95%, retrieval, RAG x sem-RAG, seguranca
```

## Limitações conhecidas

Preencher depois de rodar `run_evaluation` — ver `docs/RELATORIO.md`.
