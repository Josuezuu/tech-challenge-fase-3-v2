# HANDOFF — continue por aqui

Repositório: https://github.com/Josuezuu/tech-challenge-fase-3-v2 (privado)

O pedido original que gerou este repositório está em `PROMPT-fase3-v2.md`, na raiz.
Se for abrir uma conversa nova com o Claude Code neste computador, mande ler esse
arquivo primeiro — ele tem todo o contexto: por que este projeto existe, o que
reaproveitar do projeto anterior, o método de avaliação a reproduzir, as decisões já
tomadas (não reabrir) e as restrições de hardware.

## Estado atual (atualizado nesta máquina — a "outro computador" do handoff original)

**Ambiente completo instalado** (`pip install -e ".[dev,train]"`), `pytest -q` passa
inteiro: **59 testes**, incluindo `rag.py`, `graph.py`, `llm.py` e `evaluation.py` —
os módulos que dependiam de `langchain`/`langgraph` e não tinham sido exercitados
de verdade. `ruff check src tests scripts` também está limpo.

**Pipeline de dados rodou ponta a ponta com dados reais**, não só sintéticos:
`python -m scripts.run_prepare_data` com PubMedQA (convertido para
`data/raw/pubmedqa.jsonl` — fora do git, ver `data/raw/FONTES.md` para reconverter) e
MedQuAD clonado em `data/raw/medquad/` (fora do git). Resultado em
`data/processed/relatorio_preparo.json`: 124 exemplos curados, divididos em
**105 treino / 19 holdout** sem vazamento (`data/processed/train.jsonl` e
`holdout.jsonl`, versionados).

**Índice FAISS já construído** (`python -m scripts.run_ingest`) em
`data/processed/faiss_index/` (fora do git, ~1GB de embeddings baixados).

Um bug real foi encontrado e corrigido: o nó `recusar` do grafo (`src/graph.py`)
sobrescrevia a recusa específica de prescrição com a mensagem genérica de
fora-de-escopo. Ficou registrado em
`tests/test_graph.py::test_grafo_bloqueia_prescricao_e_preserva_a_mensagem_especifica`
como teste de regressão. Um segundo bug menor: o regex de detecção de pedido de
receita em `src/safety.py` não pegava "passar **a** receita" (só "passar uma
receita") — corrigido.

`src/finetune.py` agora também mede loss de validação durante o treino: o holdout
entra como `eval_dataset` do `Trainer` (só mede, não recebe gradiente — não viola a
garantia de holdout nunca alimentar o treino) e `docs/loss.json` grava as duas curvas
(`treino` e `validacao`), não só uma.

**Nada foi treinado ainda, nada foi avaliado.** Os números em `docs/RELATORIO.md`
são todos placeholder (`—`) — não existe `docs/evaluation_results.json` nem
`docs/loss.json` ainda. **O próximo passo real do projeto é rodar o treino.**

## Antivírus corporativo — regra que continua valendo

Nesta máquina, o Cylance (ou proteção parecida) mata processo longo disparado de
dentro da sessão do Claude Code. Por isso: quem dispara `scripts.run_finetune` e
`scripts.run_evaluation` (os dois pesados, que carregam modelo em GPU) é **você, no
seu terminal — nunca peça para o Claude Code rodar isso diretamente.**

## Ordem de execução — o que falta a partir daqui

Já feito nesta máquina: `pip install -e ".[dev,train]"`, `python -m scripts.check_env`,
`pytest -q`, `run_prepare_data`, `run_ingest`. **Próximo passo real:**

```
python -m scripts.run_finetune --etapa train
```
(QLoRA local — GPU, roda no seu terminal, não na sessão do Claude Code)

```
python -m scripts.run_finetune --etapa merge
```

Depois disso falta exportar para GGUF via llama.cpp e criar o modelo no Ollama —
`src/finetune.py:instrucoes_ollama()` imprime o passo a passo, e o mesmo texto
aparece se `src/llm.py` tentar falar com um provider Ollama que não existe ainda.

```
python -m scripts.run_evaluation --n-geracao 50
```

Isso escreve `docs/evaluation_results.json`, e é dali que todo número de
`docs/RELATORIO.md` deve ser copiado — nunca digitado à mão.

## Decisões já tomadas — não reabrir (ver PROMPT-fase3-v2.md para o porquê)

- Modelo base `microsoft/Phi-3-mini-4k-instruct`, 4 bits NF4.
- `target_modules = ["qkv_proj", "o_proj"]` — nunca `q_proj`/`v_proj` de tutorial de Llama.
- r=16, alpha=32, dropout=0.05, lr=2e-4, 3 épocas, batch 1, grad_accum 16, seq 768, seed 42.
- LLM de produção no grafo é `openai`; `finetuned`/`base` só entram no comparativo.
- Nada de spec-driven, PRD, IDs de requisito — comentário explica o que a função faz, e só.

## O que NÃO está neste repositório

O projeto anterior (`tech-challenge-fase-3`, 928 testes, relatório completo) fica só
nesta máquina, em `C:\Users\josue.oliveira\source\repos\tech-challenge-fase-3` — não
foi clonado nem referenciado aqui além de inspiração de código já incorporada. Se
precisar consultá-lo de novo, é só neste computador.
