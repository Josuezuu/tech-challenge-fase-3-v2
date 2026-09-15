# HANDOFF — continue por aqui

Repositório: https://github.com/Josuezuu/tech-challenge-fase-3-v2 (privado)

O pedido original que gerou este repositório está em `PROMPT-fase3-v2.md`, na raiz.
Se for abrir uma conversa nova com o Claude Code neste computador, mande ler esse
arquivo primeiro — ele tem todo o contexto: por que este projeto existe, o que
reaproveitar do projeto anterior, o método de avaliação a reproduzir, as decisões já
tomadas (não reabrir) e as restrições de hardware.

## Estado atual (o que já existe, o que não)

**Escrito e com sintaxe validada:** os 12 módulos de `src/`, os scripts de `scripts/`,
53 testes em `tests/`, `README.md`, `docs/RELATORIO.md` (esqueleto, sem número
inventado), `data/raw/FONTES.md`.

**Validado de verdade nesta máquina** (rodei sem instalar nada pesado): os módulos que
só usam a biblioteca padrão — `config`, `canon`, `safety`, `data`, `db`, `audit`,
`prompt_template`. O pipeline de dados roda ponta a ponta: gera 70 exemplos
sintéticos, a categoria `recusa` e `fora_escopo` têm 14 respostas distintas cada
(era o bug do projeto anterior — 32 exemplos, 1 resposta só), divide em 59 treino /
11 holdout sem vazamento.

**NÃO validado ainda** (dependem de `langchain`, `langgraph`, `torch` — não instalados
nesta máquina, e instalar aqui seria o tipo de coisa pesada que não deveria rodar
dentro da sessão do Claude Code): `rag.py`, `graph.py`, `llm.py`, `finetune.py`,
`evaluation.py`, e a suíte `pytest` inteira. **Isso é o primeiro passo a fazer no outro
computador** — se algo quebrar ao rodar `pytest`, é a primeira vez que este código
encontra essas bibliotecas de verdade.

Um bug real já foi encontrado e corrigido nessa primeira leva: o nó `recusar` do
grafo (`src/graph.py`) sobrescrevia a recusa específica de prescrição com a
mensagem genérica de fora-de-escopo. Ficou registrado em
`tests/test_graph.py::test_grafo_bloqueia_prescricao_e_preserva_a_mensagem_especifica`
como teste de regressão.

**Nada foi treinado, nada foi avaliado.** Os números em `docs/RELATORIO.md` são todos
placeholder (`—`) — não existe `docs/evaluation_results.json` nem `docs/loss.json`
ainda.

## Primeiro passo no outro computador

```
git clone https://github.com/Josuezuu/tech-challenge-fase-3-v2.git
cd tech-challenge-fase-3-v2
python -m venv .venv
.venv\Scripts\activate
```

Depois, **um comando por vez** (bloco colado corre o risco de rodar pela metade):

```
pip install -e ".[dev]"
```

> Aviso: essa instalação puxa `sentence-transformers`, que traz `torch` como
> dependência — é um download grande (alguns GB) e pode demorar. Normal.

```
python -m scripts.check_env
```

```
pytest -q
```

Me diga o resultado do `pytest` antes de seguir — é a primeira vez que a suíte roda
de verdade. Se `test_graph.py` ou `test_evaluation.py` falharem, provavelmente é
incompatibilidade de versão do `langchain`/`langgraph` (as versões em `pyproject.toml`
são mínimas, não travadas em patch) ou algo que eu não consegui prever sem executar.

## Se este computador também tiver antivírus corporativo

Nesta máquina, o Cylance mata processo longo disparado de dentro da sessão do Claude
Code (build/treino Node ou Python). Se o outro computador tiver proteção parecida,
vale confirmar logo: quem dispara `scripts.run_finetune` e `scripts.run_evaluation`
(os dois pesados, que carregam modelo em GPU) deve ser você, no seu terminal — nunca
peça para o Claude Code rodar isso diretamente.

## Depois que o `pytest` passar — ordem de execução

```
python -m scripts.run_prepare_data --pubmedqa data/raw/pubmedqa.jsonl --medquad data/raw/medquad
```

Isso exige clonar antes:
- PubMedQA: https://github.com/pubmedqa/pubmedqa (MIT) — precisa converter o
  `ori_pqal.json` para `data/raw/pubmedqa.jsonl` (um objeto JSON por linha, campos
  `QUESTION`/`LONG_ANSWER` — ver `src/data.py:carregar_pubmedqa`).
- MedQuAD: https://github.com/abachaa/MedQuAD (CC BY 4.0) — clonar direto em
  `data/raw/medquad/`, a estrutura de pastas é usada como está.

Depois, na ordem:

```
python -m scripts.run_ingest
```
(baixa ~1GB do modelo de embedding na primeira vez)

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
