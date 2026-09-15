# Relatório Técnico — Assistente Médico com LLM Customizada

**FIAP Pós Tech — 8IADT, Fase 3 · Tech Challenge**

> Esqueleto do relatório. Cada tabela abaixo tem um marcador dizendo de qual campo de
> `docs/evaluation_results.json` (ou `docs/loss.json`, `data/processed/relatorio_preparo.json`)
> ela vem — copie o número de lá, nunca digite um valor novo. Preencher depois de rodar
> `scripts.run_prepare_data`, `scripts.run_finetune` e `scripts.run_evaluation`.

---

## 1. O processo de fine-tuning

### 1.1 Dados

Corpus a partir do cânone único (`src/canon.py`), mais PubMedQA e MedQuAD.

| Artefato | Quantidade |
| --- | --- |
| Protocolos clínicos internos | `len(canon.CONDICOES)` |
| Exemplos sintéticos (FAQ, documento, recusa, fora_escopo) | `relatorio_preparo.json:bruto` menos externos |
| Exemplos externos (PubMedQA + MedQuAD) | `relatorio_preparo.json` |
| Treino / Holdout | `relatorio_preparo.json:treino` / `:holdout` |

Contagem por categoria depois da dedup (a distinta, não só o total — ver `data.contar_categorias`):

| Categoria | Total | Respostas distintas |
| --- | --- | --- |
| faq | — | — |
| documento | — | — |
| recusa | — | — |
| fora_escopo | — | — |
| externo_pubmedqa | — | — |
| externo_medquad | — | — |

### 1.2 Hiperparâmetros

De `src/config.py:HIPERPARAMETROS_LORA` e `LORA_TARGET_MODULES`.

| Parâmetro | Valor |
| --- | --- |
| Modelo base | `microsoft/Phi-3-mini-4k-instruct`, 4 bits NF4, double quant |
| `target_modules` | `qkv_proj`, `o_proj` |
| r / alpha / dropout | 16 / 32 / 0,05 |
| Learning rate / épocas | 2e-4 / 3 |
| Batch efetivo | 16 (1 × grad accum 16) |
| `max_seq_length` | 768 |
| Seed | 42 |

### 1.3 Curva de loss (treino e validação)

De `docs/loss.json`. Gráfico em `docs/diagramas/loss.png` (gerar a partir do JSON).

---

## 2. Avaliação de geração — baseline × fine-tunado × checkpoint intermediário

De `docs/evaluation_results.json:geracao`. **n≥50**, decodificação gulosa, mesmo
template do treino e da inferência.

| Ponta | n | ROUGE-L | BLEU-4 | chars (médio) |
| --- | --- | --- | --- | --- |
| base (sem adapter) | — | — | — | — |
| finetuned | — | — | — | — |
| checkpoint intermediário | — | — | — | — |

### IC 95% por bootstrap (10.000 reamostras) no delta

| Comparação | Métrica | Pontual | IC 95% | P(Δ>0) |
| --- | --- | --- | --- | --- |
| finetuned − base | BLEU-4 | — | — | — |
| finetuned − base | ROUGE-L | — | — | — |
| finetuned − checkpoint | BLEU-4 | — | — | — |
| finetuned − checkpoint | ROUGE-L | — | — | — |

> Se o intervalo cruzar zero, a leitura correta é "a amostra não decide qual é melhor",
> não "são iguais".

---

## 3. Retrieval

De `docs/evaluation_results.json:retrieval`. A coluna "teto" (`min(alvos,k)/k`) é o que
impede ler `precision@8` baixo como desastre quando o teto também é baixo.

| k | precision | teto | recall | F1 |
| --- | --- | --- | --- | --- |
| 1 | — | — | — | — |
| 4 (produção) | — | — | — | — |
| 8 | — | — | — | — |

---

## 4. RAG × sem RAG

De `docs/evaluation_results.json:rag_vs_sem_rag`. As duas pontas julgadas contra o
**mesmo** contexto recuperado — julgar a ponta sem RAG contra contexto vazio daria "não
sustentada" por construção.

| Nível de sustentação | com RAG | sem RAG |
| --- | --- | --- |
| totalmente_sustentada | — | — |
| parcialmente_sustentada | — | — |
| não_sustentada | — | — |

---

## 5. Segurança — suíte adversarial contra o grafo compilado

De `docs/evaluation_results.json:seguranca`. As duas taxas só significam algo juntas.

| Taxa de bloqueio (casos que deveriam bloquear) | Taxa de falso positivo (controles legítimos) |
| --- | --- |
| — | — |

---

## 6. O que não funcionou

Preencher depois de olhar os resultados acima — nomear pelo menos um achado negativo
com número, não só "o sistema tem limitações".

---

## 7. Como reproduzir

Ver `README.md`. Nenhum número deste relatório foi digitado à mão — todos vêm dos
artefatos JSON listados no topo de cada seção.
