"""`python -m scripts.run_evaluation` — RODE NO SEU TERMINAL (carrega modelos locais).

Roda os quatro blocos de avaliacao e grava tudo em docs/evaluation_results.json:
geracao (baseline x fine-tunado x checkpoint, IC95% por bootstrap), retrieval,
RAG x sem-RAG, suite adversarial. Nenhum numero de tabela do relatorio deve
ser digitado a mao — todos saem deste JSON.
"""

from __future__ import annotations

import argparse

from src import config, db, evaluation, graph, llm, rag
from src.data import carregar


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-geracao", type=int, default=50)
    parser.add_argument("--checkpoint-intermediario", default=None, help="pasta do checkpoint a comparar")
    args = parser.parse_args()

    resultado: dict = {}

    # --- 1. geracao: baseline x fine-tunado x checkpoint intermediario ------ #
    holdout = carregar(config.DATASET_HOLDOUT)
    providers = {"base": llm.obter_provider("base"), "finetuned": llm.obter_provider("finetuned")}
    if args.checkpoint_intermediario:
        providers["checkpoint_intermediario"] = llm.ProviderCheckpointLocal(
            "checkpoint_intermediario", args.checkpoint_intermediario
        )
    resultado["geracao"] = evaluation.avaliar_geracao(providers, holdout, n=args.n_geracao)

    # --- 2. retrieval -------------------------------------------------------- #
    indice = rag.IndiceFAISS.carregar(config.FAISS_INDEX_DIR)
    labels = [
        evaluation.LabelRetrieval(pergunta=e.input, fontes_relevantes=frozenset({e.output.split("]")[0].lstrip("[").split(":")[-1].strip()}))
        for e in holdout
        if e.output.startswith("[Fonte:") or e.output.startswith("[HSA-")
    ][:30]
    if labels:
        resultado["retrieval"] = evaluation.avaliar_retrieval(indice, labels)

    # --- 3. RAG x sem RAG ----------------------------------------------------- #
    provider_producao = llm.obter_provider("openai")
    perguntas_rag = [e.input for e in holdout if e.categoria == "faq"][:10]
    if perguntas_rag:
        resultado["rag_vs_sem_rag"] = evaluation.avaliar_rag_vs_sem_rag(
            provider_producao, provider_producao, indice, perguntas_rag
        )

    # --- 4. suite adversarial contra o grafo compilado ------------------------ #
    conexao = db.conectar(config.HOSPITAL_DB)
    grafo = graph.montar_grafo(indice, conexao, provider_producao)
    resultado["seguranca"] = evaluation.rodar_suite_adversarial(grafo)
    conexao.close()

    evaluation.salvar_avaliacao(resultado)
    print(f"Avaliacao completa gravada em {config.AVALIACAO_JSON}")


if __name__ == "__main__":
    main()
