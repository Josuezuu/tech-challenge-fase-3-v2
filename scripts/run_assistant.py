"""`LLM_PROVIDER=openai python -m scripts.run_assistant --pergunta "..." [--paciente PACIENTE_001]`

O assistente completo: sanitiza, classifica, recupera, gera, aplica
guardrails e audita — imprime a trilha (intencao, fontes, guardrails).
"""

from __future__ import annotations

import argparse

from src import config, db, graph, llm, rag


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pergunta", required=True)
    parser.add_argument("--paciente", default=None)
    args = parser.parse_args()

    indice = rag.IndiceFAISS.carregar(config.FAISS_INDEX_DIR)
    conexao = db.conectar(config.HOSPITAL_DB)
    provider = llm.obter_provider()

    grafo = graph.montar_grafo(indice, conexao, provider)
    estado_final = grafo.invoke({"pergunta": args.pergunta, "paciente_codigo": args.paciente})

    print("\n--- resposta ---")
    print(estado_final["resposta"])
    print("\n--- trilha ---")
    print(f"intencao: {estado_final.get('intencao')}")
    print(f"fontes: {estado_final.get('fontes')}")
    print(f"guardrails: {estado_final.get('guardrails_disparados')}")
    print(f"trace_id: {estado_final.get('trace_id')}")

    conexao.close()


if __name__ == "__main__":
    main()
