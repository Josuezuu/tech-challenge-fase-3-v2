"""`python -m scripts.run_ingest` — constroi o indice FAISS a partir dos protocolos.

Passo demorado (baixa o modelo de embedding na primeira vez). Corpus e dataset
ja precisam existir — rode `run_prepare_data` antes.
"""

from __future__ import annotations

from src import config, rag


def main() -> None:
    pasta_protocolos = config.DATA_RAW / "protocolos"
    documentos = rag.carregar_protocolos(pasta_protocolos)
    if not documentos:
        raise SystemExit(f"Nenhum protocolo em {pasta_protocolos}. Rode run_prepare_data primeiro.")

    chunks = rag.dividir_em_chunks(documentos)
    print(f"{len(documentos)} protocolos -> {len(chunks)} chunks.")

    indice = rag.IndiceFAISS.construir(chunks)
    indice.salvar(config.FAISS_INDEX_DIR)
    print(f"Indice salvo em {config.FAISS_INDEX_DIR}")


if __name__ == "__main__":
    main()
