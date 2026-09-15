"""`python -m scripts.run_prepare_data [--pubmedqa PATH] [--medquad PATH]`

Gera o corpus sintetico, ingere o externo se os caminhos forem passados,
roda o pipeline inteiro e escreve treino/holdout + o relatorio de preparo.
Tambem escreve os protocolos em Markdown (fonte do RAG) e semeia o banco.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src import config, data, db


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pubmedqa", type=Path, default=None, help="jsonl do PubMedQA PQA-L")
    parser.add_argument("--medquad", type=Path, default=None, help="pasta raiz do MedQuAD")
    parser.add_argument("--limite-externo", type=int, default=None)
    args = parser.parse_args()

    exemplos = data.gerar_corpus_sintetico()
    print(f"Sintetico: {len(exemplos)} exemplos.")

    if args.pubmedqa and args.pubmedqa.exists():
        externos = data.carregar_pubmedqa(args.pubmedqa, limite=args.limite_externo)
        print(f"PubMedQA: {len(externos)} exemplos.")
        exemplos += externos

    if args.medquad and args.medquad.exists():
        externos = data.carregar_medquad(args.medquad, limite=args.limite_externo)
        print(f"MedQuAD: {len(externos)} exemplos.")
        exemplos += externos

    treino, holdout, relatorio = data.preparar_dataset(exemplos, config.PROPORCAO_HOLDOUT, config.SEED)

    data.serializar(treino, config.DATASET_TREINO)
    data.serializar(holdout, config.DATASET_HOLDOUT)
    config.RELATORIO_PREPARO.write_text(json.dumps(relatorio.como_dict(), indent=2), encoding="utf-8")

    print(f"Treino: {len(treino)} | Holdout: {len(holdout)}")
    print("Contagem por categoria (treino):", data.contar_categorias(treino))

    pasta_protocolos = config.DATA_RAW / "protocolos"
    caminhos = data.escrever_protocolos(pasta_protocolos)
    print(f"Protocolos escritos: {len(caminhos)} em {pasta_protocolos}")

    conexao = db.conectar(config.HOSPITAL_DB)
    db.criar_schema(conexao)
    db.seed(conexao, seed_valor=config.SEED)
    conexao.close()
    print(f"Banco semeado em {config.HOSPITAL_DB}")


if __name__ == "__main__":
    main()
