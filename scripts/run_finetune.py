"""`python -m scripts.run_finetune --etapa {train,merge}`

RODE NO SEU TERMINAL. Processo longo e pesado em GPU — disparar de dentro do
Claude Code derruba a sessao (antivirus mata processo longo). Precisa do
extra [train]: pip install -e ".[train]"
"""

from __future__ import annotations

import argparse
import json

from src import config, finetune


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--etapa", choices=["train", "merge"], required=True)
    args = parser.parse_args()

    if args.etapa == "train":
        resultado = finetune.treinar(config.DATASET_TREINO)
        print(f"Loss final: {resultado['loss_final']:.4f}")
        print(json.dumps(resultado["curva_loss"][-5:], indent=2))
        return

    finetune.mesclar_adapter_em_cpu(config.ADAPTER_DIR, config.MODELS_DIR / "merged")
    print("Modelo mesclado. Proximo passo (fora deste repo, via llama.cpp):")
    print(finetune.instrucoes_ollama())


if __name__ == "__main__":
    main()
