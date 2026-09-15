"""O UNICO lugar onde um prompt de treino ou de inferencia e montado.

Se voce veio do treino, da inferencia ou da avaliacao: importe daqui. Nao
copie o template, nao reescreva a f-string, nao "adapte" o formato em outro
modulo — toda ponta do comparativo tem que formatar a mesma entrada do mesmo
jeito, ou o comparativo passa a medir formatacao em vez de conteudo, e nada
avisa disso: o modelo continua respondendo, as metricas continuam saindo.

Invariante mecanica, nao convencao:

    formatar_treino(ex, eos).startswith(formatar_inferencia(ex.instruction, ex.input))

`formatar_treino` e, por construcao, `formatar_inferencia` mais a resposta e
o EOS. Nao ha como as duas divergirem sem o teste de prefixo cair.
"""

from __future__ import annotations

from src import config
from src.data import Exemplo

VERSAO_TEMPLATE: str = "1.0"

MARCADOR_INSTRUCAO = "### Instrucao:"
MARCADOR_ENTRADA = "### Entrada:"
MARCADOR_RESPOSTA = "### Resposta:"

MARCADORES: tuple[str, ...] = (MARCADOR_INSTRUCAO, MARCADOR_ENTRADA, MARCADOR_RESPOSTA)

TEMPLATE: str = (
    f"{MARCADOR_INSTRUCAO}\n"
    "{instruction}\n\n"
    f"{MARCADOR_ENTRADA}\n"
    "{input}\n\n"
    f"{MARCADOR_RESPOSTA}\n"
    "{output}"
)


def formatar_inferencia(instruction: str, input_: str) -> str:
    """Monta o prompt enviado ao modelo — termina em '### Resposta:\\n', sem EOS."""
    return TEMPLATE.format(instruction=instruction, input=input_, output="")


def formatar_treino(ex: Exemplo, eos: str | None = None) -> str:
    """Prompt de inferencia + resposta + EOS — comeca exatamente como a inferencia."""
    fim = config.EOS_TOKEN if eos is None else eos
    return formatar_inferencia(ex.instruction, ex.input) + ex.output + fim


def diferenca_marcada(esperado: str, recebido: str, contexto: int = 60) -> str:
    """Aponta o primeiro caractere onde dois prompts divergem — string vazia se iguais."""
    if esperado == recebido:
        return ""

    limite = min(len(esperado), len(recebido))
    ponto = next((i for i in range(limite) if esperado[i] != recebido[i]), limite)
    inicio = max(0, ponto - contexto)
    fim = ponto + contexto

    def _trecho(texto: str) -> str:
        return repr(texto[inicio:fim])[1:-1]

    def _no_ponto(texto: str) -> str:
        return repr(texto[ponto]) if ponto < len(texto) else "<fim da string>"

    return (
        f"prompts divergem no caractere {ponto} "
        f"(esperado: {len(esperado)} chars, recebido: {len(recebido)} chars)\n"
        f"  esperado[{inicio}:{fim}]: {_trecho(esperado)}\n"
        f"  recebido[{inicio}:{fim}]: {_trecho(recebido)}\n"
        f"  no caractere {ponto}: esperado {_no_ponto(esperado)}, recebido {_no_ponto(recebido)}"
    )
