"""Guardrails de prescricao, sanitizacao de entrada e disclaimer de saida.

Tres regras atravessam o modulo inteiro:

1. substituir por marcador, nunca remover (apagar cola as pontas do texto e
   pode reconstruir o que se queria eliminar);
2. checar os dois lados — pergunta e resposta —, porque barrar so a pergunta
   deixa passar o caso em que o modelo oferece dose sem ninguem ter pedido;
3. nenhuma regex com quantificador aninhado (ReDoS).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

# --- textos --------------------------------------------------------------- #

DISCLAIMER_PREFIXO = "Esta resposta e gerada por um assistente de apoio"
DISCLAIMER = (
    f"{DISCLAIMER_PREFIXO} e nao substitui o julgamento clinico do "
    "profissional de saude responsavel pelo atendimento."
)

FALLBACK_CORPUS = (
    "Nao ha informacao suficiente nos protocolos internos para responder a essa "
    "pergunta. Tente reformular com mais detalhes clinicos, ou consulte "
    "diretamente o setor responsavel pelo protocolo relacionado."
)

RECUSA_ESCOPO = (
    "Esta pergunta esta fora do escopo deste assistente clinico interno. Ele "
    "responde sobre protocolos institucionais e sobre pacientes internados, "
    "identificados por codigo. Reformule dentro desse escopo, ou procure o "
    "canal administrativo apropriado."
)

RECUSA_PRESCRICAO = (
    "Este assistente nao prescreve nem informa dose exata de medicamento. A "
    "posologia e decisao do medico responsavel pelo atendimento, conforme a "
    "tabela de padronizacao da Farmacia Clinica Central. Consulte o "
    "profissional de saude antes de qualquer administracao."
)

ENCAMINHAMENTO_VALIDACAO = (
    "Esta sugestao de conduta precisa ser validada por um profissional de "
    "saude antes de qualquer acao — o assistente recomenda, o medico decide."
)

ESCALONAMENTO_HUMANO = (
    "Nao ha confianca suficiente nos protocolos consultados para responder "
    "com seguranca. Este caso esta sendo encaminhado para avaliacao de um "
    "profissional de saude."
)

# --- sanitizacao de entrada ------------------------------------------------ #

LIMITE_SANITIZACAO = 20_000
MARCADOR_SANITIZACAO = "[CONTEUDO_REMOVIDO]"

PADROES_INJECAO: tuple[re.Pattern[str], ...] = (
    re.compile(r"ignore\s+(?:all\s+)?previous\s+instructions?", re.IGNORECASE),
    re.compile(r"ignor[ae]\s+(?:todas\s+)?as?\s+instru[cç][õo]es?\s+anteriores", re.IGNORECASE),
    re.compile(r"disregard\s+(?:all\s+)?(?:previous|prior)\s+(?:instructions?|rules?)", re.IGNORECASE),
    re.compile(r"esque[cç]a\s+(?:tudo\s+)?(?:o\s+que\s+)?(?:foi\s+)?dito\s+antes", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\b", re.IGNORECASE),
    re.compile(r"voc[eê]\s+agora\s+[eé]\b", re.IGNORECASE),
    re.compile(r"\bjailbreak\b", re.IGNORECASE),
    re.compile(r"\bdan\s+mode\b", re.IGNORECASE),
    re.compile(r"\bmodo\s+desenvolvedor\b", re.IGNORECASE),
    re.compile(r"system\s*prompt", re.IGNORECASE),
    re.compile(r"revele\s+suas\s+instru[cç][õo]es", re.IGNORECASE),
)


@dataclass(frozen=True)
class ResultadoSanitizacao:
    texto: str
    padroes_encontrados: tuple[str, ...]
    truncado: bool


def _truncar(texto: str, limite: int) -> tuple[str, bool]:
    if len(texto) <= limite:
        return texto, False
    return texto[:limite], True


def sanitizar(texto: str, limite: int = LIMITE_SANITIZACAO) -> ResultadoSanitizacao:
    """Substitui cada padrao de injecao por um marcador — nunca remove."""
    texto_truncado, truncado_antes = _truncar(texto, limite)

    padroes_encontrados: list[str] = []
    resultado = texto_truncado
    for padrao in PADROES_INJECAO:
        if padrao.search(resultado):
            padroes_encontrados.append(padrao.pattern)
            resultado = padrao.sub(MARCADOR_SANITIZACAO, resultado)

    resultado_final, truncado_depois = _truncar(resultado, limite)
    return ResultadoSanitizacao(
        texto=resultado_final,
        padroes_encontrados=tuple(padroes_encontrados),
        truncado=truncado_antes or truncado_depois,
    )


# --- deteccao de prescricao ------------------------------------------------ #

_UNIDADES_DOSE = r"(?:mg|mcg|g|ml|UI|mEq|gotas?|comprimidos?|c[áa]psulas?)"
_PADRAO_DOSE = re.compile(rf"\b\d+(?:[.,]\d+)?\s*{_UNIDADES_DOSE}\b", re.IGNORECASE)
_PADRAO_VIA = re.compile(
    r"\bvia\s+(?:oral|intravenosa|endovenosa|subcut[âa]nea|intramuscular|"
    r"inalat[óo]ria|sublingual|retal)\b",
    re.IGNORECASE,
)
_PADRAO_FREQUENCIA = re.compile(
    r"\bde\s+\d+\s+em\s+\d+\s+horas\b|\ba\s+cada\s+\d+\s+horas\b", re.IGNORECASE
)
_PADRAO_CODIGO_FONTE = re.compile(r"\b[A-Z]+-[A-Z]+-\d+\b")

_FRASES_PEDIDO_PERGUNTA: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bme\s+passe\s+a\s+receita\b", re.IGNORECASE),
    re.compile(r"\bpassa(?:r)?\s+(?:uma\s+)?receita\b", re.IGNORECASE),
    re.compile(r"\bqual\s+a\s+dose\s+exata\b", re.IGNORECASE),
    re.compile(r"\bprescreva\b", re.IGNORECASE),
    re.compile(r"\bposologia\b", re.IGNORECASE),
    re.compile(r"\bquanto\s+devo\s+dar\s+de\b", re.IGNORECASE),
    re.compile(r"\bqual\s+dose\s+(?:eu\s+)?(?:devo|posso)\s+dar\b", re.IGNORECASE),
    re.compile(r"\bdos(?:e|agem)\b", re.IGNORECASE),
)

PERMITIDO_CITACAO_COM_FONTE = "citacao_de_protocolo_com_fonte"


@dataclass(frozen=True)
class DeteccaoPrescricao:
    disparou: bool
    lado: Literal["pergunta", "resposta"]
    evidencia: str
    permitido_por: str | None


def _detectar_na_pergunta(texto: str) -> DeteccaoPrescricao:
    for padrao in _FRASES_PEDIDO_PERGUNTA:
        encontrado = padrao.search(texto)
        if encontrado:
            return DeteccaoPrescricao(True, "pergunta", encontrado.group(0), None)

    dose = _PADRAO_DOSE.search(texto)
    if dose:
        return DeteccaoPrescricao(True, "pergunta", dose.group(0), None)

    return DeteccaoPrescricao(False, "pergunta", "", None)


def _detectar_na_resposta(texto: str, tem_fonte: bool | None) -> DeteccaoPrescricao:
    dose = _PADRAO_DOSE.search(texto)
    tem_contexto_posologico = dose and (_PADRAO_VIA.search(texto) or _PADRAO_FREQUENCIA.search(texto))

    if not tem_contexto_posologico:
        return DeteccaoPrescricao(False, "resposta", "", None)

    cita_fonte_no_texto = bool(_PADRAO_CODIGO_FONTE.search(texto))
    tem_encaminhamento = ENCAMINHAMENTO_VALIDACAO in texto

    if tem_fonte and cita_fonte_no_texto and tem_encaminhamento:
        return DeteccaoPrescricao(False, "resposta", dose.group(0), PERMITIDO_CITACAO_COM_FONTE)

    return DeteccaoPrescricao(True, "resposta", dose.group(0), None)


def detectar_prescricao(
    texto: str, lado: Literal["pergunta", "resposta"], tem_fonte: bool | None = None
) -> DeteccaoPrescricao:
    """Detecta pedido/emissao de prescricao, permitindo citacao legitima de protocolo.

    `lado="pergunta"` roda antes da classificacao; `lado="resposta"` roda
    depois da geracao, mesmo sem ninguem ter pedido.
    """
    if lado == "pergunta":
        return _detectar_na_pergunta(texto)
    return _detectar_na_resposta(texto, tem_fonte)


# --- disclaimer ------------------------------------------------------------ #

_FOLGA_PONTUACAO = 5


class RespostaVaziaError(ValueError):
    """Resposta vazia nao pode virar "so o disclaimer"."""


def tem_disclaimer(texto: str) -> bool:
    """True so quando o disclaimer fecha o texto (prefixo + posicao final)."""
    stripped = texto.rstrip()
    posicao = stripped.rfind(DISCLAIMER_PREFIXO)
    if posicao == -1:
        return False

    fim_do_prefixo = posicao + len(DISCLAIMER_PREFIXO)
    resto_esperado = len(DISCLAIMER) - len(DISCLAIMER_PREFIXO)
    resto_real = len(stripped) - fim_do_prefixo
    return abs(resto_real - resto_esperado) <= _FOLGA_PONTUACAO


def aplicar_disclaimer(texto: str) -> str:
    """Acrescenta o disclaimer no rodape — idempotente, nunca duplica."""
    if not texto.strip():
        raise RespostaVaziaError("resposta vazia nao pode receber disclaimer sozinho")
    if tem_disclaimer(texto):
        return texto
    return f"{texto.rstrip()}\n\n{DISCLAIMER}"
