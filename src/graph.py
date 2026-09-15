"""O grafo LangGraph: sanitiza, classifica, recupera, gera, guarda, audita.

Bem menor que um grafo de producao real de proposito — o alvo deste projeto e
um decimo do tamanho do anterior. Nove nos, quatro caminhos de intencao, uma
faixa de seguranca por onde todo caminho passa antes do fim:

    sanitizar -> guardrail_pergunta -+-> (disparou) -> recusar ----------+
                                     +-> classificar_intencao            |
                                          +- protocolo      -> RAG       |
                                          +- triagem_paciente -> RAG+db  |
                                          +- documento       -> LLM      |
                                          +- fora_escopo     -> recusar -+
                                                                          v
                                          guardrail_resposta -> registrar_auditoria -> fim

A classificacao de intencao e uma regra Python, nao julgamento de LLM:
deterministica, testavel, sem depender de um Phi-3 quantizado obedecer
schema — a mesma razao pela qual `src/llm.py` nao usa o fine-tunado como
orquestrador.
"""

from __future__ import annotations

import time
from typing import TypedDict

from langgraph.graph import END, StateGraph

from src import audit, canon, db, rag, safety


class EstadoAssistente(TypedDict, total=False):
    pergunta: str
    paciente_codigo: str | None
    pergunta_sanitizada: str
    intencao: str
    contexto_paciente: str
    resposta: str
    fontes: list[str]
    guardrails_disparados: list[str]
    requer_validacao_humana: bool
    trace_id: str


_PALAVRAS_DOCUMENTO = ("laudo", "receita", "modelo de", "procedimento")


def classificar_intencao(pergunta: str, paciente_codigo: str | None) -> str:
    """Regra deterministica: sinonimo de condicao => protocolo/triagem; senao documento/fora_escopo."""
    texto = pergunta.lower()

    if any(p in texto for p in _PALAVRAS_DOCUMENTO):
        return "documento"

    cita_condicao = any(
        sinonimo.lower() in texto
        for condicao in canon.CONDICOES
        for sinonimo in condicao.sinonimos
    )
    if not cita_condicao:
        return "fora_escopo"

    return "triagem_paciente" if paciente_codigo else "protocolo"


# --------------------------------------------------------------------------- #
# Nos
# --------------------------------------------------------------------------- #


def no_sanitizar_entrada(estado: EstadoAssistente) -> EstadoAssistente:
    resultado = safety.sanitizar(estado["pergunta"])
    disparados = list(estado.get("guardrails_disparados", []))
    if resultado.padroes_encontrados:
        disparados.append("sanitizacao_entrada")
    return {**estado, "pergunta_sanitizada": resultado.texto, "guardrails_disparados": disparados}


def no_guardrail_pergunta(estado: EstadoAssistente) -> EstadoAssistente:
    deteccao = safety.detectar_prescricao(estado["pergunta_sanitizada"], "pergunta")
    if not deteccao.disparou:
        return estado
    disparados = [*estado.get("guardrails_disparados", []), "prescricao_na_pergunta"]
    return {
        **estado,
        "resposta": safety.RECUSA_PRESCRICAO,
        "fontes": [],
        "guardrails_disparados": disparados,
        "intencao": "recusado_prescricao",
    }


def no_classificar_intencao(estado: EstadoAssistente) -> EstadoAssistente:
    intencao = classificar_intencao(estado["pergunta_sanitizada"], estado.get("paciente_codigo"))
    return {**estado, "intencao": intencao}


def no_recuperar_protocolo(estado: EstadoAssistente, indice: rag.IndiceFAISS, provider) -> EstadoAssistente:
    documentos = indice.buscar_mmr(estado["pergunta_sanitizada"])
    resultado = rag.filtrar_e_ranquear(estado["pergunta_sanitizada"], documentos)

    if resultado.confianca < rag.LIMIAR_CONFIANCA:
        return {
            **estado,
            "resposta": safety.ESCALONAMENTO_HUMANO,
            "fontes": [],
            "requer_validacao_humana": True,
        }

    resposta = rag.gerar_resposta_com_fonte(provider, estado["pergunta_sanitizada"], resultado)
    fontes = [d.metadata["chunk_id"] for d in resultado.documentos]
    return {**estado, "resposta": resposta, "fontes": fontes, "requer_validacao_humana": True}


def no_carregar_contexto_paciente(estado: EstadoAssistente, conexao) -> EstadoAssistente:
    contexto = db.contexto_paciente(conexao, estado["paciente_codigo"])
    return {**estado, "contexto_paciente": contexto}


def no_gerar_documento(estado: EstadoAssistente, provider) -> EstadoAssistente:
    prompt = (
        "Gere o documento pedido no formato institucional do HSA, sem preencher "
        f"dose ou posologia real:\n{estado['pergunta_sanitizada']}"
    )
    resposta = provider.gerar(prompt)
    return {**estado, "resposta": resposta, "fontes": [], "requer_validacao_humana": True}


def no_recusar(estado: EstadoAssistente) -> EstadoAssistente:
    """Recusa por fora_escopo. Se a recusa ja veio do guardrail de pergunta
    (prescricao), preserva a mensagem especifica em vez de sobrescrever."""
    if estado.get("intencao") == "recusado_prescricao":
        return {**estado, "fontes": [], "requer_validacao_humana": False}
    return {**estado, "resposta": safety.RECUSA_ESCOPO, "fontes": [], "requer_validacao_humana": False}


def no_guardrail_resposta(estado: EstadoAssistente) -> EstadoAssistente:
    resposta = estado["resposta"]
    disparados = list(estado.get("guardrails_disparados", []))

    deteccao = safety.detectar_prescricao(resposta, "resposta", tem_fonte=bool(estado.get("fontes")))
    if deteccao.disparou:
        resposta = safety.RECUSA_PRESCRICAO
        disparados.append("prescricao_na_resposta")

    if estado.get("requer_validacao_humana") and safety.ENCAMINHAMENTO_VALIDACAO not in resposta:
        resposta = f"{resposta}\n\n{safety.ENCAMINHAMENTO_VALIDACAO}"

    resposta = safety.aplicar_disclaimer(resposta)
    return {**estado, "resposta": resposta, "guardrails_disparados": disparados}


def no_registrar_auditoria(estado: EstadoAssistente, provider_nome: str, inicio: float) -> EstadoAssistente:
    registro = audit.RegistroAuditoria(
        trace_id=estado.get("trace_id") or audit.novo_trace_id(),
        pergunta=estado["pergunta_sanitizada"],
        intencao=estado.get("intencao", "desconhecida"),
        fontes=tuple(estado.get("fontes", [])),
        guardrails_disparados=tuple(estado.get("guardrails_disparados", [])),
        llm_provider=provider_nome,
        latencia_s=round(time.monotonic() - inicio, 3),
    )
    audit.registrar(registro)
    return {**estado, "trace_id": registro.trace_id}


# --------------------------------------------------------------------------- #
# Roteadores
# --------------------------------------------------------------------------- #


def rotear_pos_guardrail_pergunta(estado: EstadoAssistente) -> str:
    return "recusar" if estado.get("intencao") == "recusado_prescricao" else "classificar"


def rotear_por_intencao(estado: EstadoAssistente) -> str:
    return estado["intencao"]


# --------------------------------------------------------------------------- #
# Montagem
# --------------------------------------------------------------------------- #


def montar_grafo(indice: rag.IndiceFAISS, conexao_db, provider):
    inicio = time.monotonic()
    grafo = StateGraph(EstadoAssistente)

    grafo.add_node("sanitizar", no_sanitizar_entrada)
    grafo.add_node("guardrail_pergunta", no_guardrail_pergunta)
    grafo.add_node("classificar", no_classificar_intencao)
    grafo.add_node("recuperar_protocolo", lambda e: no_recuperar_protocolo(e, indice, provider))
    grafo.add_node("carregar_paciente", lambda e: no_carregar_contexto_paciente(e, conexao_db))
    grafo.add_node("gerar_documento", lambda e: no_gerar_documento(e, provider))
    grafo.add_node("recusar", no_recusar)
    grafo.add_node("guardrail_resposta", no_guardrail_resposta)
    grafo.add_node("registrar_auditoria", lambda e: no_registrar_auditoria(e, provider.nome, inicio))

    grafo.set_entry_point("sanitizar")
    grafo.add_edge("sanitizar", "guardrail_pergunta")
    grafo.add_conditional_edges(
        "guardrail_pergunta",
        rotear_pos_guardrail_pergunta,
        {"recusar": "recusar", "classificar": "classificar"},
    )
    grafo.add_conditional_edges(
        "classificar",
        rotear_por_intencao,
        {
            "protocolo": "recuperar_protocolo",
            "triagem_paciente": "carregar_paciente",
            "documento": "gerar_documento",
            "fora_escopo": "recusar",
        },
    )
    grafo.add_edge("carregar_paciente", "recuperar_protocolo")
    grafo.add_edge("recuperar_protocolo", "guardrail_resposta")
    grafo.add_edge("gerar_documento", "guardrail_resposta")
    grafo.add_edge("recusar", "guardrail_resposta")
    grafo.add_edge("guardrail_resposta", "registrar_auditoria")
    grafo.add_edge("registrar_auditoria", END)

    return grafo.compile()
