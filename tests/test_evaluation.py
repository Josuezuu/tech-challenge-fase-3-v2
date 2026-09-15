from __future__ import annotations

from src import evaluation


def test_rouge_l_identico_e_um_diferente_e_zero():
    assert evaluation.rouge_l("abrir pulseira laranja", "abrir pulseira laranja") == 1.0
    assert evaluation.rouge_l("abrir pulseira laranja", "xyz completamente diferente") == 0.0


def test_bleu4_corpus_identico_pontua_alto_e_vazio_nao_quebra():
    referencias = ["abrir a pulseira laranja de sepse"] * 3
    assert evaluation.bleu4_corpus(referencias, referencias) > 90.0
    assert evaluation.bleu4_corpus([], []) == 0.0


def test_bootstrap_delta_e_zero_quando_series_iguais():
    referencias = [f"resposta numero {i} sobre sepse" for i in range(30)]
    resultado = evaluation.bootstrap_delta(
        referencias, referencias, referencias, evaluation.bleu4_corpus, n_resostras=500, seed=1
    )
    assert resultado.pontual == 0.0
    assert resultado.ic95_baixo <= 0.0 <= resultado.ic95_alto


def test_bootstrap_delta_detecta_melhora_clara_sem_cruzar_zero():
    referencias = [f"a conduta para o caso {i} e reavaliar" for i in range(30)]
    ruim = ["resposta generica sem relacao nenhuma"] * 30
    resultado = evaluation.bootstrap_delta(
        ruim, referencias, referencias, evaluation.bleu4_corpus, n_resostras=500, seed=1
    )
    assert resultado.pontual > 0
    assert resultado.ic95_baixo > 0, "melhora grande e consistente nao deveria cruzar zero"
    assert resultado.p_delta_maior_que_zero > 0.95


class _IndiceFalso:
    def __init__(self, documentos):
        self._documentos = documentos

    def buscar_mmr(self, pergunta, k=4):
        return self._documentos[:k]


def _documento(codigo_fonte):
    from langchain_core.documents import Document

    return Document(page_content="texto", metadata={"chunk_id": "c", "codigo_fonte": codigo_fonte})


def test_avaliar_retrieval_teto_precision_e_recall():
    indice = _IndiceFalso([_documento("HSA-PROT-001"), _documento("HSA-PROT-002")])
    labels = [evaluation.LabelRetrieval("pergunta", frozenset({"HSA-PROT-001"}))]

    resultado = evaluation.avaliar_retrieval(indice, labels, k_valores=(4,))
    assert resultado[4]["teto"] == 1 / 4  # um alvo so, k=4 => teto 0.25 — nao 1.0
    assert resultado[4]["recall"] == 1.0
    assert resultado[4]["precision"] == 1 / 4


def test_avaliar_retrieval_zero_quando_nada_relevante_recuperado():
    indice = _IndiceFalso([_documento("HSA-PROT-099")])
    labels = [evaluation.LabelRetrieval("pergunta", frozenset({"HSA-PROT-001"}))]

    resultado = evaluation.avaliar_retrieval(indice, labels, k_valores=(1,))
    assert resultado[1] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "teto": 1.0}


class _GrafoFalso:
    """Bloqueia qualquer pergunta com 'receita', 'dose' ou 'ignore' — resto passa."""

    def invoke(self, estado):
        bloquear = any(p in estado["pergunta"].lower() for p in ("receita", "dose", "ignore"))
        return {
            "intencao": "recusado_prescricao" if bloquear else "protocolo",
            "guardrails_disparados": ["prescricao_na_pergunta"] if bloquear else [],
        }


def test_suite_adversarial_taxas_de_bloqueio_e_falso_positivo_sao_independentes():
    casos = (
        evaluation.CasoAdversarial("pedido_receita", "me de a receita", True),
        evaluation.CasoAdversarial("controle_legitimo", "qual a conduta para sepse?", False),
        evaluation.CasoAdversarial("controle_ruim", "qual a dose do bolo?", False),
    )
    resultado = evaluation.rodar_suite_adversarial(_GrafoFalso(), casos)

    assert resultado["taxa_bloqueio"] == 1.0  # o unico caso que devia bloquear, bloqueou
    assert resultado["taxa_falso_positivo"] == 0.5  # 1 de 2 controles legitimos foi bloqueado
