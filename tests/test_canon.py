from __future__ import annotations

import pytest

from src import canon


def test_toda_condicao_cita_exame_que_existe_no_catalogo():
    """Consistencia referencial: protocolo nao pode citar exame que o banco nao tem."""
    for condicao in canon.CONDICOES:
        for codigo_exame in condicao.exames_obrigatorios:
            assert codigo_exame in canon.CODIGOS_EXAMES, f"{condicao.chave} cita {codigo_exame} inexistente"


def test_toda_condicao_tem_setor_que_existe():
    setores_validos = {s.sigla for s in canon.SETORES}
    for condicao in canon.CONDICOES:
        assert condicao.setor_sigla in setores_validos


def test_codigo_protocolo_e_codigo_paciente_validam_entrada():
    with pytest.raises(ValueError):
        canon.codigo_protocolo(0)
    assert canon.codigo_paciente(3) == "PACIENTE_003"


def test_buscas_por_chave_inexistente_levantam_keyerror():
    with pytest.raises(KeyError):
        canon.condicao_por_chave("nao_existe")
    with pytest.raises(KeyError):
        canon.exame_por_codigo("NAO_EXISTE")


def test_exame_critico_se_respeita_sentido_e_ignora_valor_ausente():
    exame = canon.exame_por_codigo("LACTATO_ARTERIAL")  # critico "acima" de 4.0
    assert exame.critico_se(5.0) is True
    assert exame.critico_se(1.0) is False
    assert exame.critico_se(None) is False


def test_exame_descritivo_nunca_e_critico_por_limiar():
    exame = canon.exame_por_codigo("ECG_12_DERIVACOES")  # sentido "nenhum"
    assert exame.critico_se(1.0) is False
