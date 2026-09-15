from __future__ import annotations

import json
from pathlib import Path

from src import data


def _exemplo(instruction="I", input_="P", output="R", categoria="faq", origem="t:0"):
    return data.Exemplo(instruction=instruction, input=input_, output=output, categoria=categoria, origem=origem)


def test_dividir_estagio_nao_repete_exemplo_entre_treino_e_holdout():
    exemplos = [_exemplo(input_=f"pergunta {i}", origem=f"gerador:{i}") for i in range(40)]
    treino, holdout = data.dividir_estagio(exemplos, proporcao_holdout=0.2, seed=42)

    origens_treino = {e.origem for e in treino}
    origens_holdout = {e.origem for e in holdout}
    assert origens_treino.isdisjoint(origens_holdout)
    assert len(treino) + len(holdout) == len(exemplos)


def test_dividir_estagio_e_deterministico_pela_seed():
    exemplos = [_exemplo(input_=f"p{i}", origem=f"g:{i}") for i in range(30)]
    t1, _ = data.dividir_estagio(exemplos, 0.2, seed=7)
    t2, _ = data.dividir_estagio(exemplos, 0.2, seed=7)
    assert [e.origem for e in t1] == [e.origem for e in t2]


def test_preparar_dataset_segue_a_ordem_obrigatoria(monkeypatch):
    """anonimizar antes de curar, deduplicar antes de dividir — nao negociavel."""
    chamadas: list[str] = []

    def _rastrear(nome, original):
        def _wrapper(*args, **kwargs):
            chamadas.append(nome)
            return original(*args, **kwargs)

        return _wrapper

    for nome in ("normalizar_estagio", "anonimizar_estagio", "curar_estagio", "deduplicar_estagio", "truncar_estagio", "dividir_estagio"):
        monkeypatch.setattr(data, nome, _rastrear(nome.replace("_estagio", ""), getattr(data, nome)))

    data.preparar_dataset([_exemplo()], proporcao_holdout=0.2, seed=42)

    assert chamadas == ["normalizar", "anonimizar", "curar", "deduplicar", "truncar", "dividir"]


def test_deduplicar_remove_exemplos_identicos():
    exemplos = [_exemplo(origem="a"), _exemplo(origem="b"), _exemplo(input_="outra", origem="c")]
    assert len(data.deduplicar_estagio(exemplos)) == 2


def test_curar_protege_o_limite_de_prescricao_mas_mantem_documento_em_branco():
    documento = _exemplo(output="Posologia: [POSOLOGIA A DEFINIR PELO MEDICO ASSISTENTE]", categoria="documento")
    ensina_prescricao = _exemplo(output="Administrar 500 mg por via oral de 8 em 8 horas.", categoria="faq")
    faq_sem_dose = _exemplo(output="[Fonte: HSA-PROT-001] Exames obrigatorios: hemograma, lactato.", categoria="faq")

    resultado = data.curar_estagio([documento, ensina_prescricao, faq_sem_dose])
    assert resultado == [documento, faq_sem_dose]


def test_truncar_descarta_resposta_longa_em_vez_de_cortar():
    longa = _exemplo(output="x" * 2001)
    curta = _exemplo(output="x" * 100, origem="curta")
    assert data.truncar_estagio([longa, curta], limite_chars=2000) == [curta]


def test_anonimizar_substitui_cpf_telefone_e_nome_do_paciente():
    texto = "O paciente Joao Silva, CPF 123.456.789-00, tel (11) 99999-8888."
    anonimizado = data.anonimizar_texto(texto)
    assert "123.456.789-00" not in anonimizado and "Joao Silva" not in anonimizado
    assert "[CPF]" in anonimizado and "[NOME]" in anonimizado


def test_normalizar_colapsa_espacos_e_quebras_de_linha():
    resultado = data.normalizar("linha 1\n\n\n\nlinha  2   com espaco    demais  ")
    assert "\n\n\n" not in resultado and "  " not in resultado
    assert resultado == resultado.strip()


def test_recusas_e_fora_escopo_tem_dezenas_de_respostas_distintas():
    """Erro anterior: 32 exemplos de recusa, 1 unica resposta apos dedup. Nao pode repetir."""
    for gerador, categoria in ((data.gerar_recusas, "recusa"), (data.gerar_fora_escopo, "fora_escopo")):
        contagem = data.contar_categorias(gerador())[categoria]
        assert contagem["total"] >= 10
        assert contagem["respostas_distintas"] == contagem["total"]


def test_gerar_protocolos_md_inclui_front_matter_com_codigo(tmp_path: Path):
    documentos = data.gerar_protocolos_md()
    assert len(documentos) == len(data.canon.CONDICOES)
    assert next(iter(documentos.values())).startswith("---\ncodigo:")

    caminhos = data.escrever_protocolos(tmp_path)
    assert len(caminhos) == len(documentos) and all(c.exists() for c in caminhos)


def test_serializar_e_carregar_preserva_exemplos(tmp_path: Path):
    exemplos = [_exemplo(origem="a"), _exemplo(input_="outra", origem="b")]
    caminho = tmp_path / "dataset.jsonl"
    data.serializar(exemplos, caminho)
    assert data.carregar(caminho) == exemplos


def test_carregar_medquad_pula_pasta_ignorada_e_le_pasta_valida(tmp_path: Path):
    pasta_ignorada = tmp_path / "10_MPlus_ADAM_QA"
    pasta_ignorada.mkdir()
    (pasta_ignorada / "doc.xml").write_text(
        "<Document><QAPairs><QAPair><Question>P</Question><Answer></Answer></QAPair></QAPairs></Document>",
        encoding="utf-8",
    )
    pasta_valida = tmp_path / "01_valida"
    pasta_valida.mkdir()
    (pasta_valida / "doc.xml").write_text(
        "<Document><QAPairs><QAPair><Question>Qual a febre normal?</Question>"
        "<Answer>Ate 37.5 graus.</Answer></QAPair></QAPairs></Document>",
        encoding="utf-8",
    )

    exemplos = data.carregar_medquad(tmp_path)
    assert len(exemplos) == 1 and exemplos[0].input == "Qual a febre normal?"


def test_carregar_pubmedqa_descarta_resposta_longa_em_vez_de_truncar(tmp_path: Path):
    caminho = tmp_path / "pubmedqa.jsonl"
    with caminho.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"QUESTION": "curta?", "LONG_ANSWER": "ok"}) + "\n")
        f.write(json.dumps({"QUESTION": "longa?", "LONG_ANSWER": "x" * 2001}) + "\n")

    exemplos = data.carregar_pubmedqa(caminho)
    assert len(exemplos) == 1 and exemplos[0].input == "curta?"
