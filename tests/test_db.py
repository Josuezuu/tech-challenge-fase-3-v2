from __future__ import annotations

from pathlib import Path

import pytest

from src import canon, db


@pytest.fixture()
def conexao(tmp_path: Path):
    conn = db.conectar(tmp_path / "hospital.db")
    db.criar_schema(conn)
    db.seed(conn, n_pacientes=12, seed_valor=42)
    yield conn
    conn.close()


def test_seed_cria_o_numero_pedido_de_pacientes(conexao):
    assert conexao.execute("SELECT COUNT(*) FROM pacientes").fetchone()[0] == 12


def test_paciente_por_codigo_existente_e_inexistente(conexao):
    assert db.paciente_por_codigo(conexao, "PACIENTE_001") is not None
    assert db.paciente_por_codigo(conexao, "PACIENTE_999") is None


def test_exames_pendentes_so_traz_status_pendente(conexao):
    assert all(e.status == "pendente" for e in db.exames_pendentes(conexao, "PACIENTE_001"))


def test_exames_do_paciente_bate_com_exames_obrigatorios_da_condicao(conexao):
    paciente = db.paciente_por_codigo(conexao, "PACIENTE_001")
    condicao = canon.condicao_por_chave(paciente.condicao)
    exames = db.exames_do_paciente(conexao, "PACIENTE_001")
    assert {e.exame_codigo for e in exames} == set(condicao.exames_obrigatorios)


def test_contexto_paciente_cobre_existente_e_inexistente(conexao):
    assert "Nenhum registro" in db.contexto_paciente(conexao, "PACIENTE_999")
    contexto = db.contexto_paciente(conexao, "PACIENTE_001")
    pendentes = db.exames_pendentes(conexao, "PACIENTE_001")
    assert ("PENDENTES" in contexto) == bool(pendentes)


def test_registrar_alerta_persiste_no_banco(conexao):
    db.registrar_alerta(conexao, "PACIENTE_001", "alta", "exame critico pendente")
    assert conexao.execute("SELECT COUNT(*) FROM alertas").fetchone()[0] == 1


def test_seed_e_deterministico_pela_seed(tmp_path: Path):
    conn_a = db.conectar(tmp_path / "a.db")
    db.criar_schema(conn_a)
    db.seed(conn_a, n_pacientes=5, seed_valor=42)

    conn_b = db.conectar(tmp_path / "b.db")
    db.criar_schema(conn_b)
    db.seed(conn_b, n_pacientes=5, seed_valor=42)

    query = "SELECT idade, sexo FROM pacientes ORDER BY codigo"
    assert conn_a.execute(query).fetchall() == conn_b.execute(query).fetchall()
