"""Prontuario sintetico do HSA — schema, seed e consultas parametrizadas.

Unica porta de acesso ao banco: nenhum SQL e montado por LLM em lugar nenhum
do projeto. `paciente_codigo` sempre entra como parametro de query, nunca
interpolado na string — deixar um modelo escrever SQL contra prontuario e uma
superficie de ataque que nao precisa existir.
"""

from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from src import canon

SCHEMA = """
CREATE TABLE IF NOT EXISTS pacientes (
    codigo TEXT PRIMARY KEY,
    condicao TEXT NOT NULL,
    setor_sigla TEXT NOT NULL,
    idade INTEGER NOT NULL,
    sexo TEXT NOT NULL CHECK (sexo IN ('F', 'M', 'O'))
);

CREATE TABLE IF NOT EXISTS exames_paciente (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paciente_codigo TEXT NOT NULL REFERENCES pacientes(codigo),
    exame_codigo TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pendente', 'coletado', 'liberado')),
    valor REAL,
    data_solicitacao TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alertas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paciente_codigo TEXT NOT NULL REFERENCES pacientes(codigo),
    severidade TEXT NOT NULL,
    mensagem TEXT NOT NULL,
    criado_em TEXT NOT NULL
);
"""


def conectar(caminho: Path) -> sqlite3.Connection:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(caminho)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def criar_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def seed(conn: sqlite3.Connection, n_pacientes: int = 20, seed_valor: int = 42) -> None:
    """Popula com pacientes sinteticos, um por condicao do canone, em rodizio."""
    rng = random.Random(seed_valor)
    conn.execute("DELETE FROM alertas")
    conn.execute("DELETE FROM exames_paciente")
    conn.execute("DELETE FROM pacientes")

    for i in range(1, n_pacientes + 1):
        condicao = canon.CONDICOES[i % len(canon.CONDICOES)]
        codigo = canon.codigo_paciente(i)
        idade = rng.randint(19, 88)
        sexo = rng.choice(("F", "M", "O"))
        conn.execute(
            "INSERT INTO pacientes (codigo, condicao, setor_sigla, idade, sexo) "
            "VALUES (?, ?, ?, ?, ?)",
            (codigo, condicao.chave, condicao.setor_sigla, idade, sexo),
        )
        for exame_codigo in condicao.exames_obrigatorios:
            status = "pendente" if rng.random() < 0.3 else rng.choice(("coletado", "liberado"))
            valor = round(rng.uniform(0.1, 50.0), 1) if status == "liberado" else None
            conn.execute(
                "INSERT INTO exames_paciente "
                "(paciente_codigo, exame_codigo, status, valor, data_solicitacao) "
                "VALUES (?, ?, ?, ?, date('now'))",
                (codigo, exame_codigo, status, valor),
            )
    conn.commit()


@dataclass(frozen=True)
class Paciente:
    codigo: str
    condicao: str
    setor_sigla: str
    idade: int
    sexo: str


@dataclass(frozen=True)
class ExamePaciente:
    exame_codigo: str
    status: str
    valor: float | None
    data_solicitacao: str


def paciente_por_codigo(conn: sqlite3.Connection, codigo: str) -> Paciente | None:
    linha = conn.execute(
        "SELECT codigo, condicao, setor_sigla, idade, sexo FROM pacientes WHERE codigo = ?",
        (codigo,),
    ).fetchone()
    return Paciente(*linha) if linha else None


def exames_do_paciente(conn: sqlite3.Connection, codigo: str) -> list[ExamePaciente]:
    linhas = conn.execute(
        "SELECT exame_codigo, status, valor, data_solicitacao "
        "FROM exames_paciente WHERE paciente_codigo = ?",
        (codigo,),
    ).fetchall()
    return [ExamePaciente(*linha) for linha in linhas]


def exames_pendentes(conn: sqlite3.Connection, codigo: str) -> list[ExamePaciente]:
    """Regra deterministica: pendencia e o status, nao julgamento de LLM."""
    return [e for e in exames_do_paciente(conn, codigo) if e.status == "pendente"]


def registrar_alerta(
    conn: sqlite3.Connection, codigo: str, severidade: str, mensagem: str
) -> None:
    conn.execute(
        "INSERT INTO alertas (paciente_codigo, severidade, mensagem, criado_em) "
        "VALUES (?, ?, ?, datetime('now'))",
        (codigo, severidade, mensagem),
    )
    conn.commit()


def contexto_paciente(conn: sqlite3.Connection, codigo: str) -> str:
    """Formata prontuario + exames para entrar no prompt como dado delimitado."""
    paciente = paciente_por_codigo(conn, codigo)
    if paciente is None:
        return f"Nenhum registro encontrado para o paciente {codigo}."

    condicao = canon.condicao_por_chave(paciente.condicao)
    exames = exames_do_paciente(conn, codigo)
    pendentes = [e for e in exames if e.status == "pendente"]

    linhas = [
        f"Paciente {paciente.codigo}, {paciente.idade} anos, sexo {paciente.sexo}.",
        f"Condicao registrada: {condicao.nome} (setor {paciente.setor_sigla}).",
    ]
    if pendentes:
        nomes = ", ".join(canon.exame_por_codigo(e.exame_codigo).nome for e in pendentes)
        linhas.append(f"Exames PENDENTES (aguardando coleta ou resultado): {nomes}.")
    else:
        linhas.append("Nenhum exame pendente no momento.")
    return "\n".join(linhas)
