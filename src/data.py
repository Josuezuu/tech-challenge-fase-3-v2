"""Corpus sintetico, ingestao externa e o pipeline que vira dataset de treino.

Duas trilhas usam o mesmo canone para coisas diferentes: os protocolos em
Markdown alimentam o RAG (a fonte que sera citada); FAQ, documentos-modelo,
recusa e fora_escopo alimentam o fine-tuning (a forma de escrever do
hospital). Elas nao se encontram — fatos vem do RAG, forma vem do fine-tuning.

Ordem obrigatoria do preparo, e ela e testada:
    normalizar -> anonimizar -> curar -> deduplicar -> truncar -> dividir

Anonimizar antes de curar: curar primeiro pode descartar um exemplo por
conter um nome, quando o certo era anonimiza-lo e manter. Deduplicar antes de
dividir: dividir primeiro deixa o mesmo exemplo cair em treino e holdout, e a
metrica infla inteira (leakage).
"""

from __future__ import annotations

import json
import random
import re
import unicodedata
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from src import canon, safety
from src.canon import Condicao

# --------------------------------------------------------------------------- #
# O tipo que atravessa o pipeline inteiro — imutavel, com origem rastreavel
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Exemplo:
    """Um par de treino. `origem` prova de onde veio (arquivo:linha ou gerador)."""

    instruction: str
    input: str
    output: str
    categoria: str
    origem: str

    def com(self, **kwargs: str) -> "Exemplo":
        return replace(self, **kwargs)


@dataclass
class RelatorioPreparo:
    entradas_por_estagio: dict[str, int]

    def registrar(self, estagio: str, n: int) -> None:
        self.entradas_por_estagio[estagio] = n

    def como_dict(self) -> dict[str, int]:
        return dict(self.entradas_por_estagio)


INSTRUCAO_PADRAO = (
    f"Voce e o assistente clinico do {canon.HOSPITAL}. Responda com base nos "
    "protocolos internos, cite a fonte e nunca prescreva sem validacao humana."
)

# --------------------------------------------------------------------------- #
# Geradores — trilha RAG: protocolos em Markdown, com front-matter
# --------------------------------------------------------------------------- #


def gerar_protocolos_md(condicoes: tuple[Condicao, ...] = canon.CONDICOES) -> dict[str, str]:
    """Um .md por condicao, front-matter + secoes que o RAG vai indexar."""
    documentos: dict[str, str] = {}
    for i, condicao in enumerate(condicoes, start=1):
        codigo = canon.codigo_protocolo(i)
        setor = canon.setor_por_sigla(condicao.setor_sigla)
        exames = ", ".join(
            canon.exame_por_codigo(c).nome for c in condicao.exames_obrigatorios
        )
        linhas = [
            "---",
            f"codigo: {codigo}",
            f"condicao: {condicao.chave}",
            f"setor: {condicao.setor_sigla}",
            f"gravidade: {condicao.gravidade}",
            "---",
            f"# {codigo} — {condicao.nome}",
            "",
            f"Setor responsavel: {setor.nome} (ramal {setor.ramal}).",
            "",
            "## Criterios de inclusao",
            *[f"- {c}" for c in condicao.criterios_inclusao],
            "",
            "## Exames obrigatorios",
            f"- {exames}",
            "",
            "## Conduta",
            *[f"{n}. {passo}" for n, passo in enumerate(condicao.conduta, start=1)],
            "",
            "## Contraindicacoes",
            *[f"- {c}" for c in condicao.contraindicacoes],
            "",
            f"Reavaliar a cada {condicao.janela_reavaliacao_h}h.",
        ]
        documentos[f"{codigo}.md"] = "\n".join(linhas) + "\n"
    return documentos


def escrever_protocolos(pasta: Path, condicoes: tuple[Condicao, ...] = canon.CONDICOES) -> list[Path]:
    pasta.mkdir(parents=True, exist_ok=True)
    caminhos = []
    for nome, conteudo in gerar_protocolos_md(condicoes).items():
        caminho = pasta / nome
        caminho.write_text(conteudo, encoding="utf-8")
        caminhos.append(caminho)
    return caminhos


# --------------------------------------------------------------------------- #
# Geradores — trilha fine-tuning: FAQ ancorada nos protocolos
# --------------------------------------------------------------------------- #

_PERGUNTAS_FAQ = (
    "Qual a conduta inicial para {sinonimo}?",
    "Quais exames pedir diante de {sinonimo}?",
    "Qual o protocolo do HSA para {sinonimo}?",
    "Como reavaliar um caso de {sinonimo}?",
)


def gerar_faq(condicoes: tuple[Condicao, ...] = canon.CONDICOES) -> list[Exemplo]:
    """FAQ ancorada em cada condicao — a pergunta varia pelo sinonimo real."""
    exemplos: list[Exemplo] = []
    for i, condicao in enumerate(condicoes, start=1):
        codigo = canon.codigo_protocolo(i)
        exames = ", ".join(canon.exame_por_codigo(c).nome for c in condicao.exames_obrigatorios)
        for j, template in enumerate(_PERGUNTAS_FAQ):
            sinonimo = condicao.sinonimos[j % len(condicao.sinonimos)]
            pergunta = template.format(sinonimo=sinonimo)
            if "exames" in template:
                resposta = f"[Fonte: {codigo}] Exames obrigatorios: {exames}."
            elif "reavaliar" in template.lower():
                resposta = (
                    f"[Fonte: {codigo}] Reavaliar a cada {condicao.janela_reavaliacao_h}h, "
                    f"conforme o protocolo do setor {condicao.setor_sigla}."
                )
            else:
                primeiro_passo = condicao.conduta[0]
                resposta = f"[Fonte: {codigo}] {primeiro_passo}"
            exemplos.append(
                Exemplo(
                    instruction=INSTRUCAO_PADRAO,
                    input=pergunta,
                    output=resposta,
                    categoria="faq",
                    origem=f"gerador_faq:{condicao.chave}:{j}",
                )
            )
    return exemplos


# --------------------------------------------------------------------------- #
# Geradores — documentos-modelo (formato, nunca preenchido com dose real)
# --------------------------------------------------------------------------- #


def gerar_documentos_modelo(condicoes: tuple[Condicao, ...] = canon.CONDICOES) -> list[Exemplo]:
    """Modelos de laudo/receita/procedimento EM BRANCO — ensina formato, nao dose.

    A receita traz `[POSOLOGIA A DEFINIR PELO MEDICO ASSISTENTE]` no lugar de
    numero: o enunciado pede que o modelo aprenda o formato do documento, nao
    que ele aprenda a preencher dose — isso e o que a curadoria vai proteger.
    """
    exemplos: list[Exemplo] = []
    for i, condicao in enumerate(condicoes, start=1):
        codigo = canon.codigo_protocolo(i)

        laudo = (
            f"MODELO DE LAUDO — {condicao.nome}\n"
            f"Setor: {condicao.setor_sigla}\n"
            "Achados: [A PREENCHER PELO PROFISSIONAL]\n"
            f"Conduta sugerida: ver {codigo}.\n"
            f"{safety.ENCAMINHAMENTO_VALIDACAO}"
        )
        receita = (
            f"MODELO DE RECEITA — {condicao.nome}\n"
            "Medicamento: [A DEFINIR PELO MEDICO ASSISTENTE]\n"
            "Posologia: [POSOLOGIA A DEFINIR PELO MEDICO ASSISTENTE]\n"
            f"Fonte do protocolo institucional: {codigo}.\n"
            f"{safety.ENCAMINHAMENTO_VALIDACAO}"
        )
        procedimento = (
            f"MODELO DE PROCEDIMENTO — {condicao.nome}\n"
            f"Passos: {'; '.join(condicao.conduta)}\n"
            f"[Fonte: {codigo}]"
        )

        for nome_doc, texto in (("laudo", laudo), ("receita", receita), ("procedimento", procedimento)):
            exemplos.append(
                Exemplo(
                    instruction=INSTRUCAO_PADRAO,
                    input=f"Gere o modelo de {nome_doc} para {condicao.nome.lower()}.",
                    output=texto,
                    categoria="documento",
                    origem=f"gerador_documentos:{condicao.chave}:{nome_doc}",
                )
            )
    return exemplos


# --------------------------------------------------------------------------- #
# Geradores — recusa e fora_escopo, com dezenas de formulacoes distintas
# --------------------------------------------------------------------------- #
#
# Erro do projeto anterior: 32 exemplos de recusa com a MESMA resposta;
# a dedup sobrou 13, 2% do dataset, e o modelo decorou uma frase em vez de
# aprender a recusar. Aqui cada resposta ecoa o pedido especifico — isso
# torna o texto unico por construcao, nao so a pergunta.

_PEDIDOS_PRESCRICAO = (
    "amoxicilina para minha esposa",
    "a dose de dipirona para uma crianca",
    "quanto de insulina aplicar agora",
    "a posologia do antibiotico do protocolo de sepse",
    "a receita pronta para o paciente 12",
    "quantos ml de soro correr por hora",
    "a dose exata de morfina",
    "o antitermico certo para febre alta",
    "quanto de heparina usar no TEP",
    "a dose de furosemida para o edema",
    "o anticoagulante e a dose para AVC",
    "quanto de potassio repor na cetoacidose",
    "a posologia do broncodilatador",
    "a dose do antimicrobiano da pneumonia",
)

_PERGUNTAS_RECUSA = (
    "Pode me passar {pedido}?",
    "Preciso saber {pedido}, pode prescrever direto?",
    "Qual {pedido}? E urgente.",
    "Me diga {pedido} sem precisar de medico.",
)

_ABERTURAS_RECUSA = (
    "Nao posso prescrever nem informar dose exata.",
    "Este assistente nao emite posologia.",
    "Prescricao nao e algo que este sistema faz.",
    "Dose e decisao medica, nao deste assistente.",
)


def gerar_recusas() -> list[Exemplo]:
    exemplos: list[Exemplo] = []
    for i, pedido in enumerate(_PEDIDOS_PRESCRICAO):
        pergunta_template = _PERGUNTAS_RECUSA[i % len(_PERGUNTAS_RECUSA)]
        abertura = _ABERTURAS_RECUSA[i % len(_ABERTURAS_RECUSA)]
        pergunta = pergunta_template.format(pedido=pedido)
        resposta = (
            f"{abertura} Sobre {pedido}: a posologia e definida pelo medico "
            f"responsavel, conforme a tabela de padronizacao da Farmacia "
            f"Clinica Central. {safety.ENCAMINHAMENTO_VALIDACAO}"
        )
        exemplos.append(
            Exemplo(
                instruction=INSTRUCAO_PADRAO,
                input=pergunta,
                output=resposta,
                categoria="recusa",
                origem=f"gerador_recusas:{i}",
            )
        )
    return exemplos


_PEDIDOS_FORA_ESCOPO = (
    "qual o cardapio do refeitorio hoje",
    "como chegar ao estacionamento do hospital",
    "qual o horario de visita da UTI",
    "onde fica o RH do hospital",
    "quem e o diretor clinico do HSA",
    "como faco para trocar de plano de saude",
    "qual o telefone da ouvidoria",
    "como funciona o convenio com a prefeitura",
    "vai chover amanha na cidade",
    "quero saber sobre o resultado do jogo de ontem",
    "pode me ajudar com a declaracao de imposto de renda",
    "qual a previsao do dolar essa semana",
    "quero uma receita de bolo",
    "quem ganhou a eleicao no municipio",
)

_PERGUNTAS_FORA_ESCOPO = (
    "{pedido}?",
    "Voce sabe {pedido}?",
    "Me diga {pedido}.",
)


def gerar_fora_escopo() -> list[Exemplo]:
    exemplos: list[Exemplo] = []
    for i, pedido in enumerate(_PEDIDOS_FORA_ESCOPO):
        template = _PERGUNTAS_FORA_ESCOPO[i % len(_PERGUNTAS_FORA_ESCOPO)]
        pergunta = template.format(pedido=pedido)
        resposta = (
            f"Isso esta fora do escopo deste assistente clinico interno — a "
            f"pergunta sobre '{pedido}' nao e sobre protocolo institucional "
            f"nem sobre paciente internado. Procure o canal administrativo "
            f"apropriado do {canon.HOSPITAL_SIGLA}."
        )
        exemplos.append(
            Exemplo(
                instruction=INSTRUCAO_PADRAO,
                input=pergunta,
                output=resposta,
                categoria="fora_escopo",
                origem=f"gerador_fora_escopo:{i}",
            )
        )
    return exemplos


def gerar_corpus_sintetico() -> list[Exemplo]:
    return [
        *gerar_faq(),
        *gerar_documentos_modelo(),
        *gerar_recusas(),
        *gerar_fora_escopo(),
    ]


# --------------------------------------------------------------------------- #
# Ingestao externa — PubMedQA e MedQuAD
# --------------------------------------------------------------------------- #


def carregar_pubmedqa(caminho_jsonl: Path, limite: int | None = None) -> list[Exemplo]:
    """Le o PQA-L rotulado do PubMedQA (formato: um objeto JSON por linha)."""
    exemplos: list[Exemplo] = []
    with caminho_jsonl.open(encoding="utf-8") as f:
        for n, linha in enumerate(f, start=1):
            if limite is not None and len(exemplos) >= limite:
                break
            registro = json.loads(linha)
            pergunta = registro.get("QUESTION") or registro.get("question")
            resposta = registro.get("LONG_ANSWER") or registro.get("long_answer")
            if not pergunta or not resposta:
                continue
            if len(resposta) > 2000:
                continue  # descarta em vez de truncar — corte no meio ensina a parar mal
            exemplos.append(
                Exemplo(
                    instruction=INSTRUCAO_PADRAO,
                    input=pergunta,
                    output=resposta,
                    categoria="externo_pubmedqa",
                    origem=f"{caminho_jsonl.name}:{n}",
                )
            )
    return exemplos


def carregar_medquad(pasta_raiz: Path, limite: int | None = None) -> list[Exemplo]:
    """Le o MedQuAD (XMLs por pasta de fonte), pulando pastas com <Answer/> vazio."""
    import xml.etree.ElementTree as ET

    from src.config import MEDQUAD_PASTAS_IGNORADAS

    exemplos: list[Exemplo] = []
    for pasta in sorted(pasta_raiz.iterdir()):
        if not pasta.is_dir() or pasta.name in MEDQUAD_PASTAS_IGNORADAS:
            continue
        for arquivo in sorted(pasta.glob("*.xml")):
            if limite is not None and len(exemplos) >= limite:
                return exemplos
            try:
                raiz = ET.parse(arquivo).getroot()
            except ET.ParseError:
                continue
            for qa_pair in raiz.iter("QAPair"):
                pergunta_el = qa_pair.find("Question")
                resposta_el = qa_pair.find("Answer")
                if pergunta_el is None or resposta_el is None:
                    continue
                pergunta = (pergunta_el.text or "").strip()
                resposta = (resposta_el.text or "").strip()
                if not pergunta or not resposta or len(resposta) > 2000:
                    continue
                exemplos.append(
                    Exemplo(
                        instruction=INSTRUCAO_PADRAO,
                        input=pergunta,
                        output=resposta,
                        categoria="externo_medquad",
                        origem=f"{pasta.name}/{arquivo.name}",
                    )
                )
    return exemplos


# --------------------------------------------------------------------------- #
# Pipeline: normalizar -> anonimizar -> curar -> deduplicar -> truncar -> dividir
# --------------------------------------------------------------------------- #

_PADRAO_CPF = re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b")
_PADRAO_TELEFONE = re.compile(r"\b(?:\(\d{2}\)\s?)?\d{4,5}-\d{4}\b")
_PADRAO_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PADRAO_NOME_PACIENTE = re.compile(
    r"\bpaciente\s+([A-ZÀ-Ú][a-zà-ú]+(?:\s+[A-ZÀ-Ú][a-zà-ú]+){0,3})", re.IGNORECASE
)


def normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFC", texto)
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


def anonimizar_texto(texto: str) -> str:
    """PII vira marcador nomeado — substituicao, nunca remocao."""
    texto = _PADRAO_CPF.sub("[CPF]", texto)
    texto = _PADRAO_TELEFONE.sub("[TELEFONE]", texto)
    texto = _PADRAO_EMAIL.sub("[EMAIL]", texto)
    texto = _PADRAO_NOME_PACIENTE.sub("paciente [NOME]", texto)
    return texto


def normalizar_estagio(exemplos: list[Exemplo]) -> list[Exemplo]:
    return [e.com(input=normalizar(e.input), output=normalizar(e.output)) for e in exemplos]


def anonimizar_estagio(exemplos: list[Exemplo]) -> list[Exemplo]:
    return [
        e.com(input=anonimizar_texto(e.input), output=anonimizar_texto(e.output))
        for e in exemplos
    ]


def curar_estagio(exemplos: list[Exemplo]) -> list[Exemplo]:
    """Descarta o que ensinaria a prescrever — templates em branco ficam."""
    mantidos = []
    for e in exemplos:
        if e.categoria == "documento":
            mantidos.append(e)  # formato de documento e o que o enunciado pede
            continue
        deteccao = safety.detectar_prescricao(e.output, "resposta", tem_fonte=True)
        if deteccao.disparou:
            continue
        mantidos.append(e)
    return mantidos


def deduplicar_estagio(exemplos: list[Exemplo]) -> list[Exemplo]:
    vistos: set[tuple[str, str, str]] = set()
    mantidos = []
    for e in exemplos:
        chave = (e.instruction, e.input, e.output)
        if chave in vistos:
            continue
        vistos.add(chave)
        mantidos.append(e)
    return mantidos


def truncar_estagio(exemplos: list[Exemplo], limite_chars: int = 2000) -> list[Exemplo]:
    """Descarta resposta longa demais — nunca corta no meio da frase."""
    return [e for e in exemplos if len(e.output) <= limite_chars]


def dividir_estagio(
    exemplos: list[Exemplo], proporcao_holdout: float, seed: int
) -> tuple[list[Exemplo], list[Exemplo]]:
    """Split estratificado por categoria — cada categoria contribui para o holdout."""
    rng = random.Random(seed)
    por_categoria: dict[str, list[Exemplo]] = {}
    for e in exemplos:
        por_categoria.setdefault(e.categoria, []).append(e)

    treino: list[Exemplo] = []
    holdout: list[Exemplo] = []
    for categoria, grupo in por_categoria.items():
        grupo_embaralhado = grupo[:]
        rng.shuffle(grupo_embaralhado)
        n_holdout = max(1, round(len(grupo_embaralhado) * proporcao_holdout)) if len(grupo_embaralhado) > 1 else 0
        holdout.extend(grupo_embaralhado[:n_holdout])
        treino.extend(grupo_embaralhado[n_holdout:])
    return treino, holdout


def preparar_dataset(
    exemplos_brutos: list[Exemplo],
    proporcao_holdout: float,
    seed: int,
) -> tuple[list[Exemplo], list[Exemplo], RelatorioPreparo]:
    """Roda o pipeline inteiro, nesta ordem, e devolve treino, holdout e relatorio."""
    relatorio = RelatorioPreparo(entradas_por_estagio={})
    relatorio.registrar("bruto", len(exemplos_brutos))

    passo = normalizar_estagio(exemplos_brutos)
    relatorio.registrar("normalizado", len(passo))

    passo = anonimizar_estagio(passo)
    relatorio.registrar("anonimizado", len(passo))

    passo = curar_estagio(passo)
    relatorio.registrar("curado", len(passo))

    passo = deduplicar_estagio(passo)
    relatorio.registrar("deduplicado", len(passo))

    passo = truncar_estagio(passo)
    relatorio.registrar("truncado", len(passo))

    treino, holdout = dividir_estagio(passo, proporcao_holdout, seed)
    relatorio.registrar("treino", len(treino))
    relatorio.registrar("holdout", len(holdout))

    return treino, holdout, relatorio


# --------------------------------------------------------------------------- #
# Contabilidade e serializacao
# --------------------------------------------------------------------------- #


def contar_categorias(exemplos: list[Exemplo]) -> dict[str, dict[str, int]]:
    """Total e respostas DISTINTAS por categoria — o que teria pego o bug antigo."""
    contagem: dict[str, dict[str, int]] = {}
    for e in exemplos:
        c = contagem.setdefault(e.categoria, {"total": 0, "respostas_distintas": 0})
        c["total"] += 1
    respostas_por_categoria: dict[str, set[str]] = {}
    for e in exemplos:
        respostas_por_categoria.setdefault(e.categoria, set()).add(e.output)
    for categoria, respostas in respostas_por_categoria.items():
        contagem[categoria]["respostas_distintas"] = len(respostas)
    return contagem


def serializar(exemplos: list[Exemplo], caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", encoding="utf-8") as f:
        for e in exemplos:
            f.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")


def carregar(caminho: Path) -> list[Exemplo]:
    exemplos = []
    with caminho.open(encoding="utf-8") as f:
        for linha in f:
            exemplos.append(Exemplo(**json.loads(linha)))
    return exemplos
