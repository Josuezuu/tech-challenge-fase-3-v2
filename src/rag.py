"""RAG: recuperar o trecho de protocolo que sustenta a resposta, com fonte.

Fonte ausente e reportada, nunca preenchida — se nada relevante for
recuperado, o sistema diz isso em vez de inventar uma fonte plausivel.
`chunk_id` e estavel (codigo do protocolo + indice de secao) porque e ele que
aparece na citacao; se mudasse a cada reindexacao, toda execucao gravada
apontaria para o nada.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from src import config, safety

_PADRAO_FRONTMATTER = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    texto: str
    codigo_fonte: str
    metadados: dict[str, str]


def _parse_frontmatter(conteudo: str) -> tuple[dict[str, str], str]:
    m = _PADRAO_FRONTMATTER.match(conteudo)
    if not m:
        return {}, conteudo
    bloco, corpo = m.groups()
    metadados = {}
    for linha in bloco.splitlines():
        if ":" in linha:
            chave, _, valor = linha.partition(":")
            metadados[chave.strip()] = valor.strip()
    return metadados, corpo


def carregar_protocolos(pasta: Path) -> list[tuple[dict[str, str], str]]:
    """Le cada .md preservando o front-matter como metadado."""
    documentos = []
    for caminho in sorted(pasta.glob("*.md")):
        metadados, corpo = _parse_frontmatter(caminho.read_text(encoding="utf-8"))
        documentos.append((metadados, corpo))
    return documentos


def dividir_em_chunks(
    documentos: list[tuple[dict[str, str], str]], tamanho_max: int = 500
) -> list[Chunk]:
    """Divide por secao (##), preservando a origem e um chunk_id estavel."""
    chunks: list[Chunk] = []
    for metadados, corpo in documentos:
        codigo = metadados.get("codigo", "SEM_CODIGO")
        secoes = re.split(r"\n(?=## )", corpo)
        for i, secao in enumerate(secoes):
            secao = secao.strip()
            if not secao:
                continue
            titulo = secao.splitlines()[0].lstrip("# ").strip()
            for j in range(0, len(secao), tamanho_max):
                pedaco = secao[j : j + tamanho_max]
                indice_pedaco = f"{i}.{j // tamanho_max}"
                chunks.append(
                    Chunk(
                        chunk_id=f"{codigo} § {titulo} [{indice_pedaco}]",
                        texto=pedaco,
                        codigo_fonte=codigo,
                        metadados=metadados,
                    )
                )
    return chunks


class IndiceFAISS:
    """Envelope fino sobre langchain_community.vectorstores.FAISS."""

    def __init__(self, vectorstore) -> None:
        self._vectorstore = vectorstore

    @classmethod
    def construir(cls, chunks: list[Chunk], modelo_embedding: str = config.EMBEDDING_MODEL_ID) -> IndiceFAISS:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document

        embeddings = HuggingFaceEmbeddings(model_name=modelo_embedding)
        documentos = [
            Document(page_content=c.texto, metadata={"chunk_id": c.chunk_id, "codigo_fonte": c.codigo_fonte})
            for c in chunks
        ]
        vectorstore = FAISS.from_documents(documentos, embeddings)
        return cls(vectorstore)

    def salvar(self, caminho: Path) -> None:
        self._vectorstore.save_local(str(caminho))

    @classmethod
    def carregar(cls, caminho: Path, modelo_embedding: str = config.EMBEDDING_MODEL_ID) -> IndiceFAISS:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS

        embeddings = HuggingFaceEmbeddings(model_name=modelo_embedding)
        vectorstore = FAISS.load_local(
            str(caminho), embeddings, allow_dangerous_deserialization=True
        )
        return cls(vectorstore)

    def buscar_mmr(self, pergunta: str, k: int = config.RETRIEVER_K):
        """MMR: equilibra relevancia com diversidade — evita 4 trechos quase identicos."""
        return self._vectorstore.max_marginal_relevance_search(pergunta, k=k, fetch_k=k * 3)


@dataclass(frozen=True)
class ResultadoRecuperacao:
    documentos: list
    confianca: float


LIMIAR_CONFIANCA = 0.3


def filtrar_e_ranquear(pergunta: str, documentos: list) -> ResultadoRecuperacao:
    """Confianca = fracao de termos da pergunta presentes nos trechos recuperados."""
    termos = {t.lower() for t in re.findall(r"\w{4,}", pergunta)}
    if not termos or not documentos:
        return ResultadoRecuperacao(documentos=[], confianca=0.0)

    texto_recuperado = " ".join(d.page_content.lower() for d in documentos)
    termos_presentes = sum(1 for t in termos if t in texto_recuperado)
    confianca = termos_presentes / len(termos)
    return ResultadoRecuperacao(documentos=documentos, confianca=confianca)


def gerar_resposta_com_fonte(provider, pergunta: str, resultado: ResultadoRecuperacao) -> str:
    """Redige com base SO no contexto recuperado, citando a fonte ou dizendo que falta."""
    if resultado.confianca < LIMIAR_CONFIANCA or not resultado.documentos:
        return safety.FALLBACK_CORPUS

    contexto = "\n\n".join(
        f"[{d.metadata['chunk_id']}]\n{d.page_content}" for d in resultado.documentos
    )
    prompt = (
        "Responda a pergunta do medico usando SOMENTE o contexto abaixo. "
        "Cite a fonte entre colchetes, no formato [codigo do protocolo].\n\n"
        f"Contexto:\n{contexto}\n\nPergunta: {pergunta}\nResposta:"
    )
    return provider.gerar(prompt)
