from __future__ import annotations

from src.data import Exemplo
from src.prompt_template import diferenca_marcada, formatar_inferencia, formatar_treino

EXEMPLO = Exemplo(
    instruction="Responda com base no protocolo.",
    input="Qual a conduta para sepse?",
    output="[Fonte: HSA-PROT-001] Abrir a pulseira laranja.",
    categoria="faq",
    origem="teste:1",
)


def test_formatar_inferencia_termina_sem_eos():
    prompt = formatar_inferencia(EXEMPLO.instruction, EXEMPLO.input)
    assert prompt.endswith("### Resposta:\n")
    assert "<|endoftext|>" not in prompt


def test_formatar_treino_comeca_exatamente_com_formatar_inferencia():
    """A invariante central: divergir aqui invalida todo o comparativo em silencio."""
    treino = formatar_treino(EXEMPLO)
    inferencia = formatar_inferencia(EXEMPLO.instruction, EXEMPLO.input)
    assert treino.startswith(inferencia)


def test_formatar_treino_termina_com_eos():
    treino = formatar_treino(EXEMPLO, eos="<EOS>")
    assert treino.endswith("<EOS>")
    assert treino.endswith(EXEMPLO.output + "<EOS>")


def test_diferenca_marcada_vazia_quando_iguais():
    texto = formatar_treino(EXEMPLO)
    assert diferenca_marcada(texto, texto) == ""


def test_diferenca_marcada_aponta_o_primeiro_caractere_diferente():
    esperado = "### Instrucao:\nabc"
    recebido = "### Instrucao:\nabx"
    descricao = diferenca_marcada(esperado, recebido)
    assert "caractere 17" in descricao
