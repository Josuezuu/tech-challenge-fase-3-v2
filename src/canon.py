"""Fonte unica de vocabulario do Hospital Santa Amalia (HSA), fonte fictício.

Protocolos, FAQ, documentos-modelo, seed do banco e o RAG leem daqui. E o que
garante que o protocolo de sepse cite o mesmo HEMOGRAMA_COMPLETO que a tabela
`exames` do banco tem — sem isso a incoerencia aparece la na frente disfarcada
de "o RAG esta ruim".

Versao reduzida do canone do projeto anterior: 6 condicoes (nao 17) e os
exames que elas realmente citam (nao o catalogo inteiro).
"""

from __future__ import annotations

from dataclasses import dataclass

HOSPITAL_NOME: str = "Hospital Santa Amalia"
HOSPITAL_SIGLA: str = "HSA"
HOSPITAL: str = f"{HOSPITAL_NOME} ({HOSPITAL_SIGLA})"

#: Muda sempre que o vocabulario muda; o MANIFEST do corpus grava esta versao.
VERSAO_CANON: str = "2.0"

REGISTRO_MASCARADO: str = "CRM_XXXXX"

GRAVIDADES: tuple[str, ...] = ("baixa", "media", "alta", "critica")
SENTIDOS_CRITICOS: tuple[str, ...] = ("acima", "abaixo", "nenhum")


@dataclass(frozen=True)
class Setor:
    sigla: str
    nome: str
    ramal: str


@dataclass(frozen=True)
class Exame:
    codigo: str
    nome: str
    unidade: str
    faixa_referencia: str
    limite_critico: float | None
    sentido_critico: str

    def critico_se(self, valor: float | None) -> bool:
        """Diz se `valor` e critico para este exame."""
        if valor is None or self.limite_critico is None:
            return False
        if self.sentido_critico == "acima":
            return valor > self.limite_critico
        if self.sentido_critico == "abaixo":
            return valor < self.limite_critico
        return False


@dataclass(frozen=True)
class Condicao:
    chave: str
    nome: str
    sinonimos: tuple[str, ...]
    gravidade: str
    setor_sigla: str
    exames_obrigatorios: tuple[str, ...]
    criterios_inclusao: tuple[str, ...]
    conduta: tuple[str, ...]
    contraindicacoes: tuple[str, ...]
    janela_reavaliacao_h: int


@dataclass(frozen=True)
class Profissional:
    codigo: str
    especialidade: str
    setor_sigla: str
    registro: str = REGISTRO_MASCARADO


SETORES: tuple[Setor, ...] = (
    Setor("EMA-B", "Emergencia Adulto - Ala B", "4021"),
    Setor("UCO", "Unidade Coronariana - Bloco Sul", "4115"),
    Setor("UAVC", "Unidade de AVC - Bloco Norte", "4120"),
    Setor("CLM-3", "Clinica Medica - 3o andar", "4210"),
)

EXAMES: tuple[Exame, ...] = (
    Exame("HEMOGRAMA_COMPLETO", "Hemograma completo (leucocitos)", "10^3/uL", "4,0 a 11,0", 20.0, "acima"),
    Exame("LACTATO_ARTERIAL", "Lactato arterial", "mmol/L", "0,5 a 1,6", 4.0, "acima"),
    Exame("PROCALCITONINA", "Procalcitonina", "ng/mL", "ate 0,5", 2.0, "acima"),
    Exame("HEMOCULTURA_2_AMOSTRAS", "Hemocultura, duas amostras", "laudo", "sem crescimento em 5 dias", None, "nenhum"),
    Exame("ECG_12_DERIVACOES", "Eletrocardiograma de 12 derivacoes", "laudo", "ritmo sinusal sem supradesnivelamento", None, "nenhum"),
    Exame("TROPONINA_ULTRASSENSIVEL", "Troponina I ultrassensivel", "ng/L", "ate 14", 52.0, "acima"),
    Exame("CKMB_MASSA", "CK-MB massa", "ng/mL", "ate 5,0", 25.0, "acima"),
    Exame("TC_CRANIO_SEM_CONTRASTE", "Tomografia de cranio sem contraste", "laudo", "sem sangramento agudo", None, "nenhum"),
    Exame("GLICEMIA_CAPILAR", "Glicemia capilar", "mg/dL", "70 a 99", 400.0, "acima"),
    Exame("INR_TAP", "INR / tempo de protrombina", "razao", "0,9 a 1,2", 4.5, "acima"),
    Exame("RX_TORAX_PA_PERFIL", "Radiografia de torax PA e perfil", "laudo", "sem consolidacao ou derrame", None, "nenhum"),
    Exame("PCR_QUANTITATIVA", "Proteina C reativa quantitativa", "mg/L", "ate 5,0", 100.0, "acima"),
    Exame("SATURACAO_O2", "Saturacao periferica de oxigenio", "%", "95 a 100", 88.0, "abaixo"),
    Exame("GASOMETRIA_ARTERIAL", "Gasometria arterial (pH)", "pH", "7,35 a 7,45", 7.20, "abaixo"),
    Exame("POTASSIO_SERICO", "Potassio serico", "mEq/L", "3,5 a 5,0", 6.5, "acima"),
    Exame("EAS_URINA_TIPO_I", "EAS - urina tipo I", "laudo", "sem leucocituria ou nitrito", None, "nenhum"),
    Exame("UROCULTURA", "Urocultura com contagem de colonias", "UFC/mL", "abaixo de 10.000", 100000.0, "acima"),
    Exame("CREATININA_SERICA", "Creatinina serica", "mg/dL", "0,6 a 1,2", 3.0, "acima"),
)

CODIGOS_EXAMES: frozenset[str] = frozenset(exame.codigo for exame in EXAMES)

CONDICOES: tuple[Condicao, ...] = (
    Condicao(
        chave="sepse_adulto",
        nome="Sepse em Adulto",
        sinonimos=("sepse", "choque septico", "quadro septico", "infeccao grave"),
        gravidade="critica",
        setor_sigla="EMA-B",
        exames_obrigatorios=("HEMOGRAMA_COMPLETO", "LACTATO_ARTERIAL", "HEMOCULTURA_2_AMOSTRAS", "PROCALCITONINA"),
        criterios_inclusao=(
            "Suspeita ou confirmacao de foco infeccioso em paciente com 18 anos ou mais.",
            "Disfuncao organica nova atribuivel a infeccao, ainda que isolada.",
        ),
        conduta=(
            "Abrir a pulseira laranja de sepse do HSA e registrar o horario zero na evolucao.",
            "Coletar o pacote laboratorial obrigatorio antes da primeira dose de antimicrobiano.",
            (
                "Solicitar antimicrobiano empirico a Farmacia Clinica Central conforme a tabela "
                "HSA-FARM-02; a posologia e definida pelo medico assistente, nao pelo protocolo."
            ),
            "Reavaliar perfusao e lactato dentro da janela de reavaliacao declarada.",
        ),
        contraindicacoes=("Nao retardar o antimicrobiano a espera de exame de imagem.",),
        janela_reavaliacao_h=1,
    ),
    Condicao(
        chave="dor_toracica",
        nome="Dor Toracica Aguda",
        sinonimos=("dor precordial", "dor no peito", "precordialgia", "DT"),
        gravidade="alta",
        setor_sigla="UCO",
        exames_obrigatorios=("ECG_12_DERIVACOES", "TROPONINA_ULTRASSENSIVEL", "CKMB_MASSA"),
        criterios_inclusao=("Dor toracica com menos de 24 horas de inicio em paciente adulto.",),
        conduta=(
            "Realizar o eletrocardiograma em ate 10 minutos da chegada.",
            "Coletar a primeira troponina e programar a segunda dentro da janela de reavaliacao.",
            "Acionar a Unidade Coronariana quando houver supradesnivelamento ou troponina critica.",
        ),
        contraindicacoes=("Nao liberar o paciente antes da segunda curva de troponina.",),
        janela_reavaliacao_h=3,
    ),
    Condicao(
        chave="avc_isquemico",
        nome="AVC Isquemico Agudo",
        sinonimos=("AVC", "acidente vascular cerebral", "derrame", "AVCi"),
        gravidade="critica",
        setor_sigla="UAVC",
        exames_obrigatorios=("TC_CRANIO_SEM_CONTRASTE", "GLICEMIA_CAPILAR", "INR_TAP"),
        criterios_inclusao=("Deficit neurologico focal de instalacao subita com horario de inicio conhecido.",),
        conduta=(
            "Disparar o codigo azul-neuro do HSA e cronometrar a porta-tomografia.",
            "Excluir hipoglicemia com glicemia capilar antes de qualquer decisao terapeutica.",
            "Levar o paciente a tomografia sem contraste antes da avaliacao de terapia de reperfusao.",
        ),
        contraindicacoes=("Nao indicar terapia de reperfusao com horario de inicio desconhecido.",),
        janela_reavaliacao_h=1,
    ),
    Condicao(
        chave="pneumonia_comunitaria",
        nome="Pneumonia Adquirida na Comunidade",
        sinonimos=("PAC", "pneumonia", "pneumonia comunitaria"),
        gravidade="alta",
        setor_sigla="CLM-3",
        exames_obrigatorios=("RX_TORAX_PA_PERFIL", "HEMOGRAMA_COMPLETO", "PCR_QUANTITATIVA", "SATURACAO_O2"),
        criterios_inclusao=("Tosse, febre ou dispneia com achado radiologico compativel.",),
        conduta=(
            "Calcular o escore de gravidade e registrar o resultado.",
            "Coletar o pacote laboratorial obrigatorio antes do antimicrobiano.",
            "Solicitar o esquema empirico conforme a tabela HSA-FARM-05.",
        ),
        contraindicacoes=("Nao repetir a radiografia antes de 48 horas sem piora clinica.",),
        janela_reavaliacao_h=6,
    ),
    Condicao(
        chave="cetoacidose_diabetica",
        nome="Cetoacidose Diabetica",
        sinonimos=("CAD", "cetoacidose", "descompensacao diabetica"),
        gravidade="critica",
        setor_sigla="EMA-B",
        exames_obrigatorios=("GLICEMIA_CAPILAR", "GASOMETRIA_ARTERIAL", "POTASSIO_SERICO"),
        criterios_inclusao=("Hiperglicemia com acidose metabolica documentada em gasometria arterial.",),
        conduta=(
            "Instalar a folha de controle horario desde a admissao.",
            "Corrigir volume antes de iniciar insulinoterapia continua.",
            "Repor potassio conforme a tabela HSA-FARM-11 antes de qualquer bomba de insulina.",
        ),
        contraindicacoes=("Nao iniciar insulina com potassio abaixo da faixa de referencia.",),
        janela_reavaliacao_h=2,
    ),
    Condicao(
        chave="itu_complicada",
        nome="Infeccao do Trato Urinario Complicada",
        sinonimos=("ITU complicada", "pielonefrite", "infeccao urinaria alta"),
        gravidade="media",
        setor_sigla="CLM-3",
        exames_obrigatorios=("EAS_URINA_TIPO_I", "UROCULTURA", "CREATININA_SERICA"),
        criterios_inclusao=("Sintoma urinario com febre, dor lombar ou repercussao sistemica.",),
        conduta=(
            "Coletar urocultura antes da primeira dose de antimicrobiano.",
            "Solicitar esquema empirico conforme a tabela HSA-FARM-06.",
            "Reavaliar em 72 horas com o resultado da urocultura em maos.",
        ),
        contraindicacoes=("Nao tratar bacteriuria assintomatica fora das excecoes do protocolo.",),
        janela_reavaliacao_h=24,
    ),
)

CHAVES_CONDICOES: tuple[str, ...] = tuple(condicao.chave for condicao in CONDICOES)

EQUIPE_MEDICA: tuple[Profissional, ...] = (
    Profissional("MEDICO_A", "Medicina de Emergencia", "EMA-B"),
    Profissional("MEDICO_C", "Cardiologia", "UCO"),
    Profissional("MEDICO_D", "Neurologia Vascular", "UAVC"),
    Profissional("MEDICO_E", "Clinica Medica", "CLM-3"),
)


def codigo_protocolo(n: int) -> str:
    """Devolve o codigo interno do n-esimo protocolo: HSA-PROT-003."""
    if n <= 0:
        raise ValueError(f"numero de protocolo deve ser positivo, recebido {n}")
    return f"{HOSPITAL_SIGLA}-PROT-{n:03d}"


def codigo_paciente(n: int) -> str:
    """Devolve o pseudonimo estavel do n-esimo paciente: PACIENTE_003."""
    if n <= 0:
        raise ValueError(f"numero de paciente deve ser positivo, recebido {n}")
    return f"PACIENTE_{n:03d}"


def exame_por_codigo(codigo: str) -> Exame:
    for exame in EXAMES:
        if exame.codigo == codigo:
            return exame
    raise KeyError(f"exame {codigo!r} nao existe no canone {VERSAO_CANON}")


def condicao_por_chave(chave: str) -> Condicao:
    for condicao in CONDICOES:
        if condicao.chave == chave:
            return condicao
    raise KeyError(f"condicao {chave!r} nao existe no canone {VERSAO_CANON}")


def setor_por_sigla(sigla: str) -> Setor:
    for setor in SETORES:
        if setor.sigla == sigla:
            return setor
    raise KeyError(f"setor {sigla!r} nao existe no canone {VERSAO_CANON}")
