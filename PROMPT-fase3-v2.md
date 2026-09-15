# Projeto: Tech Challenge FIAP Fase 3 — versão enxuta, treino local

Vamos construir do zero um projeto novo em `C:\Users\josue.oliveira\source\repos\tech-challenge-fase-3-v2`.
Ainda não existe nada lá. Não crie nada antes de terminar o levantamento do passo 1.

## Quem sou e o que preciso

Aluno da FIAP Pós Tech, turma 8IADT, Fase 3. O Tech Challenge pede um
**assistente virtual médico com uma LLM customizada por fine-tuning**, orquestrado
com LangChain e LangGraph. Entrego código no GitHub, um relatório técnico e um
vídeo de até 15 minutos.

O que o enunciado exige, item por item:

| Requisito |
| --- |
| Fine-tuning de LLM com dados médicos |
| Preprocessing, anonimização e curadoria do dataset |
| Pipeline LangChain integrando a LLM customizada |
| Consulta a base estruturada de prontuários |
| Contextualizar a resposta com dados do paciente |
| Limite de atuação — **nunca prescrever sem validação humana** (destacado no enunciado) |
| Logging detalhado para auditoria |
| Explainability — toda resposta indica a fonte |
| Fluxos em LangGraph |
| Dataset anonimizado ou sintético |
| Avaliação e análise dos resultados |

## Por que um projeto novo

Já existe um projeto meu, completo e entregável, em
`C:\Users\josue.oliveira\source\repos\tech-challenge-fase-3`. Ele funciona, tem 928
testes e relatório técnico gerado. O problema é outro: ele ficou grande demais
(72 módulos em `src/`), eu me perdi dentro dele, e a avaliação do fine-tuning é
fraca — a tabela principal roda com **n=5**.

O objetivo do projeto novo é: **um décimo do tamanho, com uma avaliação de
fine-tuning muito melhor.** Alvo de 8 a 12 arquivos em `src/`, não 72.

## O que reaproveitar, e como

**Do meu repositório antigo — pode copiar código à vontade, é meu.**
Leia antes de decidir o que vale trazer:

- `src/data/canon.py` e os geradores (`gerador_protocolos`, `gerador_faq`,
  `gerador_documentos`) — a ideia de um cânone único que faz FAQ, protocolo e
  laudo concordarem entre si é boa e vale manter, mas **simplifique muito**.
- `src/finetune/prompt_template.py` — o template único entre treino e inferência.
  Essa invariante é crítica: se o prompt do treino e o da inferência divergirem,
  a avaliação mede formatação em vez de conteúdo.
- `src/safety/` — os guardrails de prescrição. Funcionam e são a espinha do
  requisito destacado em amarelo.
- `docs/RELATORIO.md` — a estrutura do relatório e o tom. O relatório novo pode
  seguir o mesmo esqueleto.
- `docs/AULA.md` — explica o projeto antigo inteiro; use como mapa do que existe.

**Do projeto de outra aluna, `https://github.com/oryange/tech-challenge-group-24`
— leia, NÃO copie código.** Somos da mesma turma e da mesma fase; dois repositórios
com código coincidente é problema para os dois. O que eu quero de lá é **o método
de avaliação**, que é melhor que o meu:

- avaliar geração em **n≈50**, não em 5;
- **intervalo de confiança por bootstrap** (10.000 reamostragens, IC 95%) no delta
  entre baseline e fine-tunado;
- registrar **loss de validação**, não só de treino;
- comparar **checkpoints intermediários** contra o final;
- reportar baseline × fine-tunado no mesmo eixo, com ROUGE-L e BLEU-4.

Ela achou algo interessante que eu quero conseguir reproduzir: o checkpoint de
menor loss de validação gerou texto *pior* que um de loss maior. Quero poder
fazer essa análise, e para isso preciso salvar checkpoints intermediários.

## Hardware e ambiente — restrições reais desta máquina

- **GPU: RTX 4060 MSI 8 GB** (Ada, sm_89, bf16 nativo). O treino roda **local**,
  não no Colab.
- Windows 11, 15,8 GB de RAM, VSCode + terminal. O shell do Claude Code é Git Bash;
  meu terminal é PowerShell.
- **O antivírus Cylance mata processos longos iniciados de dentro da sessão do
  Claude Code.** Todo treino e toda avaliação pesada eu rodo no MEU terminal, e
  te passo o arquivo de saída. Nunca dispare treino você mesmo — escreva o script
  e me diga o comando.
- Um comando por vez no PowerShell. Bloco colado roda pela metade.
- Não existe `jq` nesta máquina.
- Nunca use `git add .` — sempre caminho explícito.
- Comando pesado em foreground; rode `ollama stop` antes se o Ollama estiver ativo.

Para o QLoRA do Phi-3-mini em 8 GB, o orçamento de VRAM é este e cabe com folga:
pesos 4-bit ~2,3 GB + adapters/otimizador ~0,15 GB + ativações ~1,0 GB ≈ 3,5 GB.
Use `attn_implementation="sdpa"` (flash-attn não compila bem no Windows) e
`bitsandbytes>=0.43`, que é a versão com suporte oficial a Windows.

## Decisões já tomadas — não reabra

- **Modelo base:** `microsoft/Phi-3-mini-4k-instruct`, 4 bits NF4, double quant,
  compute em bfloat16.
- **`target_modules` do LoRA: `["qkv_proj", "o_proj"]`.** O Phi-3 funde Q/K/V num
  único `qkv_proj`. Copiar `q_proj`/`v_proj` da família Llama produz um treino que
  roda, não dá erro, e não aprende nada. Verifique contra os módulos reais do
  modelo antes de treinar.
- **Ponto de partida dos hiperparâmetros:** r=16, alpha=32, dropout=0.05, lr=2e-4,
  3 épocas, batch 1, grad_accum 16, max_seq_length 768, warmup_ratio 0.03, seed 42.
  Com 8 GB dá para tentar seq 1024 ou batch 2 — teste.
- **Salvar checkpoint a cada N passos** e avaliar geração em pelo menos dois deles.
- **Embeddings do RAG:** `intfloat/multilingual-e5-base` + FAISS.
- **LLM de produção do grafo:** OpenAI por variável de ambiente, com o modelo
  fine-tunado como ponta alternativa para o comparativo. Motivo medido no projeto
  antigo: um Phi-3-mini quantizado **não obedece JSON schema** — ele devolveu
  `nota: 3` num campo `Field(ge=0.0, le=1.0)` e derrubou metade das execuções.
  Ele serve como gerador, não como orquestrador. Desenhe o grafo sabendo disso.

## Dados

Três fontes, e a proporção entre elas vai declarada no relatório:

1. **PubMedQA** (`https://github.com/pubmedqa/pubmedqa`, PQA-L rotulado, MIT) — inglês.
2. **MedQuAD** (`https://github.com/abachaa/MedQuAD`, CC BY 4.0) — inglês.
   Atenção: as coleções `10_MPlus_ADAM_QA`, `11_MPlusDrugs_QA` e
   `12_MPlusHerbsSupplements_QA` vêm com `<Answer></Answer>` vazio por direito
   autoral do MedlinePlus. Pule pelo nome da pasta. Descarte resposta acima de
   ~2000 caracteres em vez de truncar: exemplo cortado no meio ensina o modelo a
   parar no meio da frase.
3. **Sintético em pt-BR que você vai gerar**: protocolos assistenciais, FAQ médica,
   modelos de laudo/receita/procedimento e prontuários de um hospital fictício.

Alvo: **maioria pt-BR**, com o externo em inglês como complemento de linguagem
clínica geral. Divida o orçamento igualmente entre PubMedQA e MedQuAD — o MedQuAD
é uma ordem de grandeza maior e afogaria o outro numa divisão proporcional.

**Erro que eu cometi no projeto antigo e não quero repetir.** Gerei 32 exemplos de
recusa e todos os 32 tinham **exatamente a mesma resposta**; só a pergunta variava.
A deduplicação, corretamente, cortou 16 deles, e sobraram 13 no treino — 2% do
dataset, com um único texto. O modelo não aprendeu a recusar: decorou uma frase.
No holdout, quando perguntaram "pode prescrever direto para meu paciente?", ele
respondeu **"Sim, desde que o paciente tenha mais de 60 anos"** e citou um
protocolo inexistente para dar respaldo.

Então: **`recusa` e `fora_escopo` precisam de texto variado e de volume real.**
Mínimo 8% do dataset final cada, com dezenas de formulações distintas. Verifique
a contagem de respostas *distintas* por categoria depois da dedup, não só o total.

Ordem obrigatória do preparo: **anonimizar antes de curar, deduplicar antes de
dividir.** Split com holdout de verdade, e cada exemplo carrega a `origem`
(arquivo:linha) para provar que o holdout nunca entrou no treino.

## Avaliação — é aqui que o projeto precisa ser melhor que o antigo

1. **Fine-tuning:** baseline (Phi-3 sem adapter) × fine-tunado × pelo menos um
   checkpoint intermediário, em **n≥50 do holdout**, decodificação greedy,
   ROUGE-L e BLEU-4, **com IC 95% por bootstrap** no delta. Reporte também o
   número de caracteres por resposta: ROUGE entre respostas de comprimentos muito
   diferentes mede comprimento tanto quanto conteúdo.
2. **Curvas de loss de treino E de validação**, salvas em JSON e plotadas.
3. **Retrieval:** precision/recall/F1 por k, com o **teto de precision** na mesma
   tabela (`min(alvos,k)/k`) — sem ele, precision@8 parece desastre.
4. **RAG × sem RAG**, julgados contra o **mesmo contexto recuperado**. Julgar a
   ponta sem RAG contra contexto vazio dá "não sustentada" por construção e o
   comparativo vira tautologia.
5. **Suíte adversarial** contra o grafo compilado, não contra guardrail isolado:
   pedido de receita, pedido de dose, injeção de prompt, jailbreak de persona,
   insistência após recusa, dado de outro paciente, fora de escopo. Reporte taxa
   de bloqueio **e** taxa de falso positivo — as duas só significam algo juntas.
6. Todo número de tabela sai de um artefato JSON de execução, nunca digitado à mão.

## O que NÃO fazer

- Nada de spec-driven, PRD, specs, IDs de requisito (RF-xx, AD-xx, EN-x.x) em
  comentário ou documento. Comentário explica o que a função faz, e só.
- Nada de 900 testes. Uns 30 a 50, cobrindo o que quebra silencioso: divergência
  de template entre treino e inferência, vazamento do holdout para o treino,
  guardrail de prescrição, contagem do dataset.
- Nada de camada de abstração para um caso de uso só.
- Não mexa no repositório antigo. Ele é o meu plano B.

## Primeiro passo

Antes de escrever qualquer código:

1. Leia `C:\Users\josue.oliveira\source\repos\tech-challenge-fase-3` — comece por
   `docs/AULA.md` e `README.md`, depois `src/data/canon.py`, `src/safety/` e
   `src/finetune/prompt_template.py`. Me diga o que vale trazer e o que vale jogar fora.
2. Leia o relatório da colega em
   `https://github.com/oryange/tech-challenge-group-24/blob/main/docs/relatorio-tecnico.md`
   e liste o que o método dela tem que o meu não tinha.
3. Me proponha a **árvore de arquivos** do projeto novo, com uma linha dizendo o
   que cada arquivo faz, e a ordem em que vamos construir.

Não crie o diretório nem escreva arquivo antes de eu aprovar a árvore.
