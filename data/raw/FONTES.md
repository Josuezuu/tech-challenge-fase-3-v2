# Fontes externas e licenca

| Fonte | Licenca | Como obter |
| --- | --- | --- |
| [PubMedQA (PQA-L rotulado)](https://github.com/pubmedqa/pubmedqa) | MIT | clonar o repo, converter `ori_pqal.json` para `pubmedqa.jsonl` (um objeto por linha, campos `QUESTION`/`LONG_ANSWER`) |
| [MedQuAD](https://github.com/abachaa/MedQuAD) | CC BY 4.0 | clonar o repo em `data/raw/medquad/` — a estrutura de pastas por fonte e usada como esta |

Pastas do MedQuAD com `<Answer></Answer>` vazio por direito autoral do MedlinePlus
(puladas por nome, nao por tentativa de limpeza — ver `src/data.py:carregar_medquad`):

- `10_MPlus_ADAM_QA`
- `11_MPlusDrugs_QA`
- `12_MPlusHerbsSupplements_QA`

O corpus sintetico (protocolos, FAQ, documentos-modelo, recusa, fora_escopo) e gerado
por `src/data.py` a partir de `src/canon.py` — nenhum dado real de paciente em lugar
nenhum deste repositorio.
