# Projeto: Classificação de Autoimunidade Induzida por Fármacos (InterDIA)

## Visão geral
Replicação do artigo InterDIA (doi: 10.1016/j.ejmech.2025.116006) em R, usando apenas `caret` com os métodos Random Forest, SVM radial e rede neural (nnet/MLP). O script principal `treinamento_completo.R` reproduz o pré-processamento, seleção de features (top 65 RDKit) e treinos com balanceamento. Um ensemble simples (EEC) simula o Easy Ensemble do artigo via bagging balanceado de Random Forests (undersampling).

## Dados
- `dataset/DIA_trainingset_RDKit_descriptors.csv`
- `dataset/DIA_testset_RDKit_descriptors.csv`
- Coluna `SMILES` removida; rótulo `Label` convertido para fatores `Classe0`/`Classe1`.

## Dependências principais
- R >= 4.x, pacotes: `caret`, `pROC`, `randomForest`, `gbm`, `nnet`, `doParallel` (opcional para paralelização), `DMwR`/`smotefamily` se quiser usar SMOTE. O script não baixa nada automaticamente.

## Como rodar
1. Abra o diretório do projeto no R/RStudio.
2. Instale dependências se necessário, por exemplo:  
   `install.packages(c("caret","pROC","randomForest","gbm","nnet","doParallel"))`
3. Execute: `source("treinamento_completo.R")`  
   O script já inclui pré-processamento, seleção de 65 features e treinos com validação cruzada balanceada.
4. Saídas principais: modelos `*.rds`, `resultados_completos.rds`, `comparacao_modelos.rds`, `curvas_roc.png`.

## Resultados (teste externo)
| Modelo                 | Acurácia | Sensibilidade | Especificidade | F1     | MCC    | AUC    |
|------------------------|----------|---------------|----------------|--------|--------|--------|
| SVM Radial             | 0.8500   | 0.5000        | 0.9667         | 0.6250 | 0.5659 | 0.7996 |
| Easy Ensemble (EEC)    | 0.8000   | 0.7000        | 0.8333         | 0.6364 | 0.5040 | 0.9022 |
| Random Forest          | 0.7833   | 0.3333        | 0.9333         | 0.4348 | 0.3397 | 0.8559 |
| Gradient Boosting (GBM)| 0.7917   | 0.4000        | 0.9222         | 0.4898 | 0.3822 | 0.8226 |
| Rede Neural (nnet)     | 0.7750   | 0.2667        | 0.9444         | 0.3721 | 0.2941 | 0.7907 |
| Artigo InterDIA (alvo) | ~0.85    | —             | —              | —      | —      | ~0.89  |

Notas sobre limiar de decisão:
- Cada modelo aplica limiar otimizado nas predições out-of-fold (`caret`): inicialmente por acurácia; se indisponível, usa Youden. Para SVM, o limiar atual (≈0.75) privilegiou especificidade, elevando acurácia, mas reduziu sensibilidade.
- EEC usa média das probabilidades de 20 RFs balanceados (undersampling), limiar por acurácia/Youden; oferece melhor AUC.

## Arquivos-chave
- `treinamento_completo.R`: fluxo completo (pré-processamento, seleção de 65 features, treinos RF/SVM/nnet, ensemble EEC, métricas e salvamento).
- `treino_random_forest.R`, `treino_svm_radial.R`, `treino_nnet.R`: treinos individuais simplificados.
- `curvas_roc.png`: comparação das curvas ROC geradas na última execução.

## Observações de reproduzibilidade
- O conjunto de teste é mantido fixo (120 amostras). Os seeds são fixados em 1.
- O balanceamento do EEC é undersampling aleatório por subconjunto; variar `n_subsamples` ou `ntree` pode alterar ligeiramente os números.
- Resultados dependem dos pacotes instalados e da versão do R; documente versões se for comparar.

## Próximos passos sugeridos
- Ajustar limiar do SVM/nnet para Youden ou balanced accuracy se quiser maior sensibilidade mantendo acurácia próxima do alvo do artigo.
- Se não for usar GBM, pode comentar/remover a seção 8 do script para acelerar execuções.
