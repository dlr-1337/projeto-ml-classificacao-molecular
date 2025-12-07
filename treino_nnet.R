# =============================================================================
# Treinamento com Rede Neural (nnet/MLP) usando caret
# =============================================================================

# Carregar bibliotecas necessarias
library(caret)
library(nnet)

# -----------------------------------------------------------------------------
# 1. Carregar os dados
# -----------------------------------------------------------------------------
dados_treino <- read.csv("dataset/DIA_trainingset_RDKit_descriptors.csv")
dados_teste <- read.csv("dataset/DIA_testset_RDKit_descriptors.csv")

# -----------------------------------------------------------------------------
# 2. Preparar os dados
# -----------------------------------------------------------------------------
# Remover coluna SMILES (nao e descritor numerico)
dados_treino <- dados_treino[, !names(dados_treino) %in% c("SMILES")]
dados_teste <- dados_teste[, !names(dados_teste) %in% c("SMILES")]

# Converter Label para fator (classificacao)
dados_treino$Label <- as.factor(dados_treino$Label)
dados_teste$Label <- as.factor(dados_teste$Label)

# Renomear niveis para evitar problemas com nomes numericos
levels(dados_treino$Label) <- c("Classe0", "Classe1")
levels(dados_teste$Label) <- c("Classe0", "Classe1")

# Separar features e target
X_treino <- dados_treino[, -which(names(dados_treino) == "Label")]
y_treino <- dados_treino$Label

X_teste <- dados_teste[, -which(names(dados_teste) == "Label")]
y_teste <- dados_teste$Label

# Remover colunas com variancia zero ou near-zero variance
nzv <- nearZeroVar(X_treino)
if(length(nzv) > 0) {
  X_treino <- X_treino[, -nzv]
  X_teste <- X_teste[, -nzv]
}

# Remover colunas com NA
X_treino <- X_treino[, colSums(is.na(X_treino)) == 0]
X_teste <- X_teste[, names(X_treino)]

# -----------------------------------------------------------------------------
# 3. Pre-processamento (normalizacao - muito importante para redes neurais)
# -----------------------------------------------------------------------------
preProc <- preProcess(X_treino, method = c("center", "scale"))
X_treino_norm <- predict(preProc, X_treino)
X_teste_norm <- predict(preProc, X_teste)

# -----------------------------------------------------------------------------
# 4. Configurar o treinamento com validacao cruzada
# -----------------------------------------------------------------------------
ctrl <- trainControl(
  method = "cv",              # Validacao cruzada
  number = 10,                # 10-fold CV
  classProbs = TRUE,          # Probabilidades de classe
  summaryFunction = twoClassSummary,  # Metricas para classificacao binaria
  savePredictions = "final"   # Salvar predicoes
)

# -----------------------------------------------------------------------------
# 5. Treinar o modelo de Rede Neural (nnet)
# -----------------------------------------------------------------------------
set.seed(1)

cat("Iniciando treinamento da Rede Neural (nnet)...\n")

# Grid de hiperparametros para nnet
grid_nnet <- expand.grid(
  size = c(5, 10, 15, 20),    # Numero de neuronios na camada oculta
  decay = c(0.001, 0.01, 0.1) # Regularizacao (weight decay)
)

modelo_nnet <- train(
  x = X_treino_norm,
  y = y_treino,
  method = "nnet",            # Rede Neural (MLP com 1 camada oculta)
  trControl = ctrl,
  metric = "ROC",             # Otimizar pela AUC-ROC
  tuneGrid = grid_nnet,       # Grid de hiperparametros
  preProcess = NULL,          # Ja foi normalizado
  maxit = 500,                # Maximo de iteracoes
  trace = FALSE,              # Nao mostrar iteracoes
  MaxNWts = 10000             # Maximo de pesos (para datasets grandes)
)

cat("Treinamento concluido!\n\n")

# -----------------------------------------------------------------------------
# 6. Resultados do treinamento
# -----------------------------------------------------------------------------
print(modelo_nnet)

cat("\n--- Melhores parametros ---\n")
print(modelo_nnet$bestTune)

# -----------------------------------------------------------------------------
# 7. Avaliacao no conjunto de teste
# -----------------------------------------------------------------------------
predicoes <- predict(modelo_nnet, newdata = X_teste_norm)
prob_predicoes <- predict(modelo_nnet, newdata = X_teste_norm, type = "prob")

# Matriz de confusao
cat("\n--- Matriz de Confusao (Teste) ---\n")
conf_matrix <- confusionMatrix(predicoes, y_teste, positive = "Classe1")
print(conf_matrix)

# -----------------------------------------------------------------------------
# 8. Importancia das variaveis
# -----------------------------------------------------------------------------
cat("\n--- Importancia das Variaveis (Top 20) ---\n")
tryCatch({
  importancia <- varImp(modelo_nnet)
  print(importancia, top = 20)
}, error = function(e) {
  cat("Nota: Importancia de variaveis nao disponivel para este modelo\n")
})

# -----------------------------------------------------------------------------
# 9. Salvar o modelo e preprocessador
# -----------------------------------------------------------------------------
saveRDS(modelo_nnet, "modelo_nnet.rds")
saveRDS(preProc, "preprocessador_nnet.rds")
cat("\nModelo salvo em: modelo_nnet.rds\n")
cat("Preprocessador salvo em: preprocessador_nnet.rds\n")

# -----------------------------------------------------------------------------
# 10. Resumo das metricas
# -----------------------------------------------------------------------------
cat("\n========== RESUMO DO MODELO REDE NEURAL (nnet) ==========\n")
cat("Acuracia:", round(conf_matrix$overall["Accuracy"], 4), "\n")
cat("Sensibilidade:", round(conf_matrix$byClass["Sensitivity"], 4), "\n")
cat("Especificidade:", round(conf_matrix$byClass["Specificity"], 4), "\n")
cat("F1-Score:", round(conf_matrix$byClass["F1"], 4), "\n")
cat("Kappa:", round(conf_matrix$overall["Kappa"], 4), "\n")
cat("Melhores hiperparametros:\n")
cat("  - size (neuronios):", modelo_nnet$bestTune$size, "\n")
cat("  - decay (regularizacao):", modelo_nnet$bestTune$decay, "\n")
cat("==========================================================\n")
