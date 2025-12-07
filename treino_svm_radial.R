# =============================================================================
# Treinamento com SVM Radial usando caret
# =============================================================================

# Carregar bibliotecas necessarias
library(caret)
library(kernlab)

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
# 3. Pre-processamento (normalizacao - importante para SVM)
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
# 5. Treinar o modelo SVM Radial
# -----------------------------------------------------------------------------
set.seed(1)

cat("Iniciando treinamento do SVM Radial...\n")

# Grid de hiperparametros para SVM Radial
grid_svm <- expand.grid(
  sigma = c(0.001, 0.01, 0.1),   # Parametro do kernel RBF
  C = c(0.1, 1, 10, 100)         # Parametro de custo
)

modelo_svm <- train(
  x = X_treino_norm,
  y = y_treino,
  method = "svmRadial",       # SVM com kernel Radial (RBF)
  trControl = ctrl,
  metric = "ROC",             # Otimizar pela AUC-ROC
  tuneGrid = grid_svm,        # Grid de hiperparametros
  preProcess = NULL           # Ja foi normalizado
)

cat("Treinamento concluido!\n\n")

# -----------------------------------------------------------------------------
# 6. Resultados do treinamento
# -----------------------------------------------------------------------------
print(modelo_svm)

cat("\n--- Melhores parametros ---\n")
print(modelo_svm$bestTune)

# -----------------------------------------------------------------------------
# 7. Avaliacao no conjunto de teste
# -----------------------------------------------------------------------------
predicoes <- predict(modelo_svm, newdata = X_teste_norm)
prob_predicoes <- predict(modelo_svm, newdata = X_teste_norm, type = "prob")

# Matriz de confusao
cat("\n--- Matriz de Confusao (Teste) ---\n")
conf_matrix <- confusionMatrix(predicoes, y_teste, positive = "Classe1")
print(conf_matrix)

# -----------------------------------------------------------------------------
# 8. Importancia das variaveis
# -----------------------------------------------------------------------------
cat("\n--- Importancia das Variaveis (Top 20) ---\n")
tryCatch({
  importancia <- varImp(modelo_svm)
  print(importancia, top = 20)
}, error = function(e) {
  cat("Nota: Importancia de variaveis nao disponivel para este modelo SVM\n")
})

# -----------------------------------------------------------------------------
# 9. Salvar o modelo e preprocessador
# -----------------------------------------------------------------------------
saveRDS(modelo_svm, "modelo_svm_radial.rds")
saveRDS(preProc, "preprocessador_svm.rds")
cat("\nModelo salvo em: modelo_svm_radial.rds\n")
cat("Preprocessador salvo em: preprocessador_svm.rds\n")

# -----------------------------------------------------------------------------
# 10. Resumo das metricas
# -----------------------------------------------------------------------------
cat("\n========== RESUMO DO MODELO SVM RADIAL ==========\n")
cat("Acuracia:", round(conf_matrix$overall["Accuracy"], 4), "\n")
cat("Sensibilidade:", round(conf_matrix$byClass["Sensitivity"], 4), "\n")
cat("Especificidade:", round(conf_matrix$byClass["Specificity"], 4), "\n")
cat("F1-Score:", round(conf_matrix$byClass["F1"], 4), "\n")
cat("Kappa:", round(conf_matrix$overall["Kappa"], 4), "\n")
cat("Melhores hiperparametros:\n")
cat("  - sigma:", modelo_svm$bestTune$sigma, "\n")
cat("  - C:", modelo_svm$bestTune$C, "\n")
cat("=================================================\n")
