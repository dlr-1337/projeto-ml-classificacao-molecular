# =============================================================================
# Treinamento com Random Forest usando caret
# =============================================================================

# Carregar bibliotecas necessarias
library(caret)
library(randomForest)

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
# 3. Configurar o treinamento com validacao cruzada
# -----------------------------------------------------------------------------
ctrl <- trainControl(
  method = "cv",              # Validacao cruzada
  number = 10,                # 10-fold CV
  classProbs = TRUE,          # Probabilidades de classe
  summaryFunction = twoClassSummary,  # Metricas para classificacao binaria
  savePredictions = "final"   # Salvar predicoes
)

# -----------------------------------------------------------------------------
# 4. Treinar o modelo Random Forest
# -----------------------------------------------------------------------------
set.seed(1)

cat("Iniciando treinamento do Random Forest...\n")

modelo_rf <- train(
  x = X_treino,
  y = y_treino,
  method = "rf",              # Random Forest
  trControl = ctrl,
  metric = "ROC",             # Otimizar pela AUC-ROC
  tuneLength = 5,             # Testar 5 valores de mtry
  importance = TRUE           # Calcular importancia das variaveis
)

cat("Treinamento concluido!\n\n")

# -----------------------------------------------------------------------------
# 5. Resultados do treinamento
# -----------------------------------------------------------------------------
print(modelo_rf)

cat("\n--- Melhores parametros ---\n")
print(modelo_rf$bestTune)

# -----------------------------------------------------------------------------
# 6. Avaliacao no conjunto de teste
# -----------------------------------------------------------------------------
predicoes <- predict(modelo_rf, newdata = X_teste)
prob_predicoes <- predict(modelo_rf, newdata = X_teste, type = "prob")

# Matriz de confusao
cat("\n--- Matriz de Confusao (Teste) ---\n")
conf_matrix <- confusionMatrix(predicoes, y_teste, positive = "Classe1")
print(conf_matrix)

# -----------------------------------------------------------------------------
# 7. Importancia das variaveis
# -----------------------------------------------------------------------------
cat("\n--- Top 20 Variaveis Mais Importantes ---\n")
importancia <- varImp(modelo_rf)
print(importancia, top = 20)

# -----------------------------------------------------------------------------
# 8. Salvar o modelo
# -----------------------------------------------------------------------------
saveRDS(modelo_rf, "modelo_random_forest.rds")
cat("\nModelo salvo em: modelo_random_forest.rds\n")

# -----------------------------------------------------------------------------
# 9. Resumo das metricas
# -----------------------------------------------------------------------------
cat("\n========== RESUMO DO MODELO RANDOM FOREST ==========\n")
cat("Acuracia:", round(conf_matrix$overall["Accuracy"], 4), "\n")
cat("Sensibilidade:", round(conf_matrix$byClass["Sensitivity"], 4), "\n")
cat("Especificidade:", round(conf_matrix$byClass["Specificity"], 4), "\n")
cat("F1-Score:", round(conf_matrix$byClass["F1"], 4), "\n")
cat("Kappa:", round(conf_matrix$overall["Kappa"], 4), "\n")
cat("=====================================================\n")
