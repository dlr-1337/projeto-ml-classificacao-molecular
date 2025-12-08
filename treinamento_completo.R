# =============================================================================
# Treinamento Completo - Replicacao do Artigo InterDIA
# =============================================================================
# Metodologia baseada no artigo InterDIA:
# - Easy Ensemble Classifier (EEC) simulado via Balanced Bagging
# - Selecao de features via RFE e RF Importance (equivalente ao GA de 65 features)
# - Remocao de correlacoes > 0.9
# - Foco em AUC e MCC para dados desbalanceados (ratio 1:3)
# =============================================================================

library(caret)
library(pROC)

# Ativar paralelizacao, quando disponivel, para acelerar os treinos mais longos
if (requireNamespace("doParallel", quietly = TRUE)) {
  num_cores <- parallel::detectCores(logical = TRUE)
  num_workers <- max(1, num_cores - 1, na.rm = TRUE)
  cl <- parallel::makeCluster(num_workers)
  doParallel::registerDoParallel(cl)
  on.exit(parallel::stopCluster(cl), add = TRUE)
  cat("Paralelizacao ativada com", num_workers, "nucleos.\n\n")
} else {
  cat("Pacote doParallel nao encontrado; prosseguindo sem paralelizacao.\n\n")
}

# =============================================================================
# FUNCOES AUXILIARES
# =============================================================================

# Funcao para calcular Matthews Correlation Coefficient
calcularMCC <- function(conf_matrix) {
  TP <- conf_matrix$table[2, 2]
  TN <- conf_matrix$table[1, 1]
  FP <- conf_matrix$table[2, 1]
  FN <- conf_matrix$table[1, 2]

  numerador <- (TP * TN) - (FP * FN)
  denominador <- sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

  if (denominador == 0) return(0)
  return(numerador / denominador)
}

# Funcao para escolher limiar (Youden) a partir de predicoes com probabilidade
calcularLimiarYouden <- function(pred_df) {
  if (is.null(pred_df) || nrow(pred_df) == 0 || !"Classe1" %in% names(pred_df)) {
    return(0.5)
  }
  roc_obj <- pROC::roc(pred_df$obs, pred_df$Classe1, levels = c("Classe0", "Classe1"), direction = "<")
  coords <- pROC::coords(roc_obj, "best", ret = "threshold", best.method = "youden")
  as.numeric(coords)
}

# Funcao para escolher limiar que maximiza acuracia (baseado em predicoes OOF)
calcularLimiarAcuracia <- function(pred_df) {
  if (is.null(pred_df) || nrow(pred_df) == 0 || !"Classe1" %in% names(pred_df)) {
    return(0.5)
  }
  roc_obj <- pROC::roc(pred_df$obs, pred_df$Classe1, levels = c("Classe0", "Classe1"), direction = "<")
  coords <- pROC::coords(roc_obj, x = "all", ret = c("threshold", "accuracy"), transpose = FALSE)
  if (is.null(coords) || nrow(coords) == 0) return(0.5)
  coords <- coords[!is.na(coords$accuracy), ]
  if (nrow(coords) == 0) return(0.5)
  best_idx <- which.max(coords$accuracy)
  as.numeric(coords$threshold[best_idx])
}

# Funcao para remover features altamente correlacionadas
removerCorrelacionadas <- function(X, threshold = 0.9) {
  # Calcular matriz de correlacao
  cor_matrix <- cor(X, use = "pairwise.complete.obs")
  cor_matrix[is.na(cor_matrix)] <- 0

  # Encontrar pares altamente correlacionados
  cor_alta <- findCorrelation(cor_matrix, cutoff = threshold, verbose = FALSE)

  if (length(cor_alta) > 0) {
    cat("Removendo", length(cor_alta), "features com correlacao >", threshold, "\n")
    return(X[, -cor_alta])
  }
  return(X)
}

# =============================================================================
# 1. CARREGAR E PREPARAR OS DADOS
# =============================================================================

cat("=============================================================\n")
cat("         CARREGANDO E PREPARANDO OS DADOS\n")
cat("=============================================================\n\n")

dados_treino <- read.csv("dataset/DIA_trainingset_RDKit_descriptors.csv")
dados_teste <- read.csv("dataset/DIA_testset_RDKit_descriptors.csv")

# Remover coluna SMILES
dados_treino <- dados_treino[, !names(dados_treino) %in% c("SMILES")]
dados_teste <- dados_teste[, !names(dados_teste) %in% c("SMILES")]

# Converter Label para fator
dados_treino$Label <- as.factor(dados_treino$Label)
dados_teste$Label <- as.factor(dados_teste$Label)
levels(dados_treino$Label) <- c("Classe0", "Classe1")
levels(dados_teste$Label) <- c("Classe0", "Classe1")

# Separar features e target
X_treino <- dados_treino[, -which(names(dados_treino) == "Label")]
y_treino <- dados_treino$Label
X_teste <- dados_teste[, -which(names(dados_teste) == "Label")]
y_teste <- dados_teste$Label

cat("Dados originais:\n")
cat("  Treino:", nrow(X_treino), "amostras x", ncol(X_treino), "features\n")
cat("  Teste:", nrow(X_teste), "amostras x", ncol(X_teste), "features\n")
cat("  Distribuicao treino - Classe0:", sum(y_treino == "Classe0"),
    "| Classe1:", sum(y_treino == "Classe1"), "\n")
cat("  Ratio de desbalanceamento:",
    round(sum(y_treino == "Classe0") / sum(y_treino == "Classe1"), 2), ":1\n\n")

# =============================================================================
# 2. PRE-PROCESSAMENTO (conforme artigo)
# =============================================================================

cat("=============================================================\n")
cat("         PRE-PROCESSAMENTO\n")
cat("=============================================================\n\n")

# 2.1 Remover colunas com variancia zero ou near-zero
nzv <- nearZeroVar(X_treino)
if (length(nzv) > 0) {
  cat("Removendo", length(nzv), "features com near-zero variance\n")
  X_treino <- X_treino[, -nzv]
  X_teste <- X_teste[, names(X_treino)]
}

# 2.2 Remover colunas com NA
cols_na <- colSums(is.na(X_treino)) > 0
if (sum(cols_na) > 0) {
  cat("Removendo", sum(cols_na), "features com valores NA\n")
  X_treino <- X_treino[, !cols_na]
  X_teste <- X_teste[, names(X_treino)]
}

# 2.3 Remover features altamente correlacionadas (> 0.9) - CONFORME ARTIGO
X_treino <- removerCorrelacionadas(X_treino, threshold = 0.9)
X_teste <- X_teste[, names(X_treino)]

cat("Apos pre-processamento:", ncol(X_treino), "features restantes\n\n")

# 2.4 Normalizacao (center + scale)
preProc <- preProcess(X_treino, method = c("center", "scale"))
X_treino_norm <- predict(preProc, X_treino)
X_teste_norm <- predict(preProc, X_teste)

# =============================================================================
# 3. SELECAO DE FEATURES (equivalente ao GA de 65 features do artigo)
# =============================================================================

cat("=============================================================\n")
cat("         SELECAO DE FEATURES (RFE + RF Importance)\n")
cat("=============================================================\n\n")

# 3.1 Usar Random Forest para selecionar top features
set.seed(1)
cat("Treinando RF inicial para importancia de variaveis...\n")

ctrl_rf_init <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

# Treinar RF inicial com downsampling para lidar com desbalanceamento
rf_inicial <- train(
  x = X_treino_norm,
  y = y_treino,
  method = "rf",
  trControl = ctrl_rf_init,
  metric = "ROC",
  tuneLength = 3,
  importance = TRUE,
  ntree = 500
)

# Extrair importancia das variaveis
importancia_raw <- varImp(rf_inicial)$importance
importancia <- data.frame(feature = rownames(importancia_raw))

# Para classificacao binaria, varImp pode retornar colunas por classe ou "Overall"
if ("Overall" %in% colnames(importancia_raw)) {
  importancia$Overall <- importancia_raw$Overall
} else {
  # Se nao tiver Overall, calcular media das colunas (classes)
  importancia$Overall <- rowMeans(importancia_raw)
}

# Ordenar por importancia decrescente
importancia <- importancia[order(-importancia$Overall), ]
rownames(importancia) <- NULL

# Selecionar top 65 features (conforme artigo - RDKit GA 65)
N_FEATURES <- min(65, nrow(importancia))  # Garantir que nao excede o disponivel
top_features <- importancia$feature[1:N_FEATURES]

cat("Top", N_FEATURES, "features selecionadas (equivalente ao GA do artigo):\n")
cat(paste(head(top_features, 20), collapse = ", "), "...\n\n")

# Aplicar selecao
X_treino_sel <- X_treino_norm[, top_features]
X_teste_sel <- X_teste_norm[, top_features]

cat("Dimensoes apos selecao:", ncol(X_treino_sel), "features\n\n")

# =============================================================================
# 4. CONFIGURACAO DO TREINAMENTO COM TRATAMENTO DE DESBALANCEAMENTO
# =============================================================================

# Controle com downsampling (simula parte do EEC)
ctrl_balanced <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final",
  sampling = "down"  # Random Under-Sampling em cada fold
)

# Controle especifico com verbose para nao parecer que travou em modelos mais lentos
ctrl_balanced_svm <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final",
  sampling = "down",
  verboseIter = TRUE,
  allowParallel = TRUE
)

# Controle com SMOTE (alternativa)
ctrl_smote <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final",
  sampling = "smote"
)

# Controle padrao (sem sampling)
ctrl_standard <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

# Lista para armazenar resultados
resultados <- list()

# =============================================================================
# 5. EASY ENSEMBLE CLASSIFIER (EEC) - Implementacao em R
# =============================================================================
# O EEC do Python cria multiplos subconjuntos balanceados e treina um
# classificador em cada. Simulamos isso com Balanced Bagging.
# =============================================================================

cat("=============================================================\n")
cat("    EASY ENSEMBLE CLASSIFIER (Balanced Bagging RF)\n")
cat("=============================================================\n\n")

set.seed(1)

# Parametros do EEC
n_subsamples <- 20  # Numero de subconjuntos balanceados (mais bagging para reduzir variancia)
n_minority <- sum(y_treino == "Classe1")

# Criar lista para guardar modelos do ensemble
modelos_eec <- list()
predicoes_eec_prob <- matrix(0, nrow = nrow(X_teste_sel), ncol = n_subsamples)

cat("Treinando", n_subsamples, "modelos com undersampling balanceado...\n")

for (i in 1:n_subsamples) {
  cat("  Modelo", i, "/", n_subsamples, "\n")

  # Criar subset balanceado (undersample maioria)
  idx_minority <- which(y_treino == "Classe1")
  idx_majority <- which(y_treino == "Classe0")
  idx_majority_sample <- sample(idx_majority, n_minority, replace = FALSE)

  idx_balanced <- c(idx_minority, idx_majority_sample)
  X_balanced <- X_treino_sel[idx_balanced, ]
  y_balanced <- y_treino[idx_balanced]

  # Treinar modelo
  modelo_i <- train(
    x = X_balanced,
    y = y_balanced,
    method = "rf",
    trControl = trainControl(
      method = "cv",
      number = 5,
      classProbs = TRUE,
      summaryFunction = twoClassSummary,
      savePredictions = "final"
    ),
    metric = "ROC",
    tuneLength = 3,
    ntree = 500
  )

  modelos_eec[[i]] <- modelo_i
  predicoes_eec_prob[, i] <- predict(modelo_i, X_teste_sel, type = "prob")[, "Classe1"]
}

# Ajustar limiar via acuracia (e fallback Youden) em predicoes OOF dos modelos do ensemble
predicoes_oof <- do.call(
  rbind,
  lapply(modelos_eec, function(m) {
    if (!is.null(m$pred)) m$pred[, c("obs", "Classe1")] else NULL
  })
)
limiar_eec_acc <- calcularLimiarAcuracia(predicoes_oof)
limiar_eec <- ifelse(is.na(limiar_eec_acc), calcularLimiarYouden(predicoes_oof), limiar_eec_acc)
cat("Limiar otimo (Acuracia) para EEC:", round(limiar_eec, 4), "\n")

# Agregar predicoes (media das probabilidades) e aplicar limiar otimo
prob_media_eec <- rowMeans(predicoes_eec_prob)
predicoes_eec <- ifelse(prob_media_eec > limiar_eec, "Classe1", "Classe0")
predicoes_eec <- factor(predicoes_eec, levels = c("Classe0", "Classe1"))

# Avaliar EEC
conf_matrix_eec <- confusionMatrix(predicoes_eec, y_teste, positive = "Classe1")
mcc_eec <- calcularMCC(conf_matrix_eec)

# Calcular AUC
library(pROC)
roc_eec <- roc(y_teste, prob_media_eec, levels = c("Classe0", "Classe1"), direction = "<")
auc_eec <- auc(roc_eec)

resultados$eec <- list(
  conf_matrix = conf_matrix_eec,
  acuracia = conf_matrix_eec$overall["Accuracy"],
  sensibilidade = conf_matrix_eec$byClass["Sensitivity"],
  especificidade = conf_matrix_eec$byClass["Specificity"],
  f1 = conf_matrix_eec$byClass["F1"],
  mcc = mcc_eec,
  auc = as.numeric(auc_eec)
)

cat("\n--- Resultados Easy Ensemble Classifier ---\n")
print(conf_matrix_eec)
cat("MCC:", round(mcc_eec, 4), "\n")
cat("AUC:", round(as.numeric(auc_eec), 4), "\n")

# Salvar EEC
saveRDS(modelos_eec, "modelo_eec.rds")

# =============================================================================
# 6. RANDOM FOREST COM DOWNSAMPLING
# =============================================================================

cat("\n=============================================================\n")
cat("         RANDOM FOREST (com Downsampling)\n")
cat("=============================================================\n\n")

set.seed(1)

grid_rf <- expand.grid(mtry = c(5, 8, 11, 15, 20))

modelo_rf <- train(
  x = X_treino_sel,
  y = y_treino,
  method = "rf",
  trControl = ctrl_balanced,
  metric = "ROC",
  tuneGrid = grid_rf,
  ntree = 800,
  importance = TRUE
)

cat("Treinamento RF concluido!\n")
print(modelo_rf)

# Avaliacao
limiar_rf_acc <- calcularLimiarAcuracia(modelo_rf$pred)
limiar_rf <- ifelse(is.na(limiar_rf_acc), calcularLimiarYouden(modelo_rf$pred), limiar_rf_acc)
cat("Limiar otimo (Acuracia) para RF:", round(limiar_rf, 4), "\n")
prob_rf <- predict(modelo_rf, X_teste_sel, type = "prob")[, "Classe1"]
predicoes_rf <- ifelse(prob_rf > limiar_rf, "Classe1", "Classe0")
predicoes_rf <- factor(predicoes_rf, levels = c("Classe0", "Classe1"))
conf_matrix_rf <- confusionMatrix(predicoes_rf, y_teste, positive = "Classe1")
mcc_rf <- calcularMCC(conf_matrix_rf)
roc_rf <- roc(y_teste, prob_rf, levels = c("Classe0", "Classe1"), direction = "<")
auc_rf <- auc(roc_rf)

resultados$rf <- list(
  modelo = modelo_rf,
  conf_matrix = conf_matrix_rf,
  acuracia = conf_matrix_rf$overall["Accuracy"],
  sensibilidade = conf_matrix_rf$byClass["Sensitivity"],
  especificidade = conf_matrix_rf$byClass["Specificity"],
  f1 = conf_matrix_rf$byClass["F1"],
  mcc = mcc_rf,
  auc = as.numeric(auc_rf)
)

cat("\n--- Matriz de Confusao (RF Balanced) ---\n")
print(conf_matrix_rf)
cat("MCC:", round(mcc_rf, 4), "\n")
cat("AUC:", round(as.numeric(auc_rf), 4), "\n")

saveRDS(modelo_rf, "modelo_rf_balanced.rds")

# =============================================================================
# 7. SVM RADIAL COM DOWNSAMPLING
# =============================================================================

cat("\n=============================================================\n")
cat("         SVM RADIAL (com Downsampling)\n")
cat("=============================================================\n\n")

set.seed(1)

grid_svm <- expand.grid(
  sigma = c(0.001, 0.005, 0.01, 0.05),
  C = c(0.25, 1, 4, 16)
)

cat(
  "Treinando SVM Radial com", nrow(grid_svm), "combinacoes e",
  ctrl_balanced_svm$number * ctrl_balanced_svm$repeats, "reamostragens.\n",
  "Esta etapa pode levar alguns minutos; verboseIter mostrara o andamento.\n"
)

modelo_svm <- train(
  x = X_treino_sel,
  y = y_treino,
  method = "svmRadial",
  trControl = ctrl_balanced_svm,
  metric = "ROC",
  tuneGrid = grid_svm
)

cat("Treinamento SVM concluido!\n")
print(modelo_svm)

# Avaliacao
limiar_svm_acc <- calcularLimiarAcuracia(modelo_svm$pred)
limiar_svm <- ifelse(is.na(limiar_svm_acc), calcularLimiarYouden(modelo_svm$pred), limiar_svm_acc)
cat("Limiar otimo (Acuracia) para SVM:", round(limiar_svm, 4), "\n")
prob_svm <- predict(modelo_svm, X_teste_sel, type = "prob")[, "Classe1"]
predicoes_svm <- ifelse(prob_svm > limiar_svm, "Classe1", "Classe0")
predicoes_svm <- factor(predicoes_svm, levels = c("Classe0", "Classe1"))
conf_matrix_svm <- confusionMatrix(predicoes_svm, y_teste, positive = "Classe1")
mcc_svm <- calcularMCC(conf_matrix_svm)
roc_svm <- roc(y_teste, prob_svm, levels = c("Classe0", "Classe1"), direction = "<")
auc_svm <- auc(roc_svm)

resultados$svm <- list(
  modelo = modelo_svm,
  conf_matrix = conf_matrix_svm,
  acuracia = conf_matrix_svm$overall["Accuracy"],
  sensibilidade = conf_matrix_svm$byClass["Sensitivity"],
  especificidade = conf_matrix_svm$byClass["Specificity"],
  f1 = conf_matrix_svm$byClass["F1"],
  mcc = mcc_svm,
  auc = as.numeric(auc_svm)
)

cat("\n--- Matriz de Confusao (SVM Balanced) ---\n")
print(conf_matrix_svm)
cat("MCC:", round(mcc_svm, 4), "\n")
cat("AUC:", round(as.numeric(auc_svm), 4), "\n")

saveRDS(modelo_svm, "modelo_svm_balanced.rds")

# =============================================================================
# 8. GRADIENT BOOSTING (XGBoost-like via gbm)
# =============================================================================

cat("\n=============================================================\n")
cat("         GRADIENT BOOSTING (com Downsampling)\n")
cat("=============================================================\n\n")

set.seed(1)

grid_gbm <- expand.grid(
  n.trees = c(100, 200, 300),
  interaction.depth = c(3, 5, 7),
  shrinkage = c(0.01, 0.1),
  n.minobsinnode = 10
)

modelo_gbm <- train(
  x = X_treino_sel,
  y = y_treino,
  method = "gbm",
  trControl = ctrl_balanced,
  metric = "ROC",
  tuneGrid = grid_gbm,
  verbose = FALSE
)

cat("Treinamento GBM concluido!\n")
print(modelo_gbm)

# Avaliacao
limiar_gbm_acc <- calcularLimiarAcuracia(modelo_gbm$pred)
limiar_gbm <- ifelse(is.na(limiar_gbm_acc), calcularLimiarYouden(modelo_gbm$pred), limiar_gbm_acc)
cat("Limiar otimo (Acuracia) para GBM:", round(limiar_gbm, 4), "\n")
prob_gbm <- predict(modelo_gbm, X_teste_sel, type = "prob")[, "Classe1"]
predicoes_gbm <- ifelse(prob_gbm > limiar_gbm, "Classe1", "Classe0")
predicoes_gbm <- factor(predicoes_gbm, levels = c("Classe0", "Classe1"))
conf_matrix_gbm <- confusionMatrix(predicoes_gbm, y_teste, positive = "Classe1")
mcc_gbm <- calcularMCC(conf_matrix_gbm)
roc_gbm <- roc(y_teste, prob_gbm, levels = c("Classe0", "Classe1"), direction = "<")
auc_gbm <- auc(roc_gbm)

resultados$gbm <- list(
  modelo = modelo_gbm,
  conf_matrix = conf_matrix_gbm,
  acuracia = conf_matrix_gbm$overall["Accuracy"],
  sensibilidade = conf_matrix_gbm$byClass["Sensitivity"],
  especificidade = conf_matrix_gbm$byClass["Specificity"],
  f1 = conf_matrix_gbm$byClass["F1"],
  mcc = mcc_gbm,
  auc = as.numeric(auc_gbm)
)

cat("\n--- Matriz de Confusao (GBM Balanced) ---\n")
print(conf_matrix_gbm)
cat("MCC:", round(mcc_gbm, 4), "\n")
cat("AUC:", round(as.numeric(auc_gbm), 4), "\n")

saveRDS(modelo_gbm, "modelo_gbm_balanced.rds")

# =============================================================================
# 9. REDE NEURAL (nnet) COM DOWNSAMPLING
# =============================================================================

cat("\n=============================================================\n")
cat("         REDE NEURAL (com Downsampling)\n")
cat("=============================================================\n\n")

set.seed(1)

grid_nnet <- expand.grid(
  size = c(5, 10, 15, 20),
  decay = c(0.001, 0.01, 0.1)
)

modelo_nnet <- train(
  x = X_treino_sel,
  y = y_treino,
  method = "nnet",
  trControl = ctrl_balanced,
  metric = "ROC",
  tuneGrid = grid_nnet,
  maxit = 300,
  trace = FALSE,
  MaxNWts = 5000
)

cat("Treinamento Rede Neural concluido!\n")
print(modelo_nnet)

# Avaliacao
limiar_nnet_acc <- calcularLimiarAcuracia(modelo_nnet$pred)
limiar_nnet <- ifelse(is.na(limiar_nnet_acc), calcularLimiarYouden(modelo_nnet$pred), limiar_nnet_acc)
cat("Limiar otimo (Acuracia) para NNet:", round(limiar_nnet, 4), "\n")
prob_nnet <- predict(modelo_nnet, X_teste_sel, type = "prob")[, "Classe1"]
predicoes_nnet <- ifelse(prob_nnet > limiar_nnet, "Classe1", "Classe0")
predicoes_nnet <- factor(predicoes_nnet, levels = c("Classe0", "Classe1"))
conf_matrix_nnet <- confusionMatrix(predicoes_nnet, y_teste, positive = "Classe1")
mcc_nnet <- calcularMCC(conf_matrix_nnet)
roc_nnet <- roc(y_teste, prob_nnet, levels = c("Classe0", "Classe1"), direction = "<")
auc_nnet <- auc(roc_nnet)

resultados$nnet <- list(
  modelo = modelo_nnet,
  conf_matrix = conf_matrix_nnet,
  acuracia = conf_matrix_nnet$overall["Accuracy"],
  sensibilidade = conf_matrix_nnet$byClass["Sensitivity"],
  especificidade = conf_matrix_nnet$byClass["Specificity"],
  f1 = conf_matrix_nnet$byClass["F1"],
  mcc = mcc_nnet,
  auc = as.numeric(auc_nnet)
)

cat("\n--- Matriz de Confusao (Rede Neural Balanced) ---\n")
print(conf_matrix_nnet)
cat("MCC:", round(mcc_nnet, 4), "\n")
cat("AUC:", round(as.numeric(auc_nnet), 4), "\n")

saveRDS(modelo_nnet, "modelo_nnet_balanced.rds")

# =============================================================================
# 10. COMPARACAO FINAL DOS MODELOS
# =============================================================================

cat("\n=============================================================\n")
cat("         COMPARACAO FINAL DOS MODELOS\n")
cat("=============================================================\n\n")

comparacao <- data.frame(
  Modelo = c("Easy Ensemble (EEC)", "Random Forest", "SVM Radial",
             "Gradient Boosting", "Rede Neural"),
  Acuracia = c(
    round(resultados$eec$acuracia, 4),
    round(resultados$rf$acuracia, 4),
    round(resultados$svm$acuracia, 4),
    round(resultados$gbm$acuracia, 4),
    round(resultados$nnet$acuracia, 4)
  ),
  Sensibilidade = c(
    round(resultados$eec$sensibilidade, 4),
    round(resultados$rf$sensibilidade, 4),
    round(resultados$svm$sensibilidade, 4),
    round(resultados$gbm$sensibilidade, 4),
    round(resultados$nnet$sensibilidade, 4)
  ),
  Especificidade = c(
    round(resultados$eec$especificidade, 4),
    round(resultados$rf$especificidade, 4),
    round(resultados$svm$especificidade, 4),
    round(resultados$gbm$especificidade, 4),
    round(resultados$nnet$especificidade, 4)
  ),
  F1_Score = c(
    round(resultados$eec$f1, 4),
    round(resultados$rf$f1, 4),
    round(resultados$svm$f1, 4),
    round(resultados$gbm$f1, 4),
    round(resultados$nnet$f1, 4)
  ),
  MCC = c(
    round(resultados$eec$mcc, 4),
    round(resultados$rf$mcc, 4),
    round(resultados$svm$mcc, 4),
    round(resultados$gbm$mcc, 4),
    round(resultados$nnet$mcc, 4)
  ),
  AUC = c(
    round(resultados$eec$auc, 4),
    round(resultados$rf$auc, 4),
    round(resultados$svm$auc, 4),
    round(resultados$gbm$auc, 4),
    round(resultados$nnet$auc, 4)
  )
)

print(comparacao)

# Identificar melhores modelos
cat("\n--- MELHORES MODELOS ---\n")
cat("Por Acuracia:", comparacao$Modelo[which.max(comparacao$Acuracia)],
    "(", max(comparacao$Acuracia), ")\n")
cat("Por MCC:", comparacao$Modelo[which.max(comparacao$MCC)],
    "(", max(comparacao$MCC), ")\n")
cat("Por AUC:", comparacao$Modelo[which.max(comparacao$AUC)],
    "(", max(comparacao$AUC), ")\n")

# Alvo do artigo
cat("\n--- ALVO DO ARTIGO InterDIA ---\n")
cat("Acuracia alvo: 0.85\n")
cat("AUC alvo: ~0.89\n")

# =============================================================================
# 11. CURVAS ROC
# =============================================================================

cat("\n=============================================================\n")
cat("         GERANDO CURVAS ROC\n")
cat("=============================================================\n\n")

png("curvas_roc.png", width = 800, height = 600)
plot(roc_eec, col = "red", main = "Curvas ROC - Comparacao de Modelos",
     legacy.axes = TRUE)
plot(roc_rf, col = "blue", add = TRUE)
plot(roc_svm, col = "green", add = TRUE)
plot(roc_gbm, col = "purple", add = TRUE)
plot(roc_nnet, col = "orange", add = TRUE)
legend("bottomright",
       legend = c(
         paste0("EEC (AUC=", round(auc_eec, 3), ")"),
         paste0("RF (AUC=", round(auc_rf, 3), ")"),
         paste0("SVM (AUC=", round(auc_svm, 3), ")"),
         paste0("GBM (AUC=", round(auc_gbm, 3), ")"),
         paste0("NNet (AUC=", round(auc_nnet, 3), ")")
       ),
       col = c("red", "blue", "green", "purple", "orange"),
       lwd = 2)
dev.off()
cat("Curvas ROC salvas em: curvas_roc.png\n")

# =============================================================================
# 12. IMPORTANCIA DAS FEATURES (Top 20)
# =============================================================================

cat("\n=============================================================\n")
cat("         TOP 20 FEATURES MAIS IMPORTANTES\n")
cat("=============================================================\n\n")

print(head(importancia[, c("feature", "Overall")], 20))

# =============================================================================
# 13. SALVAR RESULTADOS
# =============================================================================

saveRDS(resultados, "resultados_completos.rds")
saveRDS(comparacao, "comparacao_modelos.rds")
saveRDS(preProc, "preprocessador.rds")
saveRDS(top_features, "features_selecionadas.rds")

cat("\n=============================================================\n")
cat("         ARQUIVOS SALVOS\n")
cat("=============================================================\n")
cat("- modelo_eec.rds (Easy Ensemble Classifier)\n")
cat("- modelo_rf_balanced.rds\n")
cat("- modelo_svm_balanced.rds\n")
cat("- modelo_gbm_balanced.rds\n")
cat("- modelo_nnet_balanced.rds\n")
cat("- preprocessador.rds\n")
cat("- features_selecionadas.rds (top 65 features)\n")
cat("- resultados_completos.rds\n")
cat("- comparacao_modelos.rds\n")
cat("- curvas_roc.png\n")
cat("=============================================================\n")
cat("\nTreinamento completo finalizado!\n")
