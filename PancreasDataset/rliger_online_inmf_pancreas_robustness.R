# rliger_online_inmf_pancreas_robustness_canonical_v1.R
# Purpose: Generate Online iNMF comparison embeddings for pancreas robustness conditions.
# Author: Ariana Rahman (Arizona State University)
# -----------------------------------------------------------------------------
# Strict Online iNMF reruns for pancreas robustness perturbations.
# Reads manifests created by make_pancreas_robustness_manifests_canonical_v1.py
# and exports one Online iNMF embedding CSV per perturbation condition.
# -----------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(rliger)
  library(R.matlab)
  library(Matrix)
})

set.seed(0)

BASE_DIR <- Sys.getenv(
  "GENOREFINE_DATASET_ROOT",
  unset = getwd()
)
setwd(BASE_DIR)

DATA_FOLDER <- "./Dataset"
OUT_DIR <- "./Benchmark_Out"
ROBUST_DIR <- file.path(OUT_DIR, "Robustness_R_Embeddings")
dir.create(ROBUST_DIR, showWarnings = FALSE, recursive = TRUE)

DATA_FILES <- c("dataBaronX.mat", "dataMuraroX.mat", "dataScapleX.mat", "dataWangX.mat", "dataXinX.mat")
BATCH_NAMES <- c("Baron", "Muraro", "Segerstolpe", "Wang", "Xin")
stopifnot(length(DATA_FILES) == length(BATCH_NAMES))

CANONICAL_HVG_FILE <- file.path(OUT_DIR, "pancreas_hvg_canonical.txt")
REQUIRE_EXACT_CANONICAL_HVG_MATCH <- TRUE
K_FACTORS <- 30
LAMBDA <- 5
MIN_SHARED_GENES <- 500
MIN_BATCH_CELLS <- 5

get_data_key <- function(mat_data) {
  candidates <- grep("^data", names(mat_data), value = TRUE, ignore.case = TRUE)
  if (length(candidates) == 0) candidates <- grep("data", names(mat_data), value = TRUE, ignore.case = TRUE)
  if (length(candidates) == 0) return(NA_character_)
  candidates[1]
}

to_dgC <- function(x) as(Matrix(x, sparse = TRUE), "dgCMatrix")

read_canonical_hvgs <- function(path) {
  if (!file.exists(path)) stop("Canonical HVG file not found: ", path)
  x <- readLines(path, warn = FALSE)
  x <- trimws(x)
  x <- x[nzchar(x)]
  unique(as.character(x[!is.na(x)]))
}

get_liger_raw_list_compat <- function(lig) {
  out <- NULL
  try({ out <- rawData(lig) }, silent = TRUE)
  if (is.null(out)) try({ out <- getMatrix(lig, "rawData") }, silent = TRUE)
  if (is.null(out) && "raw.data" %in% slotNames(lig)) out <- lig@raw.data
  out
}

set_var_features_compat <- function(lig, genes) {
  genes <- unique(as.character(genes[!is.na(genes)]))
  ok <- FALSE
  try({ varFeatures(lig) <- genes; ok <- TRUE }, silent = TRUE)
  if (!ok && "var.genes" %in% slotNames(lig)) { lig@var.genes <- genes; ok <- TRUE }
  if (!ok) stop("Could not set variable features on liger object.")
  lig
}

get_var_features_compat <- function(lig) {
  out <- NULL
  try({ out <- varFeatures(lig) }, silent = TRUE)
  if (is.null(out) && "var.genes" %in% slotNames(lig)) out <- lig@var.genes
  unique(as.character(out[!is.na(out)]))
}

run_online_inmf_compat <- function(lig, k, lambda) {
  out <- NULL
  try({ out <- runOnlineINMF(lig, k = k, lambda = lambda) }, silent = TRUE)
  if (is.null(out)) try({ out <- online_iNMF(lig, k = k, lambda = lambda) }, silent = TRUE)
  if (is.null(out)) stop("Could not run online iNMF with either runOnlineINMF() or online_iNMF().")
  out
}

quantile_norm_compat <- function(lig) {
  out <- NULL
  try({ out <- quantileNorm(lig) }, silent = TRUE)
  if (is.null(out)) try({ out <- quantile_norm(lig) }, silent = TRUE)
  if (is.null(out)) stop("Could not run quantile normalization with either quantileNorm() or quantile_norm().")
  out
}

get_H_list_compat <- function(lig) {
  out <- NULL
  try({ out <- getMatrix(lig, "H") }, silent = TRUE)
  if (is.null(out) && "H" %in% slotNames(lig)) out <- lig@H
  if (is.null(out)) stop("Could not retrieve H matrices from liger object.")
  if (inherits(out, "list")) return(out)
  list(all = out)
}

coerce_H_to_cells_by_k <- function(h, expected_cells, batch_name) {
  h <- as.matrix(h)
  rn <- rownames(h)
  cn <- colnames(h)
  n_expected <- length(expected_cells)

  assign_factor_names <- function(m) {
    if (is.null(colnames(m)) || any(is.na(colnames(m))) || any(colnames(m) == "")) {
      colnames(m) <- paste0("Factor_", seq_len(ncol(m)))
    }
    m
  }

  if (!is.null(rn) && all(expected_cells %in% rn)) {
    h <- h[expected_cells, , drop = FALSE]
    return(assign_factor_names(h))
  }
  if (!is.null(cn) && all(expected_cells %in% cn)) {
    h <- t(h[, expected_cells, drop = FALSE])
    rownames(h) <- expected_cells
    return(assign_factor_names(h))
  }
  if (nrow(h) == n_expected) {
    rownames(h) <- expected_cells
    return(assign_factor_names(h))
  }
  if (ncol(h) == n_expected) {
    h <- t(h)
    rownames(h) <- expected_cells
    return(assign_factor_names(h))
  }
  stop("Could not align H matrix for batch '", batch_name, "'. Expected ", n_expected,
       " cells, got dim=", paste(dim(h), collapse = "x"))
}

load_pancreas_mats <- function() {
  mat_list <- list()
  for (i in seq_along(DATA_FILES)) {
    f <- DATA_FILES[i]
    batch_name <- BATCH_NAMES[i]
    path <- file.path(DATA_FOLDER, f)
    if (!file.exists(path)) stop("Missing file: ", path)
    mat_data <- readMat(path)
    key_name <- get_data_key(mat_data)
    if (is.na(key_name)) stop("Could not find a data key in: ", f)
    X <- mat_data[[key_name]]
    message(sprintf("  %s key=%s raw_dim=%s", f, key_name, paste(dim(X), collapse = "x")))
    counts <- to_dgC(t(X))
    rownames(counts) <- as.character(seq_len(nrow(counts)) - 1L)
    colnames(counts) <- paste0("Cell-", seq_len(ncol(counts)), "-Batch-", batch_name)
    mat_list[[batch_name]] <- counts
  }
  mat_list
}

subset_mats_to_manifest <- function(mat_list, manifest_path) {
  manifest <- read.csv(manifest_path, stringsAsFactors = FALSE)
  if (!"cell_id" %in% colnames(manifest)) stop("Manifest lacks cell_id column: ", manifest_path)
  keep_cells <- as.character(manifest$cell_id)

  cond <- list()
  for (b in BATCH_NAMES) {
    cells_b <- keep_cells[keep_cells %in% colnames(mat_list[[b]])]
    if (length(cells_b) >= MIN_BATCH_CELLS) {
      cond[[b]] <- mat_list[[b]][, cells_b, drop = FALSE]
    }
  }
  if (length(cond) < 2) stop("Need >=2 batches for condition: ", manifest_path)

  for (b in names(cond)) {
    nz <- Matrix::colSums(cond[[b]] != 0)
    if (any(nz == 0)) {
      message("Removing ", sum(nz == 0), " zero-count cells in ", b)
      cond[[b]] <- cond[[b]][, nz > 0, drop = FALSE]
    }
  }
  actual_cells <- keep_cells[keep_cells %in% unlist(lapply(cond, colnames), use.names = FALSE)]
  list(mat_list = cond, cells = actual_cells)
}

run_online_condition <- function(cond_mat_list, condition_cells, out_csv) {
  lig <- createLiger(cond_mat_list)

  common_genes_after <- Reduce(intersect, lapply(cond_mat_list, rownames))
  raw_list_after <- get_liger_raw_list_compat(lig)
  if (!is.null(raw_list_after)) {
    try({ common_genes_after <- Reduce(intersect, lapply(raw_list_after, rownames)) }, silent = TRUE)
  }
  common_genes_after <- unique(as.character(common_genes_after[!is.na(common_genes_after)]))
  selected_use <- selected_genes[selected_genes %in% common_genes_after]
  if (length(selected_use) < MIN_SHARED_GENES) stop("Too few selected genes after LIGER object creation: ", length(selected_use))

  lig <- set_var_features_compat(lig, selected_use)
  var_genes_now <- get_var_features_compat(lig)

  message("normalize() ...")
  lig <- normalize(lig)
  message("scaleNotCenter() ...")
  lig <- scaleNotCenter(lig)

  k_use <- min(K_FACTORS, length(var_genes_now) - 1)
  if (k_use < 10) stop("Not enough genes for k factors. varFeatures=", length(var_genes_now))
  message("Online iNMF(k=", k_use, ", lambda=", LAMBDA, ") ...")
  lig <- run_online_inmf_compat(lig, k = k_use, lambda = LAMBDA)

  message("Quantile normalization ...")
  lig <- quantile_norm_compat(lig)

  H_raw_list <- get_H_list_compat(lig)
  if ((is.null(names(H_raw_list)) || any(names(H_raw_list) == "")) && length(H_raw_list) == length(names(cond_mat_list))) {
    names(H_raw_list) <- names(cond_mat_list)
  }
  if (!all(names(cond_mat_list) %in% names(H_raw_list))) {
    stop("LIGER H missing batches: ", paste(setdiff(names(cond_mat_list), names(H_raw_list)), collapse = ", "))
  }

  H_list <- list()
  for (b in names(cond_mat_list)) {
    H_list[[b]] <- coerce_H_to_cells_by_k(H_raw_list[[b]], expected_cells = colnames(cond_mat_list[[b]]), batch_name = b)
  }
  H <- do.call(rbind, H_list[names(cond_mat_list)])
  missing <- setdiff(condition_cells, rownames(H))
  if (length(missing) > 0) stop("H missing condition cells. Examples: ", paste(head(missing, 10), collapse = ", "))
  H <- H[condition_cells, , drop = FALSE]

  write.csv(H, out_csv, quote = FALSE)
  message("Wrote ", out_csv, " rows=", nrow(H), " cols=", ncol(H))
}

message("Loading pancreas matrices...")
mat_list <- load_pancreas_mats()
shared_genes <- Reduce(intersect, lapply(mat_list, rownames))
mat_list <- lapply(mat_list, function(m) m[shared_genes, , drop = FALSE])

selected_genes <- read_canonical_hvgs(CANONICAL_HVG_FILE)
missing_canonical <- setdiff(selected_genes, shared_genes)
if (REQUIRE_EXACT_CANONICAL_HVG_MATCH && length(missing_canonical) > 0) {
  stop("Canonical HVGs missing from shared genes. Examples: ", paste(head(missing_canonical, 10), collapse = ", "))
}
selected_genes <- selected_genes[selected_genes %in% shared_genes]
if (length(selected_genes) < MIN_SHARED_GENES) stop("Too few selected canonical HVGs: ", length(selected_genes))
mat_list <- lapply(mat_list, function(m) m[selected_genes, , drop = FALSE])

manifest_files <- list.files(ROBUST_DIR, pattern = "^cells_.*\\.csv$", full.names = TRUE)
if (length(manifest_files) == 0) {
  stop("No cell manifests found in ", ROBUST_DIR, ". Run make_pancreas_robustness_manifests_canonical_v1.py first.")
}

for (manifest_path in manifest_files) {
  ctx <- sub("^cells_", "", tools::file_path_sans_ext(basename(manifest_path)))
  out_csv <- file.path(ROBUST_DIR, paste0("X_online_inmf_pancreas_", ctx, ".csv"))
  message("\n=== Online iNMF condition: ", ctx, " ===")
  cond <- subset_mats_to_manifest(mat_list, manifest_path)
  run_online_condition(cond$mat_list, cond$cells, out_csv)
}

message("✅ Strict Online iNMF robustness reruns complete.")
