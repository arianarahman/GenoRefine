# rliger_online_inmf_pancreas_v14.R
# Purpose: Generate the Online iNMF comparison embedding for the five-study pancreas benchmark.
# Author: Ariana Rahman (Arizona State University)
# Online iNMF baseline for 5-pancreas .mat benchmark
# Exports H (cells x k) to CSV with rownames matching Python v11 obs_names:
#   "Cell-<i>-Batch-<Baron|Muraro|Segerstolpe|Wang|Xin>"
#
# v14 CHANGE FROM v13:
#   - updates BASE_DIR to the current OneDrive project path used in recent runs
#
# v13 CHANGE FROM v12:
#   - fixes H export when rliger stores H as factors x cells (k x n_cells)
#   - adds dimension-based fallback alignment when cell names are absent or non-canonical
#   - keeps v12 compatibility wrappers and frozen canonical HVG logic
#
# PIPELINE INTENT REMAINS THE SAME:
#   - online iNMF baseline for pancreas
#   - export H (not H.norm) aligned to Python obs_names
#   - same loading / naming / ordering logic

suppressPackageStartupMessages({
  library(rliger)
  library(R.matlab)
  library(Matrix)
})

set.seed(0)

# ---------------------------
# Config
# ---------------------------
BASE_DIR <- Sys.getenv("GENOREFINE_DATASET_ROOT", unset = getwd())
setwd(BASE_DIR)

DATA_FOLDER   <- "./Dataset"
OUTPUT_FOLDER <- "./Benchmark_Out"
dir.create(OUTPUT_FOLDER, showWarnings = FALSE, recursive = TRUE)

DATA_FILES <- c(
  "dataBaronX.mat",
  "dataMuraroX.mat",
  "dataScapleX.mat",  # token "Scaple" -> displayed as "Segerstolpe"
  "dataWangX.mat",
  "dataXinX.mat"
)

# Batch names in SAME ORDER as DATA_FILES
BATCH_NAMES <- c("Baron", "Muraro", "Segerstolpe", "Wang", "Xin")
stopifnot(length(DATA_FILES) == length(BATCH_NAMES))

OUT_CSV <- file.path(OUTPUT_FOLDER, "X_online_inmf_pancreas.csv")

# Frozen canonical HVG file (v11 default path)
USE_CANONICAL_HVG_FILE <- TRUE
CANONICAL_HVG_FILE <- file.path(OUTPUT_FOLDER, "pancreas_hvg_canonical.txt")
FALLBACK_TO_V10_FAIR_HVGS_IF_MISSING <- FALSE
REQUIRE_EXACT_CANONICAL_HVG_MATCH <- TRUE

# LIGER params
K_FACTORS <- 30
LAMBDA <- 5
MIN_SHARED_GENES <- 500

# Optional fallback params from v10
N_VAR_GENES_PER_BATCH <- 2000
N_VAR_GENES_FINAL <- 3000

# ---------------------------
# Helpers
# ---------------------------
get_data_key <- function(mat_data) {
  candidates <- grep("^data", names(mat_data), value = TRUE, ignore.case = TRUE)
  if (length(candidates) == 0) {
    candidates <- grep("data", names(mat_data), value = TRUE, ignore.case = TRUE)
  }
  if (length(candidates) == 0) return(NA_character_)
  candidates[1]
}

to_dgC <- function(x) {
  as(Matrix(x, sparse = TRUE), "dgCMatrix")
}

read_canonical_hvgs <- function(path) {
  genes <- readLines(path, warn = FALSE)
  genes <- trimws(genes)
  genes <- genes[nzchar(genes)]
  genes <- unique(as.character(genes))
  genes <- genes[!is.na(genes)]
  genes
}

select_fair_hvgs_v10 <- function(mat_list, n_per_batch, n_final) {
  message("Selecting HVGs per batch (v10 fair union fallback)...")

  hvg_lists <- list()

  for (b in names(mat_list)) {
    message("  - batch: ", b)

    tmp <- createLiger(setNames(list(mat_list[[b]]), b))
    tmp <- normalize(tmp)
    tmp <- selectGenes(tmp, var.thresh = 0, num.genes = n_per_batch)

    hvgs_b <- NULL
    try({
      hvgs_b <- varFeatures(tmp)
    }, silent = TRUE)
    if (is.null(hvgs_b) && "var.genes" %in% slotNames(tmp)) {
      hvgs_b <- tmp@var.genes
    }
    hvgs_b <- unique(as.character(hvgs_b))
    hvgs_b <- hvgs_b[!is.na(hvgs_b)]

    hvg_lists[[b]] <- hvgs_b
  }

  all_hvgs <- unlist(hvg_lists, use.names = FALSE)
  hvg_freq <- sort(table(all_hvgs), decreasing = TRUE)

  selected_genes <- names(hvg_freq)
  if (length(selected_genes) > n_final) {
    selected_genes <- selected_genes[seq_len(n_final)]
  }

  selected_genes <- unique(as.character(selected_genes))
  selected_genes <- selected_genes[!is.na(selected_genes)]
  selected_genes
}

get_liger_raw_list_compat <- function(lig) {
  out <- NULL

  try({
    out <- rawData(lig)
  }, silent = TRUE)

  if (is.null(out)) {
    try({
      out <- getMatrix(lig, "rawData")
    }, silent = TRUE)
  }

  if (is.null(out) && "raw.data" %in% slotNames(lig)) {
    out <- lig@raw.data
  }

  out
}

set_var_features_compat <- function(lig, genes) {
  genes <- unique(as.character(genes))
  genes <- genes[!is.na(genes)]
  ok <- FALSE

  try({
    varFeatures(lig) <- genes
    ok <- TRUE
  }, silent = TRUE)

  if (!ok && "var.genes" %in% slotNames(lig)) {
    lig@var.genes <- genes
    ok <- TRUE
  }

  if (!ok) {
    stop("Could not set variable features on liger object.")
  }

  lig
}

get_var_features_compat <- function(lig) {
  out <- NULL

  try({
    out <- varFeatures(lig)
  }, silent = TRUE)

  if (is.null(out) && "var.genes" %in% slotNames(lig)) {
    out <- lig@var.genes
  }

  out <- unique(as.character(out))
  out <- out[!is.na(out)]
  out
}

run_online_inmf_compat <- function(lig, k, lambda) {
  out <- NULL

  try({
    out <- runOnlineINMF(lig, k = k, lambda = lambda)
  }, silent = TRUE)

  if (is.null(out)) {
    try({
      out <- online_iNMF(lig, k = k, lambda = lambda)
    }, silent = TRUE)
  }

  if (is.null(out)) {
    stop("Could not run online iNMF with either runOnlineINMF() or online_iNMF().")
  }

  out
}

quantile_norm_compat <- function(lig) {
  out <- NULL

  try({
    out <- quantileNorm(lig)
  }, silent = TRUE)

  if (is.null(out)) {
    try({
      out <- quantile_norm(lig)
    }, silent = TRUE)
  }

  if (is.null(out)) {
    stop("Could not run quantile normalization with either quantileNorm() or quantile_norm().")
  }

  out
}

get_H_list_compat <- function(lig) {
  out <- NULL

  try({
    out <- getMatrix(lig, "H")
  }, silent = TRUE)

  if (is.null(out) && "H" %in% slotNames(lig)) {
    out <- lig@H
  }

  if (is.null(out)) {
    stop("Could not retrieve H matrices from liger object.")
  }

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

  # Best case: rows already correspond to expected cell IDs
  if (!is.null(rn) && all(expected_cells %in% rn)) {
    h <- h[expected_cells, , drop = FALSE]
    h <- assign_factor_names(h)
    return(h)
  }

  # Best case: columns correspond to expected cell IDs, so transpose to cells x k
  if (!is.null(cn) && all(expected_cells %in% cn)) {
    h <- t(h[, expected_cells, drop = FALSE])
    rownames(h) <- expected_cells
    h <- assign_factor_names(h)
    return(h)
  }

  # Names may exist but still be non-canonical. If dimensions match exactly,
  # fall back to order-preserving alignment.
  if (nrow(h) == n_expected) {
    if (!is.null(rn) && length(rn) == n_expected && setequal(rn, expected_cells)) {
      h <- h[expected_cells, , drop = FALSE]
    } else {
      message(
        "Warning: H rownames for batch '", batch_name,
        "' are missing or non-canonical; assuming row order matches expected cell order."
      )
      rownames(h) <- expected_cells
    }
    h <- assign_factor_names(h)
    return(h)
  }

  if (ncol(h) == n_expected) {
    if (!is.null(cn) && length(cn) == n_expected && setequal(cn, expected_cells)) {
      h <- t(h[, expected_cells, drop = FALSE])
    } else {
      message(
        "Warning: H colnames for batch '", batch_name,
        "' are missing or non-canonical; assuming column order matches expected cell order and transposing."
      )
      h <- t(h)
    }
    rownames(h) <- expected_cells
    h <- assign_factor_names(h)
    return(h)
  }

  stop(
    "Could not align H matrix for batch '", batch_name, "'. ",
    "Expected ", n_expected, " cells, but got matrix dim=",
    paste(dim(h), collapse = "x")
  )
}

# ---------------------------
# 1) Load .mat datasets into list of (genes x cells) sparse matrices
#    with benchmark-compatible feature IDs
# ---------------------------
message("Loading .mat datasets...")

mat_list <- list()

for (b in seq_along(DATA_FILES)) {
  f <- DATA_FILES[b]
  path <- file.path(DATA_FOLDER, f)
  if (!file.exists(path)) stop("Missing file: ", path)

  mat_data <- readMat(path)
  key_name <- get_data_key(mat_data)
  if (is.na(key_name)) stop("Could not find a data key in: ", f)

  X <- mat_data[[key_name]]
  message(sprintf("  %s key=%s raw_dim=%s", f, key_name, paste(dim(X), collapse = "x")))

  # counts should be genes x cells (transpose if stored as cells x genes)
  counts <- t(X)

  # Sparse
  counts <- to_dgC(counts)

  # Benchmark-compatible implicit feature IDs: "0", "1", ..., "n-1"
  rownames(counts) <- as.character(seq_len(nrow(counts)) - 1L)

  # Cell IDs:
  # Cell-<cellIndex>-Batch-<BatchName>
  colnames(counts) <- paste0("Cell-", seq_len(ncol(counts)), "-Batch-", BATCH_NAMES[b])

  # Use batch names as list keys
  mat_list[[BATCH_NAMES[b]]] <- counts
}

# ---------------------------
# 2) Enforce shared genes across all batches
# ---------------------------
gene_sets <- lapply(mat_list, rownames)
shared_genes <- Reduce(intersect, gene_sets)
shared_genes <- unique(as.character(shared_genes))
shared_genes <- shared_genes[!is.na(shared_genes)]

message("Shared genes across batches: ", length(shared_genes))
if (length(shared_genes) < MIN_SHARED_GENES) {
  stop("Too few shared genes across batches: ", length(shared_genes))
}

mat_list <- lapply(mat_list, function(m) m[shared_genes, , drop = FALSE])

message("Cells per batch:")
print(sapply(mat_list, ncol))

# Canonical cell order expected everywhere
all_cells <- unlist(lapply(mat_list, colnames), use.names = FALSE)
all_cells_by_batch <- lapply(mat_list, colnames)
message("Total cells across batches: ", length(all_cells))

# ---------------------------
# 3) v11 HVG logic: frozen canonical file by default
# ---------------------------
message("Selecting variable genes...")

selected_genes <- character(0)
hvg_logic_used <- ""
used_canonical_file <- FALSE

if (USE_CANONICAL_HVG_FILE) {
  message("Attempting to use frozen canonical HVG file: ", CANONICAL_HVG_FILE)

  if (!file.exists(CANONICAL_HVG_FILE)) {
    if (!FALLBACK_TO_V10_FAIR_HVGS_IF_MISSING) {
      stop("Canonical HVG file not found: ", CANONICAL_HVG_FILE)
    }
    message("Canonical HVG file missing; falling back to v10 fair per-batch HVG logic.")
    selected_genes <- select_fair_hvgs_v10(mat_list, N_VAR_GENES_PER_BATCH, N_VAR_GENES_FINAL)
    selected_genes <- intersect(selected_genes, shared_genes)
    selected_genes <- unique(as.character(selected_genes))
    selected_genes <- selected_genes[!is.na(selected_genes)]
    hvg_logic_used <- "v10 fallback: per-batch HVGs -> union -> cap"
  } else {
    canonical_genes <- read_canonical_hvgs(CANONICAL_HVG_FILE)
    if (length(canonical_genes) == 0) {
      stop("Canonical HVG file is empty: ", CANONICAL_HVG_FILE)
    }

    missing_canonical <- setdiff(canonical_genes, shared_genes)
    if (REQUIRE_EXACT_CANONICAL_HVG_MATCH && length(missing_canonical) > 0) {
      stop(
        "Canonical HVG file contains genes not present in the shared feature space. ",
        "Example missing: ", paste(head(missing_canonical, 10), collapse = ", ")
      )
    }

    selected_genes <- canonical_genes[canonical_genes %in% shared_genes]
    selected_genes <- unique(as.character(selected_genes))
    selected_genes <- selected_genes[!is.na(selected_genes)]

    message("Canonical HVGs loaded: ", length(canonical_genes))
    message("Canonical HVGs matched shared genes: ", length(selected_genes))

    if (length(selected_genes) < MIN_SHARED_GENES) {
      if (!FALLBACK_TO_V10_FAIR_HVGS_IF_MISSING) {
        stop("Too few canonical HVGs matched the shared feature space: ", length(selected_genes))
      }
      message("Too few canonical HVGs matched; falling back to v10 fair per-batch HVG logic.")
      selected_genes <- select_fair_hvgs_v10(mat_list, N_VAR_GENES_PER_BATCH, N_VAR_GENES_FINAL)
      selected_genes <- intersect(selected_genes, shared_genes)
      selected_genes <- unique(as.character(selected_genes))
      selected_genes <- selected_genes[!is.na(selected_genes)]
      hvg_logic_used <- "v10 fallback: per-batch HVGs -> union -> cap"
    } else {
      used_canonical_file <- TRUE
      hvg_logic_used <- paste0("frozen canonical HVG file: ", basename(CANONICAL_HVG_FILE))
    }
  }
} else {
  selected_genes <- select_fair_hvgs_v10(mat_list, N_VAR_GENES_PER_BATCH, N_VAR_GENES_FINAL)
  selected_genes <- intersect(selected_genes, shared_genes)
  selected_genes <- unique(as.character(selected_genes))
  selected_genes <- selected_genes[!is.na(selected_genes)]
  hvg_logic_used <- "v10 fair per-batch HVGs -> union -> cap"
}

if (length(selected_genes) < MIN_SHARED_GENES) {
  stop("Too few selected genes before LIGER object creation: ", length(selected_genes))
}

message("Selected genes before createLiger: ", length(selected_genes))

# Restrict all batches to selected genes, preserving selected_genes order
mat_list <- lapply(mat_list, function(m) m[selected_genes, , drop = FALSE])

# ---------------------------
# 4) Create LIGER object
# ---------------------------
lig <- createLiger(mat_list)

# Check gene availability after object creation without assuming legacy slots
common_genes_after <- Reduce(intersect, lapply(mat_list, rownames))
raw_list_after <- get_liger_raw_list_compat(lig)
if (!is.null(raw_list_after)) {
  try({
    common_genes_after <- Reduce(intersect, lapply(raw_list_after, rownames))
  }, silent = TRUE)
}
common_genes_after <- unique(as.character(common_genes_after))
common_genes_after <- common_genes_after[!is.na(common_genes_after)]

if (used_canonical_file) {
  missing_after <- setdiff(selected_genes, common_genes_after)
  if (REQUIRE_EXACT_CANONICAL_HVG_MATCH && length(missing_after) > 0) {
    stop(
      "Canonical selected genes are missing after LIGER object creation. ",
      "Example missing: ", paste(head(missing_after, 10), collapse = ", ")
    )
  }
}

selected_genes <- selected_genes[selected_genes %in% common_genes_after]
selected_genes <- unique(as.character(selected_genes))
selected_genes <- selected_genes[!is.na(selected_genes)]

message("Selected genes present after LIGER object creation: ", length(selected_genes))

if (length(selected_genes) < MIN_SHARED_GENES) {
  if (used_canonical_file && !FALLBACK_TO_V10_FAIR_HVGS_IF_MISSING) {
    stop("Too few canonical genes remain after LIGER object creation: ", length(selected_genes))
  }
  message("Too few selected genes remain after object creation (", length(selected_genes), "). Falling back to common genes.")
  selected_genes <- common_genes_after
}

lig <- set_var_features_compat(lig, selected_genes)
var_genes_now <- get_var_features_compat(lig)

if (length(var_genes_now) < MIN_SHARED_GENES) {
  message("Too few varFeatures (", length(var_genes_now), "). Using all shared/common genes.")
  lig <- set_var_features_compat(lig, common_genes_after)
  var_genes_now <- get_var_features_compat(lig)
}

message("Variable genes set to: ", length(var_genes_now))
message("HVG logic: ", hvg_logic_used)

# ---------------------------
# 5) Seurat-like preprocessing
# ---------------------------
message("normalize() ...")
lig <- normalize(lig)

message("scaleNotCenter() ...")
lig <- scaleNotCenter(lig)

# ---------------------------
# 6) Online iNMF
# ---------------------------
k_use <- min(K_FACTORS, length(var_genes_now) - 1)
if (k_use < 10) stop("Not enough genes for k factors. varFeatures=", length(var_genes_now))

message("Online iNMF(k=", k_use, ", lambda=", LAMBDA, ") ...")
lig <- run_online_inmf_compat(lig, k = k_use, lambda = LAMBDA)

# ---------------------------
# 7) Quantile normalization
# ---------------------------
message("Quantile normalization ...")
lig <- quantile_norm_compat(lig)

# ---------------------------
# 8) Export embedding H (cells x k) to CSV
#    lig@H is a list with one H per batch
# ---------------------------
message("Exporting H (cells x k) ...")

H_raw_list <- get_H_list_compat(lig)
if ((is.null(names(H_raw_list)) || any(names(H_raw_list) == "")) && length(H_raw_list) == length(BATCH_NAMES)) {
  names(H_raw_list) <- BATCH_NAMES
}

if (!all(BATCH_NAMES %in% names(H_raw_list))) {
  missing_h <- setdiff(BATCH_NAMES, names(H_raw_list))
  stop("LIGER H is missing batches: ", paste(missing_h, collapse = ", "))
}

H_list <- list()
for (b in BATCH_NAMES) {
  H_list[[b]] <- coerce_H_to_cells_by_k(
    H_raw_list[[b]],
    expected_cells = all_cells_by_batch[[b]],
    batch_name = b
  )
}

H <- do.call(rbind, H_list[BATCH_NAMES])

# Safety checks
if (is.null(rownames(H))) stop("H has no rownames; cannot align cells.")
missing <- setdiff(all_cells, rownames(H))
extra   <- setdiff(rownames(H), all_cells)

if (length(missing) > 0) {
  stop("H is missing expected cells. Example missing: ", paste(head(missing, 10), collapse = ", "))
}
if (length(extra) > 0) {
  message("Warning: H has extra cells not expected (showing up to 10): ",
          paste(head(extra, 10), collapse = ", "))
}

# Reorder to canonical cell order
H <- H[all_cells, , drop = FALSE]

write.csv(H, OUT_CSV, quote = FALSE)
message("✅ Done. Wrote: ", OUT_CSV)
message("Final HVG logic used: ", hvg_logic_used)
