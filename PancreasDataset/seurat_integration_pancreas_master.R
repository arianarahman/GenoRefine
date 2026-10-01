# ============================================================
# Purpose: Generate the Seurat comparison embedding for the five-study pancreas benchmark.
# Author: Ariana Rahman (Arizona State University)
# FILE: seurat_integration_pancreas_v5.R
# PURPOSE:
#   Run Seurat integration on the 5-pancreas .mat benchmark,
#   updated to use the frozen pancreas canonical HVG file by default.
#
# v5 CHANGE FROM v4:
#   PRIMARY change is feature selection:
#     - read pancreas_hvg_canonical.txt by default
#     - preserve canonical HVG order when subsetting features
#     - keep v4 fair-HVG union logic only as an optional fallback
#
#   COMPATIBILITY change:
#     - use benchmark-compatible numeric feature IDs ("0".."1999")
#       instead of "Gene-<i>" so the canonical HVG file matches exactly
#
# EVERYTHING ELSE intentionally remains aligned with v4:
#   - same loading / naming / batch ordering logic
#   - same Seurat integration workflow
#   - same PCA export aligned to canonical pancreas obs_names
#
# INPUT:
#   ./Dataset/dataBaronX.mat
#   ./Dataset/dataMuraroX.mat
#   ./Dataset/dataScapleX.mat
#   ./Dataset/dataWangX.mat
#   ./Dataset/dataXinX.mat
#   ./Benchmark_Out/pancreas_hvg_canonical.txt
#
# OUTPUT:
#   ./Benchmark_Out/X_seurat_pancreas.csv
# ============================================================

suppressPackageStartupMessages({
  library(Seurat)
  library(R.matlab)
  library(Matrix)
})

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR <- Sys.getenv("GENOREFINE_DATASET_ROOT", unset = getwd())
setwd(BASE_DIR)

DATA_FOLDER <- "./Dataset"
OUT_DIR <- "./Benchmark_Out"
OUT_CSV <- file.path(OUT_DIR, "X_seurat_pancreas.csv")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

DATA_FILES <- c(
  "dataBaronX.mat",
  "dataMuraroX.mat",
  "dataScapleX.mat",   # token "Scaple" -> displayed as "Segerstolpe"
  "dataWangX.mat",
  "dataXinX.mat"
)

BATCH_NAMES <- c("Baron", "Muraro", "Segerstolpe", "Wang", "Xin")
stopifnot(length(DATA_FILES) == length(BATCH_NAMES))

# Seurat params
N_PCS <- 50
ANCHOR_DIMS <- 30
MIN_BATCH_CELLS <- 5

# Canonical HVG controls (NEW IN v5)
USE_CANONICAL_HVG_FILE <- TRUE
CANONICAL_HVG_FILE <- file.path(OUT_DIR, "pancreas_hvg_canonical.txt")
FALLBACK_TO_V4_FAIR_HVGS_IF_MISSING <- FALSE
REQUIRE_EXACT_CANONICAL_HVG_MATCH <- TRUE

# Legacy v4 fallback params (used only if fallback is enabled)
N_VAR_GENES_PER_BATCH <- 2000
N_VAR_GENES_FINAL <- 3000

set.seed(0)

# -----------------------------
# Helpers
# -----------------------------
get_data_key <- function(mat_data) {
  candidates <- grep("^data", names(mat_data), value = TRUE, ignore.case = TRUE)
  if (length(candidates) == 0) {
    candidates <- grep("data", names(mat_data), value = TRUE, ignore.case = TRUE)
  }
  if (length(candidates) == 0) return(NA_character_)
  candidates[1]
}

safe_as_dgC <- function(x) {
  if (!inherits(x, "dgCMatrix")) x <- as(x, "dgCMatrix")
  x
}

read_canonical_hvgs <- function(path) {
  if (!file.exists(path)) {
    stop("Canonical HVG file not found: ", path)
  }

  hvgs <- readLines(path, warn = FALSE)
  hvgs <- trimws(hvgs)
  hvgs <- hvgs[nzchar(hvgs)]
  hvgs <- unique(as.character(hvgs))
  hvgs <- hvgs[!is.na(hvgs)]
  hvgs
}

# -----------------------------
# 1) Load .mat datasets into sparse matrices (genes x cells)
#    with canonical pancreas cell naming
# -----------------------------
cat("Loading .mat datasets...\n")

mat_list <- list()
all_cells <- character(0)
all_genes <- NULL

for (i in seq_along(DATA_FILES)) {
  f <- DATA_FILES[i]
  batch_name <- BATCH_NAMES[i]
  path <- file.path(DATA_FOLDER, f)
  if (!file.exists(path)) stop("Missing file: ", path)

  mat_data <- readMat(path)
  key_name <- get_data_key(mat_data)
  if (is.na(key_name)) stop("Could not find a data key in: ", f)

  X <- mat_data[[key_name]]
  cat(sprintf("  %s key=%s raw_dim=%s\n", f, key_name, paste(dim(X), collapse = "x")))

  # Expected source matrices are cells x genes -> transpose to genes x cells
  counts <- t(X)
  counts <- as.matrix(counts)
  counts[counts < 0] <- 0
  counts <- round(counts)
  counts <- Matrix(counts, sparse = TRUE)
  counts <- safe_as_dgC(counts)

  # Benchmark-compatible dummy feature naming by position across batches
  # Python canonical HVG generation uses implicit var names: "0".."n-1"
  rownames(counts) <- as.character(seq_len(nrow(counts)) - 1L)
  colnames(counts) <- paste0("Cell-", seq_len(ncol(counts)), "-Batch-", batch_name)

  mat_list[[batch_name]] <- counts
  all_cells <- c(all_cells, colnames(counts))
  all_genes <- rownames(counts)
}

cat("Cells per batch:\n")
print(sapply(mat_list, ncol))
cat("Total cells across batches:", length(all_cells), "\n")

# -----------------------------
# 2) Enforce shared genes across all batches
# -----------------------------
shared_genes <- Reduce(intersect, lapply(mat_list, rownames))
cat("Shared genes across batches:", length(shared_genes), "\n")
if (length(shared_genes) < 500) {
  stop("Too few shared genes across batches: ", length(shared_genes))
}

mat_list <- lapply(mat_list, function(m) m[shared_genes, , drop = FALSE])

# -----------------------------
# 3) Feature selection: canonical HVG file by default
#    fallback to v4 fair-HVG union only if enabled
# -----------------------------
skipped_batches <- character(0)
selected_genes <- character(0)
feature_selection_mode <- ""

if (isTRUE(USE_CANONICAL_HVG_FILE)) {
  if (file.exists(CANONICAL_HVG_FILE)) {
    cat("Using canonical HVG file:", CANONICAL_HVG_FILE, "\n")
    canonical_hvgs <- read_canonical_hvgs(CANONICAL_HVG_FILE)
    cat("Canonical HVGs loaded:", length(canonical_hvgs), "\n")

    missing_from_data <- setdiff(canonical_hvgs, shared_genes)
    extra_in_data <- setdiff(shared_genes, canonical_hvgs)

    if (isTRUE(REQUIRE_EXACT_CANONICAL_HVG_MATCH)) {
      if (length(missing_from_data) > 0 || length(extra_in_data) > 0) {
        stop(
          paste0(
            "Canonical HVG mismatch detected. ",
            "Missing from data=", length(missing_from_data),
            ", extra in data=", length(extra_in_data), ". ",
            "Examples missing: ", paste(head(missing_from_data, 10), collapse = ", "), "; ",
            "examples extra: ", paste(head(extra_in_data, 10), collapse = ", ")
          )
        )
      }
    } else {
      if (length(missing_from_data) > 0) {
        cat("Warning: canonical genes missing from data:", length(missing_from_data), "\n")
      }
    }

    # Preserve file order exactly
    selected_genes <- canonical_hvgs[canonical_hvgs %in% shared_genes]
    selected_genes <- unique(as.character(selected_genes))
    selected_genes <- selected_genes[!is.na(selected_genes)]

    cat("Canonical genes present in shared feature space:", length(selected_genes), "\n")
    feature_selection_mode <- "canonical_file"
  } else {
    msg <- paste0("Canonical HVG file not found: ", CANONICAL_HVG_FILE)
    if (isTRUE(FALLBACK_TO_V4_FAIR_HVGS_IF_MISSING)) {
      cat(msg, "\n")
      cat("Falling back to v4 fair-HVG union logic...\n")
      feature_selection_mode <- "fallback_v4_fair_hvg"
    } else {
      stop(msg)
    }
  }
} else {
  feature_selection_mode <- "fallback_v4_fair_hvg"
}

if (identical(feature_selection_mode, "fallback_v4_fair_hvg")) {
  cat("Selecting HVGs per batch (v4 fair union fallback)...\n")

  hvg_lists <- list()

  for (b in names(mat_list)) {
    n_cells <- ncol(mat_list[[b]])
    cat("  - batch:", b, "cells:", n_cells, "\n")

    if (n_cells < MIN_BATCH_CELLS) {
      cat("    Skipping tiny batch:", b, "\n")
      skipped_batches <- c(skipped_batches, b)
      next
    }

    if (length(mat_list[[b]]@x) > 0) {
      mat_list[[b]]@x[mat_list[[b]]@x < 0] <- 0
    }

    tmp <- CreateSeuratObject(counts = mat_list[[b]], project = paste0("Pancreas_", b))
    tmp <- NormalizeData(tmp, verbose = FALSE)
    tmp <- FindVariableFeatures(
      tmp,
      selection.method = "vst",
      nfeatures = N_VAR_GENES_PER_BATCH,
      verbose = FALSE
    )

    hvg_lists[[b]] <- VariableFeatures(tmp)
  }

  # Keep only batches that were not skipped
  mat_list <- mat_list[setdiff(names(mat_list), skipped_batches)]

  if (length(mat_list) < 2) {
    stop("Need >=2 usable batches after filtering tiny batches. Found: ", length(mat_list))
  }

  all_hvgs <- unlist(hvg_lists, use.names = FALSE)
  if (length(all_hvgs) == 0) {
    stop("No HVGs were selected from any batch.")
  }

  hvg_freq <- sort(table(all_hvgs), decreasing = TRUE)
  selected_genes <- names(hvg_freq)
  if (length(selected_genes) > N_VAR_GENES_FINAL) {
    selected_genes <- selected_genes[seq_len(N_VAR_GENES_FINAL)]
  }

  # Preserve fallback ranking order
  selected_genes <- selected_genes[selected_genes %in% shared_genes]
  selected_genes <- unique(as.character(selected_genes))
  selected_genes <- selected_genes[!is.na(selected_genes)]

  cat("Fallback fair-HVG union size (after cap):", length(selected_genes), "\n")
}

if (length(selected_genes) < 500) {
  stop("Too few selected genes after feature selection: ", length(selected_genes))
}

# Subset matrices to selected genes in selected order
for (b in names(mat_list)) {
  keep_b <- selected_genes[selected_genes %in% rownames(mat_list[[b]])]
  mat_list[[b]] <- mat_list[[b]][keep_b, , drop = FALSE]
}

# Remove cells with zero counts across selected genes
removed_zero_cells <- character(0)
for (b in names(mat_list)) {
  nz <- Matrix::colSums(mat_list[[b]] != 0)
  keep <- (nz > 0)
  if (sum(!keep) > 0) {
    cat("Removing", sum(!keep), "cells with 0 counts across selected genes in", b, ".\n")
    removed_zero_cells <- c(removed_zero_cells, colnames(mat_list[[b]])[!keep])
    mat_list[[b]] <- mat_list[[b]][, keep, drop = FALSE]
  }
}

# Remove batches that became too small after filtering
post_batches <- names(mat_list)
valid_batches <- c()
for (b in post_batches) {
  if (ncol(mat_list[[b]]) >= MIN_BATCH_CELLS) {
    valid_batches <- c(valid_batches, b)
  } else {
    cat("Dropping batch after filtering because it has < ", MIN_BATCH_CELLS, " cells: ", b, "\n", sep = "")
    skipped_batches <- c(skipped_batches, b)
  }
}
mat_list <- mat_list[valid_batches]

if (length(mat_list) < 2) {
  stop("Need >=2 usable batches after zero-cell filtering. Found: ", length(mat_list))
}

# -----------------------------
# 4) Create Seurat objects + integrate + PCA
# -----------------------------
cat("Creating Seurat objects...\n")

seurat_list <- list()
for (b in names(mat_list)) {
  obj <- CreateSeuratObject(
    counts = mat_list[[b]],
    project = paste0("Pancreas_", b)
  )

  obj$batch <- b
  obj$batch_name <- b

  obj <- NormalizeData(obj, verbose = FALSE)

  common_selected <- selected_genes[selected_genes %in% rownames(obj)]
  VariableFeatures(obj) <- common_selected

  seurat_list[[b]] <- obj
}

shared_genes_use <- Reduce(intersect, lapply(seurat_list, rownames))
cat("Shared genes across usable batches:", length(shared_genes_use), "\n")

selected_genes_use <- selected_genes[selected_genes %in% shared_genes_use]
cat("Selected genes present in ALL datasets:", length(selected_genes_use), "\n")
if (length(selected_genes_use) < 500) {
  stop("Too few selected genes shared across all datasets: ", length(selected_genes_use))
}

for (b in names(seurat_list)) {
  seurat_list[[b]] <- seurat_list[[b]][selected_genes_use, ]
  VariableFeatures(seurat_list[[b]]) <- selected_genes_use
}

anchor_dims_use <- min(ANCHOR_DIMS, length(selected_genes_use) - 1)
if (anchor_dims_use < 2) {
  stop("anchor_dims_use became too small: ", anchor_dims_use)
}
cat("Using anchor dims: 1:", anchor_dims_use, "\n", sep = "")

cat("Finding integration anchors...\n")
anchors <- FindIntegrationAnchors(
  object.list = seurat_list,
  anchor.features = selected_genes_use,
  dims = 1:anchor_dims_use
)

cat("Integrating data...\n")
seurat_integrated <- IntegrateData(
  anchorset = anchors,
  dims = 1:anchor_dims_use
)

DefaultAssay(seurat_integrated) <- "integrated"

cat("Scaling integrated assay...\n")
seurat_integrated <- ScaleData(seurat_integrated, verbose = FALSE)

npcs_use <- min(N_PCS, ncol(seurat_integrated) - 1, length(selected_genes_use) - 1)
if (npcs_use < 2) {
  stop("npcs_use became too small: ", npcs_use)
}

cat("Running PCA with", npcs_use, "PCs...\n")
seurat_integrated <- RunPCA(seurat_integrated, npcs = npcs_use, verbose = FALSE)

# -----------------------------
# 5) Export embedding aligned to canonical pancreas cell order
# -----------------------------
pca_coords <- Embeddings(seurat_integrated, reduction = "pca")

if (is.null(rownames(pca_coords))) {
  stop("PCA embedding has NULL rownames; cannot align to original pancreas cell order.")
}
if (is.null(colnames(pca_coords))) {
  colnames(pca_coords) <- paste0("PC_", seq_len(ncol(pca_coords)))
}

# Build FULL embedding matrix in canonical order, filling removed cells with NA
pca_full <- matrix(NA_real_, nrow = length(all_cells), ncol = ncol(pca_coords))
rownames(pca_full) <- all_cells
colnames(pca_full) <- colnames(pca_coords)

idx <- match(rownames(pca_coords), all_cells)
ok <- !is.na(idx)
if (any(!ok)) {
  warning(paste0("Some PCA rownames not found in expected all_cells: ", sum(!ok)))
}

pca_full[idx[ok], ] <- as.matrix(pca_coords[ok, , drop = FALSE])

embedded_cells <- rownames(pca_coords)
removed_cells <- setdiff(all_cells, embedded_cells)
cat("Cells in original canonical order:", length(all_cells), "\n")
cat("Cells with embeddings:", nrow(pca_coords), "\n")
cat("Cells removed / missing embeddings:", length(removed_cells), "\n")

write.csv(pca_full, OUT_CSV, quote = FALSE)
cat("Wrote embedding CSV (full, aligned):", OUT_CSV, "\n")

if (length(removed_cells) > 0) {
  removed_path <- file.path(OUT_DIR, "X_seurat_pancreas_removed_cells.txt")
  writeLines(removed_cells, removed_path)
  cat("Wrote removed cell list:", removed_path, "\n")
}

cat("Feature selection mode:", feature_selection_mode, "\n")
cat("✅ Seurat integration complete (pancreas).\n")
