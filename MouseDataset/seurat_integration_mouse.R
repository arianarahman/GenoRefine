# ============================================================
# Purpose: Generate the Seurat comparison embedding for the mouse atlas benchmark.
# Author: Ariana Rahman (Arizona State University)
# FILE: seurat_integration_mouse_v5.R
# PURPOSE:
#   Run Seurat integration on mouse Tabula Muris Senis (.h5ad),
#   using a separately generated frozen canonical HVG list.
#
#   v5 changes:
#     - switched FindIntegrationAnchors from default CCA to RPCA
#       (CCA is O(n^2) and freezes on 25k x 25k cells; RPCA is O(n))
#     - added per-batch ScaleData + RunPCA required by RPCA mode
#     - all other logic identical to v4
#
#   v4 updates:
#     - uses zellkonverter::readH5AD() current interface
#     - loads canonical HVGs from file and preserves their order
#     - fixes MD5 computation to match the Python canonical-HVG generator
#     - uses a fixed batch key for strict consistency
#     - exports the exact HVG list actually used by Seurat
#     - robustly aligns final PCA embedding back to original AnnData cell IDs
#
# INPUT (recommended):
#   ./Dataset/tabula-muris_sub50k_combined.h5ad
#   ./Benchmark_Mouse_Out/mouse_hvg_canonical.txt
#
# OUTPUT:
#   Benchmark_Mouse_Out/X_seurat_mouse.csv
#   Benchmark_Mouse_Out/seurat_mouse_hvg_used.txt
#   Benchmark_Mouse_Out/seurat_mouse_hvg_used.md5
#   Benchmark_Mouse_Out/X_seurat_mouse_removed_cells.txt   (if any)
# ============================================================

suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
  library(zellkonverter)
  library(SingleCellExperiment)
})

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR <- Sys.getenv("GENOREFINE_DATASET_ROOT", unset = getwd())
if (dir.exists(BASE_DIR)) {
  setwd(BASE_DIR)
} else {
  message("BASE_DIR not found. Using current working directory: ", getwd())
}

IN_H5AD <- "./Dataset/tabula-muris_sub50k_combined.h5ad"
OUT_DIR <- "Benchmark_Mouse_Out"
OUT_CSV <- file.path(OUT_DIR, "X_seurat_mouse.csv")

# Canonical HVG artifact created separately
CANONICAL_HVG_FILE <- file.path(OUT_DIR, "mouse_hvg_canonical.txt")
CANONICAL_HVG_MD5  <- file.path(OUT_DIR, "mouse_hvg_canonical.md5")

# Audit outputs for the exact gene set actually used by Seurat
OUT_HVG_USED_TXT <- file.path(OUT_DIR, "seurat_mouse_hvg_used.txt")
OUT_HVG_USED_MD5 <- file.path(OUT_DIR, "seurat_mouse_hvg_used.md5")
OUT_REMOVED_CELLS <- file.path(OUT_DIR, "X_seurat_mouse_removed_cells.txt")

# Fix batch key for strict consistency across scripts
FIXED_BATCH_KEY <- "dataset"

# Seurat params
N_PCS <- 50
ANCHOR_DIMS <- 30
MIN_BATCH_CELLS <- 5
MIN_GENES_REQUIRED <- 500
MIN_MATCH_RATE_WARN <- 0.80

set.seed(0)
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# -----------------------------
# Helpers
# -----------------------------
get_counts_matrix <- function(sce) {
  assay_names <- assayNames(sce)
  cat("Assays found:", paste(assay_names, collapse = ", "), "\n")
  if ("counts" %in% assay_names) return(assay(sce, "counts"))
  if ("X" %in% assay_names) return(assay(sce, "X"))
  if ("logcounts" %in% assay_names) return(assay(sce, "logcounts"))
  return(assay(sce, assay_names[1]))
}

safe_as_dgC <- function(x) {
  if (!inherits(x, "dgCMatrix")) x <- as(x, "dgCMatrix")
  x
}

get_celltype_column <- function(md) {
  candidates <- c("celltype", "cell_type", "cell_ontology_class", "class", "annotation")
  hit <- candidates[candidates %in% colnames(md)]
  if (length(hit) == 0) return(NA_character_)
  hit[1]
}

# Match Python exactly: join genes by LF ("\n"), no trailing newline
compute_gene_md5 <- function(genes) {
  genes <- as.character(genes)
  genes <- genes[!is.na(genes)]
  payload <- paste(genes, collapse = "\n")

  tf <- tempfile(fileext = ".txt")
  con <- file(tf, open = "wb")
  on.exit({
    try(close(con), silent = TRUE)
    unlink(tf)
  }, add = TRUE)

  writeChar(payload, con, eos = NULL, useBytes = TRUE)
  close(con)

  unname(tools::md5sum(tf))
}

load_canonical_hvgs <- function(path) {
  if (!file.exists(path)) {
    stop("Canonical HVG file not found: ", path)
  }

  if (grepl("\\.csv$", path, ignore.case = TRUE)) {
    df <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
    if ("gene" %in% colnames(df)) {
      genes <- df$gene
    } else {
      genes <- df[[1]]
    }
  } else {
    genes <- readLines(path, warn = FALSE)
  }

  genes <- as.character(genes)
  genes <- genes[!is.na(genes)]
  genes <- genes[nzchar(genes)]
  genes <- genes[!duplicated(genes)]
  genes
}

normalize_obs_name <- function(x) {
  x <- as.character(x)
  x <- gsub("-1$", "", x)
  x <- gsub("\\.1$", "", x)
  x <- gsub("\\.", "-", x)
  x
}

align_embedding_to_original_ids <- function(embedding_cell_x_k, original_cell_ids) {
  emb_ids <- rownames(embedding_cell_x_k)
  if (is.null(emb_ids)) {
    stop("Embedding matrix has NULL rownames; cannot align to original AnnData cell IDs.")
  }

  direct_idx <- match(original_cell_ids, emb_ids)
  direct_rate <- mean(!is.na(direct_idx))

  emb_norm <- normalize_obs_name(emb_ids)
  orig_norm <- normalize_obs_name(original_cell_ids)
  norm_idx <- match(orig_norm, emb_norm)
  norm_rate <- mean(!is.na(norm_idx))

  if (direct_rate >= norm_rate) {
    best_idx <- direct_idx
    best_rate <- direct_rate
    best_name <- "direct"
  } else {
    best_idx <- norm_idx
    best_rate <- norm_rate
    best_name <- "normalized"
  }

  cat("Best cell-ID alignment strategy:", best_name, "\n")
  cat("Best cell-ID match rate:", sprintf("%.4f", best_rate), "\n")

  if (best_rate < MIN_MATCH_RATE_WARN) {
    warning(
      "Low match rate when aligning Seurat PCA embedding back to original AnnData cell IDs: ",
      sprintf("%.4f", best_rate),
      ". Check whether barcode naming changed across workflows."
    )
  }

  full <- matrix(NA_real_, nrow = length(original_cell_ids), ncol = ncol(embedding_cell_x_k))
  rownames(full) <- original_cell_ids
  colnames(full) <- colnames(embedding_cell_x_k)

  ok <- !is.na(best_idx)
  full[ok, ] <- as.matrix(embedding_cell_x_k[best_idx[ok], , drop = FALSE])

  removed_cells <- original_cell_ids[!ok]
  list(full = full, removed_cells = removed_cells, match_rate = best_rate, strategy = best_name)
}

# -----------------------------
# 1) Load AnnData (.h5ad)
# -----------------------------
cat("Reading:", IN_H5AD, "\n")
sce <- zellkonverter::readH5AD(IN_H5AD, verbose = TRUE)

counts <- get_counts_matrix(sce)
counts <- safe_as_dgC(counts)

md <- as.data.frame(colData(sce))

# Ensure fixed dataset field exists
if (!(FIXED_BATCH_KEY %in% colnames(md))) {
  if ("method" %in% colnames(md)) {
    md[[FIXED_BATCH_KEY]] <- as.character(md$method)
  } else if ("technology" %in% colnames(md)) {
    md[[FIXED_BATCH_KEY]] <- as.character(md$technology)
  } else {
    stop(
      "Missing fixed batch key '", FIXED_BATCH_KEY, "' and could not infer it from method/technology.\n",
      "Available metadata columns: ", paste(colnames(md), collapse = ", ")
    )
  }
}

BATCH_KEY <- FIXED_BATCH_KEY
cat("Using fixed batch key:", BATCH_KEY, "\n")

cell_ids <- colnames(counts)
if (is.null(cell_ids)) {
  stop("Counts matrix has NULL colnames; cannot align embeddings. Ensure AnnData has obs_names.")
}

gene_names <- rownames(counts)
if (is.null(gene_names)) {
  stop("Counts matrix has NULL rownames; cannot run Seurat integration safely.")
}

rownames(md) <- cell_ids

# -----------------------------
# 2) Split by batch
# -----------------------------
batches <- as.character(md[[BATCH_KEY]])
names(batches) <- cell_ids

batch_levels <- sort(unique(batches))
cat("Number of batches:", length(batch_levels), "\n")

if (length(batch_levels) < 2) {
  warning(
    paste0(
      "Only 1 batch level found for key '", BATCH_KEY, "'. ",
      "Seurat integration will run but this is not a meaningful integration setting."
    )
  )
}

mat_list <- lapply(batch_levels, function(b) {
  idx <- which(batches == b)
  counts[, idx, drop = FALSE]
})
names(mat_list) <- batch_levels

for (b in names(mat_list)) {
  mat_list[[b]] <- safe_as_dgC(mat_list[[b]])
  rownames(mat_list[[b]]) <- gene_names
  if (is.null(colnames(mat_list[[b]]))) {
    colnames(mat_list[[b]]) <- names(batches)[batches == b]
  }
  # Safety: clamp negatives if present
  if (length(mat_list[[b]]@x) > 0) {
    mat_list[[b]]@x[mat_list[[b]]@x < 0] <- 0
  }
}

batch_tab <- sort(table(batches), decreasing = TRUE)
cat("Top batch sizes:\n")
print(head(batch_tab, 10))

# -----------------------------
# 3) Load canonical HVGs from file
# -----------------------------
cat("Loading canonical HVGs from:", CANONICAL_HVG_FILE, "\n")
selected_genes <- load_canonical_hvgs(CANONICAL_HVG_FILE)
cat("Canonical HVGs loaded:", length(selected_genes), "\n")

loaded_md5 <- compute_gene_md5(selected_genes)
cat("Loaded canonical HVG md5:", loaded_md5, "\n")

if (file.exists(CANONICAL_HVG_MD5)) {
  expected_md5 <- readLines(CANONICAL_HVG_MD5, warn = FALSE)
  expected_md5 <- expected_md5[nzchar(expected_md5)][1]
  if (!is.na(expected_md5) && nzchar(expected_md5)) {
    cat("Expected canonical HVG md5:", expected_md5, "\n")
    if (!identical(expected_md5, loaded_md5)) {
      warning(
        "Canonical HVG md5 mismatch.\n",
        "Expected: ", expected_md5, "\n",
        "Loaded:   ", loaded_md5, "\n",
        "If this persists, regenerate the .md5 file from the same canonical gene text file."
      )
    }
  }
}

# Keep only genes present in the dataset, preserving canonical order
selected_genes <- selected_genes[selected_genes %in% gene_names]
cat("Canonical HVGs present in dataset:", length(selected_genes), "\n")

if (length(selected_genes) < MIN_GENES_REQUIRED) {
  stop(
    "Too few canonical HVGs present in dataset after matching rownames: ",
    length(selected_genes)
  )
}

# Subset matrices to selected genes in canonical order
for (b in names(mat_list)) {
  mat_list[[b]] <- mat_list[[b]][selected_genes, , drop = FALSE]
}

# Remove cells with zero counts across selected genes
removed_zero_cells <- character(0)
for (b in names(mat_list)) {
  nz <- Matrix::colSums(mat_list[[b]] != 0)
  keep <- (nz > 0)
  if (sum(!keep) > 0) {
    cat("Removing", sum(!keep), "cells with 0 counts across canonical genes in", b, ".\n")
    removed_zero_cells <- c(removed_zero_cells, colnames(mat_list[[b]])[!keep])
    mat_list[[b]] <- mat_list[[b]][, keep, drop = FALSE]
  }
}

# Remove batches that became too small after filtering
valid_batches <- character(0)
for (b in names(mat_list)) {
  if (ncol(mat_list[[b]]) >= MIN_BATCH_CELLS) {
    valid_batches <- c(valid_batches, b)
  } else {
    cat("Dropping batch because it has < ", MIN_BATCH_CELLS, " cells after filtering: ", b, "\n", sep = "")
  }
}
mat_list <- mat_list[valid_batches]

if (length(mat_list) < 2) {
  stop("Need >=2 usable batches after canonical-HVG filtering. Found: ", length(mat_list))
}

# -----------------------------
# 4) Create Seurat objects + integrate + PCA
# -----------------------------
cat("Creating Seurat objects...\n")

celltype_col <- get_celltype_column(md)
seurat_list <- list()

for (b in names(mat_list)) {
  obj <- CreateSeuratObject(
    counts = mat_list[[b]],
    project = paste0("Mouse_", b)
  )

  obj$batch <- b
  obj[[BATCH_KEY]] <- b

  # optional celltype metadata
  if (!is.na(celltype_col)) {
    obj$celltype <- md[colnames(obj), celltype_col]
  }

  obj <- NormalizeData(obj, verbose = FALSE)

  # use canonical genes, preserving order
  common_selected <- selected_genes[selected_genes %in% rownames(obj)]
  VariableFeatures(obj) <- common_selected

  seurat_list[[b]] <- obj
}

shared_genes <- Reduce(intersect, lapply(seurat_list, rownames))
cat("Shared genes across usable batches:", length(shared_genes), "\n")

selected_genes_use <- selected_genes[selected_genes %in% shared_genes]
cat("Canonical genes present in ALL datasets:", length(selected_genes_use), "\n")

if (length(selected_genes_use) < MIN_GENES_REQUIRED) {
  stop("Too few canonical genes shared across all datasets: ", length(selected_genes_use))
}

# Audit files for the exact gene set used by Seurat
writeLines(selected_genes_use, OUT_HVG_USED_TXT)
used_md5 <- compute_gene_md5(selected_genes_use)
writeLines(used_md5, OUT_HVG_USED_MD5)
cat("Wrote exact HVG list used by Seurat:", OUT_HVG_USED_TXT, "\n")
cat("Used HVG md5:", used_md5, "\n")

# Subset all objects to the same shared selected genes, in the same order
for (b in names(seurat_list)) {
  seurat_list[[b]] <- subset(seurat_list[[b]], features = selected_genes_use)
  VariableFeatures(seurat_list[[b]]) <- selected_genes_use
}

# Anchor dims cannot exceed number of variable features - 1
anchor_dims_use <- min(ANCHOR_DIMS, length(selected_genes_use) - 1)
if (anchor_dims_use < 2) {
  stop("anchor_dims_use became too small: ", anchor_dims_use)
}
cat("Using anchor dims: 1:", anchor_dims_use, "\n", sep = "")

# -----------------------------
# FIX (v5): Per-batch ScaleData + RunPCA required by RPCA mode.
# CCA (the default) is O(n^2) and freezes on 25k x 25k cells.
# RPCA runs each batch's PCA independently then projects, making
# it feasible on large datasets. Runtime drops from hours/never
# to ~5 minutes on standard workstation hardware.
# -----------------------------
cat("Running per-batch ScaleData + PCA (required for RPCA integration)...\n")
for (b in names(seurat_list)) {
  seurat_list[[b]] <- ScaleData(seurat_list[[b]], verbose = FALSE)
  seurat_list[[b]] <- RunPCA(seurat_list[[b]], npcs = anchor_dims_use, verbose = FALSE)
  cat(" Done:", b, "\n")
}

cat("Finding integration anchors (reduction = rpca)...\n")
anchors <- FindIntegrationAnchors(
  object.list  = seurat_list,
  anchor.features = selected_genes_use,
  dims         = 1:anchor_dims_use,
  reduction    = "rpca"   # KEY FIX: replaces default "cca" which is O(n^2)
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
# 5) Export embedding aligned to AnnData cell order
# -----------------------------
pca_coords <- Embeddings(seurat_integrated, reduction = "pca")

if (is.null(rownames(pca_coords))) {
  stop("PCA embedding has NULL rownames; cannot align to original AnnData order.")
}

if (is.null(colnames(pca_coords))) {
  colnames(pca_coords) <- paste0("PC_", seq_len(ncol(pca_coords)))
}

aligned <- align_embedding_to_original_ids(
  embedding_cell_x_k = pca_coords,
  original_cell_ids  = cell_ids
)

pca_full      <- aligned$full
removed_cells <- aligned$removed_cells

cat("Cells in original AnnData:", length(cell_ids), "\n")
cat("Cells with matched embeddings:", sum(complete.cases(pca_full[, 1, drop = FALSE])), "\n")
cat("Cells removed / missing embeddings:", length(removed_cells), "\n")

write.csv(pca_full, OUT_CSV, quote = FALSE)
cat("Wrote embedding CSV (full, aligned):", OUT_CSV, "\n")

if (length(removed_cells) > 0) {
  writeLines(removed_cells, OUT_REMOVED_CELLS)
  cat("Wrote removed cell list:", OUT_REMOVED_CELLS, "\n")
}

cat("✅ Seurat integration complete (mouse, RPCA) using separately generated canonical HVGs.\n")
