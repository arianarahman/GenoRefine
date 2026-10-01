# seurat_integration_pancreas_robustness_canonical_v1.R
# Purpose: Generate Seurat comparison embeddings for pancreas robustness conditions.
# Author: Ariana Rahman (Arizona State University)
# -----------------------------------------------------------------------------
# Strict Seurat reruns for pancreas robustness perturbations.
# Reads manifests created by make_pancreas_robustness_manifests_canonical_v1.py
# and exports one Seurat embedding CSV per perturbation condition.
# -----------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(Seurat)
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

N_PCS <- 50
ANCHOR_DIMS <- 30
MIN_BATCH_CELLS <- 5
CANONICAL_HVG_FILE <- file.path(OUT_DIR, "pancreas_hvg_canonical.txt")
REQUIRE_EXACT_CANONICAL_HVG_MATCH <- TRUE

get_data_key <- function(mat_data) {
  candidates <- grep("^data", names(mat_data), value = TRUE, ignore.case = TRUE)
  if (length(candidates) == 0) candidates <- grep("data", names(mat_data), value = TRUE, ignore.case = TRUE)
  if (length(candidates) == 0) return(NA_character_)
  candidates[1]
}

safe_as_dgC <- function(x) {
  if (!inherits(x, "dgCMatrix")) x <- as(x, "dgCMatrix")
  x
}

read_canonical_hvgs <- function(path) {
  if (!file.exists(path)) stop("Canonical HVG file not found: ", path)
  x <- readLines(path, warn = FALSE)
  x <- trimws(x)
  x <- x[nzchar(x)]
  unique(as.character(x[!is.na(x)]))
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
    cat(sprintf("  %s key=%s raw_dim=%s\n", f, key_name, paste(dim(X), collapse = "x")))

    counts <- t(X)                    # genes x cells
    counts <- as.matrix(counts)
    counts[counts < 0] <- 0
    counts <- round(counts)
    counts <- Matrix(counts, sparse = TRUE)
    counts <- safe_as_dgC(counts)

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

  # Remove zero-count cells after canonical HVG subsetting, if any.
  for (b in names(cond)) {
    nz <- Matrix::colSums(cond[[b]] != 0)
    if (any(nz == 0)) {
      cat("Removing", sum(nz == 0), "zero-count cells in", b, "\n")
      cond[[b]] <- cond[[b]][, nz > 0, drop = FALSE]
    }
  }

  actual_cells <- keep_cells[keep_cells %in% unlist(lapply(cond, colnames), use.names = FALSE)]
  list(mat_list = cond, cells = actual_cells)
}

run_seurat_condition <- function(cond_mat_list, condition_cells, out_csv) {
  cat("Creating Seurat objects for condition; cells=", length(condition_cells), "\n")
  seurat_list <- list()
  for (b in names(cond_mat_list)) {
    obj <- CreateSeuratObject(counts = cond_mat_list[[b]], project = paste0("Pancreas_", b))
    obj$batch <- b
    obj$batch_name <- b
    obj <- NormalizeData(obj, verbose = FALSE)
    common_selected <- selected_genes[selected_genes %in% rownames(obj)]
    VariableFeatures(obj) <- common_selected
    seurat_list[[b]] <- obj
  }

  shared_genes_use <- Reduce(intersect, lapply(seurat_list, rownames))
  selected_genes_use <- selected_genes[selected_genes %in% shared_genes_use]
  if (length(selected_genes_use) < 500) stop("Too few selected genes shared across condition batches: ", length(selected_genes_use))

  for (b in names(seurat_list)) {
    seurat_list[[b]] <- seurat_list[[b]][selected_genes_use, ]
    VariableFeatures(seurat_list[[b]]) <- selected_genes_use
  }

  anchor_dims_use <- min(ANCHOR_DIMS, length(selected_genes_use) - 1)
  if (anchor_dims_use < 2) stop("anchor_dims_use became too small")

  anchors <- FindIntegrationAnchors(object.list = seurat_list, anchor.features = selected_genes_use, dims = 1:anchor_dims_use)
  integrated <- IntegrateData(anchorset = anchors, dims = 1:anchor_dims_use)
  DefaultAssay(integrated) <- "integrated"
  integrated <- ScaleData(integrated, verbose = FALSE)

  npcs_use <- min(N_PCS, ncol(integrated) - 1, length(selected_genes_use) - 1)
  integrated <- RunPCA(integrated, npcs = npcs_use, verbose = FALSE)
  pca_coords <- Embeddings(integrated, reduction = "pca")
  if (is.null(colnames(pca_coords))) colnames(pca_coords) <- paste0("PC_", seq_len(ncol(pca_coords)))

  missing <- setdiff(condition_cells, rownames(pca_coords))
  if (length(missing) > 0) {
    stop("Seurat PCA missing condition cells. Examples: ", paste(head(missing, 10), collapse = ", "))
  }

  pca_out <- as.matrix(pca_coords[condition_cells, , drop = FALSE])
  write.csv(pca_out, out_csv, quote = FALSE)
  cat("Wrote ", out_csv, " rows=", nrow(pca_out), " cols=", ncol(pca_out), "\n", sep = "")
}

cat("Loading pancreas matrices...\n")
mat_list <- load_pancreas_mats()
shared_genes <- Reduce(intersect, lapply(mat_list, rownames))
mat_list <- lapply(mat_list, function(m) m[shared_genes, , drop = FALSE])

selected_genes <- read_canonical_hvgs(CANONICAL_HVG_FILE)
missing_canonical <- setdiff(selected_genes, shared_genes)
if (REQUIRE_EXACT_CANONICAL_HVG_MATCH && length(missing_canonical) > 0) {
  stop("Canonical HVGs missing from shared genes. Examples: ", paste(head(missing_canonical, 10), collapse = ", "))
}
selected_genes <- selected_genes[selected_genes %in% shared_genes]
if (length(selected_genes) < 500) stop("Too few selected canonical HVGs: ", length(selected_genes))
mat_list <- lapply(mat_list, function(m) m[selected_genes, , drop = FALSE])

manifest_files <- list.files(ROBUST_DIR, pattern = "^cells_.*\\.csv$", full.names = TRUE)
if (length(manifest_files) == 0) {
  stop("No cell manifests found in ", ROBUST_DIR, ". Run make_pancreas_robustness_manifests_canonical_v1.py first.")
}

for (manifest_path in manifest_files) {
  ctx <- sub("^cells_", "", tools::file_path_sans_ext(basename(manifest_path)))
  out_csv <- file.path(ROBUST_DIR, paste0("X_seurat_pancreas_", ctx, ".csv"))
  cat("\n=== Seurat condition: ", ctx, " ===\n", sep = "")
  cond <- subset_mats_to_manifest(mat_list, manifest_path)
  run_seurat_condition(cond$mat_list, cond$cells, out_csv)
}

cat("✅ Strict Seurat robustness reruns complete.\n")
