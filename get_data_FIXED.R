library(tidyverse)
library(SummarizedExperiment)
library(TCGAbiolinks)


query_LUAD = GDCquery(
  project = "TCGA-LUAD",
  data.category = "Transcriptome Profiling", 
  data.type = "Gene Expression Quantification",
  experimental.strategy = "RNA-Seq",
  workflow.type = "STAR - Counts",
  sample.type = c("Primary Tumor", "Solid Tissue Normal"),
)

GDCdownload(
  query_LUAD, 
  method = "api", 
  files.per.chunk = 10
)

 


LUAD_data = GDCprepare(query_LUAD) 

LUADMatrix <- SummarizedExperiment::assay(LUAD_data)

sample_metadata = data.frame(colData(LUAD_data))

gene_metadata = data.frame(rowData(LUAD_data))

write.table(LUADMatrix, file="count_matrix.txt")
write.table(gene_metadata, file="gene_metadata.txt")

library(readxl)
write_excel_csv(sample_metadata, file = "sample_metadata.csv")

