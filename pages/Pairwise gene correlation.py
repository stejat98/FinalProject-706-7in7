# -*- coding: utf-8 -*-
"""Page3.ipynb

Automatically generated by Colaboratory.

paper_expression_subtypeal file is located at
    https://colab.research.google.com/drive/1BilmLReqiJXpD3lb5QcbGbM6345fGOV8
"""

### Loading necessary modules/packages
#from google.colab import drive #https://stackoverflow.com/questions/69869534/files-and-folders-in-google-colab
#drive.mount("/content/gdrive")
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

import streamlit as st
import time
import numpy as np

st.set_page_config(page_title="Heatmap clustering of the top 1000 genes & Stacked bar plot of expression subtype and tumor stage", page_icon="📈")

st.markdown("# Heatmap clustering of the top 1000 genes")
st.sidebar.header("Heatmap clustering of the top 1000 genes")
st.write(
    """This visualization depicts a clustered heatmap reflecting pairwise correlations between the top 1000 genes."""
)


#### Loading necessary files (we will provide the files and the R script we used to generate the raw files)
count_matrix = pd.read_table("./Data/count_matrix.txt", delimiter = ' ')
genes_metadata = pd.read_table("./Data/gene_metadata.txt", delimiter = ' ')
sample_metadata = pd.read_csv("./Data/sample_metadata.csv", delimiter = ",")

count_matrix_just_10_or_more = count_matrix[count_matrix.sum(axis=1) >= 10]
# https://stackoverflow.com/questions/40425484/filter-dataframe-in-pandas-on-sum-of-rows

count_matrix_filtered_normalized=count_matrix_just_10_or_more.copy()
for i in list(count_matrix_filtered_normalized.columns):
    count_matrix_filtered_normalized[i] = np.log2(count_matrix_filtered_normalized[i]/np.sum(count_matrix_filtered_normalized[i])*1000000+0.001)


### https://stackoverflow.com/questions/66555842/perform-log2-normalization-over-columns-in-dataframe
### https://www.geeksforgeeks.org/how-to-get-column-names-in-pandas-dataframe/

### filter samples with only specified subtypes


sample_metadata_filtered=sample_metadata[["barcode","paper_expression_subtype","paper_Tumor.stage",'paper_expression_subtype']]
sample_metadata_filtered=sample_metadata_filtered.dropna()

### Data Processing

#1. Remove genes that have less than 10 counts across all samples
count_matrix_just_10_or_more = count_matrix[count_matrix.sum(axis=1) >= 10]
# https://stackoverflow.com/questions/40425484/filter-dataframe-in-pandas-on-sum-of-rows

genes_metadata_filtered_10_or_more=genes_metadata.loc[genes_metadata.index.isin(list(count_matrix_just_10_or_more.index)),:] ## updating gene metadata

#2. Normalize counts using log 2 counts per million (CPM)
count_matrix_filtered_normalized=count_matrix_just_10_or_more.copy()
for i in list(count_matrix_filtered_normalized.columns):
    count_matrix_filtered_normalized[i] = np.log2(count_matrix_filtered_normalized[i]/np.sum(count_matrix_filtered_normalized[i])*1000000+0.001)

### https://stackoverflow.com/questions/66555842/perform-log2-normalization-over-columns-in-dataframe
### https://www.geeksforgeeks.org/how-to-get-column-names-in-pandas-dataframe/


# 3. Filter samples with specified subtypes (Proximal Inflammatory, Proximal Profillerative, TRU, or normal non-tumor) (and remove NA's)


sample_metadata_filtered=sample_metadata[["barcode","paper_expression_subtype", "sample_type","paper_Tumor.stage"]] ### getting sample metadata


add_normal_subtype = sample_metadata_filtered['sample_type'] == "Solid Tissue Normal" 
sample_metadata_filtered.loc[add_normal_subtype, 'paper_expression_subtype'] = "Normal non-tumor" ### Adding normal non-tumor subtype as a subtype

sample_metadata_filtered=sample_metadata_filtered[["barcode","paper_expression_subtype","paper_Tumor.stage"]]
                                                  

sample_metadata_filtered=sample_metadata_filtered.dropna() ### dropping samples that do not have specified subtype
count_matrix_filtered_normalized_subtype=count_matrix_filtered_normalized.loc[:, count_matrix_filtered_normalized.columns.isin(sample_metadata_filtered["barcode"])]
# ^ Filtering for samples that have specified subtype


### 4. Getting Top 1000 most variable genes (by computing variance)
genes_metadata_filtered_10_or_more["variance"]=list(np.var(count_matrix_filtered_normalized_subtype, axis=1))

genes_names_top_1000=list(genes_metadata_filtered_10_or_more.sort_values(by=["variance"], ascending=False).index)[0:500]
### https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html
### https://www.datasciencemadesimple.com/variance-function-python-pandas-dataframe-row-column-wise-variance/

count_matrix_filtered_normalized_subtype_top_1000=count_matrix_filtered_normalized_subtype.loc[count_matrix_filtered_normalized_subtype.index.isin(genes_names_top_1000),:] ## updating gene metadata

import pandas as pd

corr_matrix = count_matrix_filtered_normalized_subtype_top_1000.T.corr()

corr_matrix.head()

import streamlit as st
##import fastcluster
import seaborn as sns
##import matplotlib
import matplotlib.pyplot as plt
#plt.style.use('ggplot')
#matplotlib.use('tkagg')

#fig = plt.figure(figsize=(10, 10))
#sns.clustermap(corr_matrix,
#                     row_cluster=True,
#                     col_cluster=True)


#st.pyplot(fig)



fig, ax = plt.subplots()


fig = sns.clustermap(corr_matrix,
                     row_cluster=True,
                     col_cluster=True)


st.pyplot(fig)


import pandas as pd
data = sample_metadata_filtered[["paper_expression_subtype","paper_Tumor.stage"]]
data_df = pd.DataFrame(data)



### Source: https://altair-viz.github.io/gallery/mosaic_with_labels.html

import altair as alt
#from vega_datasets import data

data_df = data_df.rename(columns={"paper_expression_subtype": "Expression_subtype", "paper_Tumor.stage": "Tumor_stage"})



fig2, ax2 = plt.subplots()


fig2 = sns.histplot(binwidth=0.5, x="Expression_subtype", hue="Tumor_stage", data=data_df, stat="count", multiple="stack").figure
sns.move_legend(ax2, "upper left", bbox_to_anchor=(1, 1))
st.pyplot(fig2)




