# -*- coding: utf-8 -*-


### Loading necessary modules/packages
import pandas as pd
import numpy as np
import sklearn
from sklearn.decomposition import PCA 
import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="PCA using top 500 genes of lung adenocarcinoma", page_icon="📈")

st.markdown("# PCA using top 500 genes of lung adenocarcinoma")
st.sidebar.header("PCA using top 500 genes of lung adenocarcinoma")
st.write(
    """This visualization depicts the top two principal components among the top 500 most variable genes for lung adenocarcinoma"""
)


#### Loading necessary files (we will provide the files and the R script we used to generate the raw files)
count_matrix = pd.read_table("./Data/count_matrix.txt", delimiter = ' ')
genes_metadata = pd.read_table("./Data/gene_metadata.txt", delimiter = ' ')
sample_metadata = pd.read_csv("./Data/sample_metadata.csv", delimiter = ",")


### Data Processing

#1. Remove genes that have less than 10 counts across all samples
count_matrix_just_10_or_more = count_matrix[count_matrix.sum(axis=1) >= 10]

# Sources:
# https://stackoverflow.com/questions/40425484/filter-dataframe-in-pandas-on-sum-of-rows

genes_metadata_filtered_10_or_more=genes_metadata.loc[genes_metadata.index.isin(list(count_matrix_just_10_or_more.index)),:] ## updating gene metadata

#2. Normalize counts using log 2 counts per million (CPM)
count_matrix_filtered_normalized=count_matrix_just_10_or_more.copy()
for i in list(count_matrix_filtered_normalized.columns):
    count_matrix_filtered_normalized[i] = np.log2(count_matrix_filtered_normalized[i]/np.sum(count_matrix_filtered_normalized[i])*1000000+1)

### Sources:
### https://stackoverflow.com/questions/66555842/perform-log2-normalization-over-columns-in-dataframe
### https://www.geeksforgeeks.org/how-to-get-column-names-in-pandas-dataframe/


# 3. Filter samples with specified subtypes (Proximal Inflammatory, Proximal Profillerative, TRU, or normal non-tumor) (and remove NA's)


sample_metadata_filtered=sample_metadata[["barcode","paper_expression_subtype", "sample_type", "paper_Sex", "age_at_index", "race", "paper_Tumor.stage", "cigarettes_per_day"]] ### getting sample metadata

add_normal_subtype = sample_metadata_filtered['sample_type'] == "Solid Tissue Normal"  
sample_metadata_filtered.loc[add_normal_subtype, 'paper_expression_subtype'] = "Normal non-tumor" ### Adding normal non-tumor samples as a subtype
sample_metadata_filtered=sample_metadata_filtered[["barcode","paper_expression_subtype", "paper_Sex", "age_at_index", "race", "paper_Tumor.stage", "cigarettes_per_day"]] ### subsetting sample metadata
                                                  

sample_metadata_filtered=sample_metadata_filtered[sample_metadata_filtered['paper_expression_subtype'].notna()] ### dropping samples that do not have specified subtype

# Source for above:
### https://stackoverflow.com/questions/13413590/how-to-drop-rows-of-pandas-dataframe-whose-value-in-a-certain-column-is-nan


count_matrix_filtered_normalized_subtype=count_matrix_filtered_normalized.loc[:, count_matrix_filtered_normalized.columns.isin(sample_metadata_filtered["barcode"])] # Filtering for samples that have specified subtype


####################### Generating PCA###################


#### Select Tumor Subtype Options 
subtypes = list(set(sample_metadata_filtered["paper_expression_subtype"]))
subtypes_select=st.multiselect("Select Subtype", options=sample_metadata_filtered["paper_expression_subtype"].unique(),   default=subtypes)


#### Filtering sample metadata for subtypes selected

sample_metadata_filtered_select=  sample_metadata_filtered[sample_metadata_filtered["paper_expression_subtype"].isin(subtypes_select)]


### Filtering Count Matrix for samples based on subtypes selected 

count_matrix_filtered_normalized_subtype_select=count_matrix_filtered_normalized_subtype.loc[:, count_matrix_filtered_normalized_subtype.columns.isin(sample_metadata_filtered_select["barcode"])]


### 4. Getting Top 1000 most variable genes (by computing variance)
genes_metadata_filtered_10_or_more["variance"]=list(np.var(count_matrix_filtered_normalized_subtype_select, axis=1))

genes_names_top_1000=list(genes_metadata_filtered_10_or_more.sort_values(by=["variance"], ascending=False).index)[0:500]

### Sources:
### https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html
### https://www.datasciencemadesimple.com/variance-function-python-pandas-dataframe-row-column-wise-variance/


count_matrix_filtered_normalized_subtype_top_1000=count_matrix_filtered_normalized_subtype_select.loc[count_matrix_filtered_normalized_subtype_select.index.isin(genes_names_top_1000),:] ## Filtering count matrix for top 1000 gnes

genes_metadata_top_1000 = genes_metadata_filtered_10_or_more.loc[genes_metadata_filtered_10_or_more.index.isin(genes_names_top_1000),:] ## updating gene metadata


### Creating PCA


#### Running PCA (Below code based on https://www.reneshbedre.com/blog/principal-component-analysis.html)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

count_matrix_filtered_normalized_subtype_st =  StandardScaler().fit_transform(count_matrix_filtered_normalized_subtype_top_1000) ### Scaling Count Matrix

count_matrix_filtered_normalized_subtype_st=pd.DataFrame(count_matrix_filtered_normalized_subtype_st, 
             columns=count_matrix_filtered_normalized_subtype_top_1000.columns,
             index=count_matrix_filtered_normalized_subtype_top_1000.index)



PCA_of_subtypes = PCA().fit(count_matrix_filtered_normalized_subtype_st) #### Run PCA on scaled count matrix


output_PCA_sample_PCS = PCA_of_subtypes.components_ ### Getting loadings
total_PCs = PCA_of_subtypes.n_features_ ### geting number of PCS
total_PCs = ["PC"+str(i) for i in list(range(1, total_PCs+1))]
output_PCA_sample_PCS_df = pd.DataFrame.from_dict(dict(zip(total_PCs, output_PCA_sample_PCS)))
output_PCA_sample_PCS_df['samples'] = count_matrix_filtered_normalized_subtype_st.columns.values
output_PCA_sample_PCS_df = output_PCA_sample_PCS_df.set_index('samples')


output_PCA_sample_PCS_df["sample_id"]=list(output_PCA_sample_PCS_df.index)
add_normal_subtype = sample_metadata_filtered_select['paper_expression_subtype'] == "Solid Tissue Normal" 
sample_metadata_filtered_select.loc[add_normal_subtype, 'paper_expression_subtype'] = "Normal non-tumor" ### Adding normal non-tumor subtype as a subtype
output_PCA_sample_PCS_df=output_PCA_sample_PCS_df.join(sample_metadata_filtered_select.set_index('barcode'), on='sample_id') ### combining sample metadata with PCA data

output_PCA_sample_PCS_df.columns=total_PCs + ["sample_id", "Sample Subtype", "Sex", "Age", "Race", "Tumor Stage", "Num of Cigarettes per Day"]



### Fixing NA's of other key variables to deal with tooltip issues

for i in  ["Sample Subtype", "Sex", "Age", "Race", "Tumor Stage", "Num of Cigarettes per Day"]:

        for j in list(range(len(output_PCA_sample_PCS_df[i]))):

            if output_PCA_sample_PCS_df[i][j] == "[unknown]":
                output_PCA_sample_PCS_df[i][j] = "Not Reported"
            
            if output_PCA_sample_PCS_df[i][j] == "":
                output_PCA_sample_PCS_df[i][j] = "Not Reported"

            if output_PCA_sample_PCS_df[i][j] == "[Not Available]":
                output_PCA_sample_PCS_df[i][j] = "Not Reported"
        


output_PCA_sample_PCS_df["Num of Cigarettes per Day"][pd.isna(output_PCA_sample_PCS_df["Num of Cigarettes per Day"])]="Not Reported"
output_PCA_sample_PCS_df["Age"][pd.isna(output_PCA_sample_PCS_df["Age"])]="Not Reported"




#output_PCA_sample_PCS_df["# of Cigarettes per Day"]=list(output_PCA_sample_PCS_df["Num. of Cigarettes per Day"])




PCA_chart=alt.Chart(output_PCA_sample_PCS_df).mark_circle(size=60).encode(
    x=alt.X('PC1:Q',  title=("PC1"+ " (" + str(round(PCA_of_subtypes.explained_variance_ratio_[0],4)*100) + "%)")),
    y=alt.Y('PC2:Q',  title=("PC2"+ " (" + str(round(PCA_of_subtypes.explained_variance_ratio_[1],4)*100) + "%)")),
    color=alt.Color("Sample Subtype:N",  title="Subtype"), ### Color by Subtype 
    #  https://vega.github.io/vega/docs/schemes/

    tooltip=['Sample Subtype:N', 'Age:N', 'Tumor Stage:N', 'Sex:N', "Race:N", "Num of Cigarettes per Day:N"]
).interactive()

### Source:
# https://altair-viz.github.io/gallery/scatter_tooltips.html



st.altair_chart(PCA_chart, use_container_width=True)












