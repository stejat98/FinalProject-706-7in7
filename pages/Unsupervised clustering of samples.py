# -*- coding: utf-8 -*-


### Loading necessary modules/packages
import pandas as pd
import numpy as np
import sklearn
from sklearn.decomposition import PCA 
import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="User-specified stratification of lung adenocarcinoma gene expression signatures by variable of interest", page_icon="📈")

st.markdown("# User-specified stratification by variable of interest of the top 1000 genes across lung adenocarcinoma samples")
st.sidebar.header("User-specified stratification by variable of interest of the top 1000 genes across lung adenocarcinoma samples")
st.write(
    """This visualization depicts the top 1000 genes across lung adenocarcinoma samples stratified by a user-specified variable (e.g., age group)."""
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
subtypes_select=subtypes


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








############# Heatmap + Boxplot ##############


#### Getting Top 10 Variable Genes for the boxplot


genes_names_top_10_symbol=list(genes_metadata_filtered_10_or_more.sort_values(by=["variance"], ascending=False)["gene_name"])[0:10]

genes_names_top_10_ensembl=list(genes_metadata_filtered_10_or_more.sort_values(by=["variance"], ascending=False).index)[0:10]

### Source:
### https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html
### https://www.datasciencemadesimple.com/variance-function-python-pandas-dataframe-row-column-wise-variance/





sample_metadata_filtered_select_arranged_fixed_column_names=sample_metadata_filtered 
sample_metadata_filtered_select_arranged_fixed_column_names.columns=np.asarray(["Patient", "Sample Subtype", "Sex", "Age", "Race", "Tumor Stage", "Cigarettes Per Day"]) ### Fixing Column Names of Sample Metadaa


###### Creating Age Groups based on percentiles 

first_group=np.percentile(sample_metadata_filtered_select_arranged_fixed_column_names["Age"].dropna(), [0,25])
second_group=np.percentile(sample_metadata_filtered_select_arranged_fixed_column_names["Age"].dropna(), [26,50])
third_group=np.percentile(sample_metadata_filtered_select_arranged_fixed_column_names["Age"].dropna(), [51,75])
fourth_group=np.percentile(sample_metadata_filtered_select_arranged_fixed_column_names["Age"].dropna(), [76,100])


sample_metadata_filtered_select_arranged_fixed_column_names["Age_Group"]=""
sample_metadata_filtered_select_arranged_fixed_column_names.index=list(range(len(sample_metadata_filtered_select_arranged_fixed_column_names)))

####### Creating a column of Age Groups 

for i in list(range(len(sample_metadata_filtered_select_arranged_fixed_column_names["Age_Group"]))):

    if  pd.isna(list(sample_metadata_filtered_select_arranged_fixed_column_names["Age"])[i]): # https://www.w3schools.com/python/pandas/ref_df_isna.asp
        continue

    if (list(sample_metadata_filtered_select_arranged_fixed_column_names["Age"])[i]) >= first_group[0] and (list(sample_metadata_filtered_select_arranged_fixed_column_names["Age"])[i])  <= first_group[1]: ### if age number falls into this category 
        sample_metadata_filtered_select_arranged_fixed_column_names["Age_Group"][i]=str(int(first_group[0])) + "-" +  str(int(first_group[1]))

    if (list(sample_metadata_filtered_select_arranged_fixed_column_names["Age"])[i])  >= second_group[0] and (list(sample_metadata_filtered_select_arranged_fixed_column_names["Age"])[i])  <= second_group[1]:
        sample_metadata_filtered_select_arranged_fixed_column_names["Age_Group"][i]=str(int(second_group[0])) + "-" +  str(int(second_group[1]))

    if (list(sample_metadata_filtered_select_arranged_fixed_column_names["Age"])[i]) >= third_group[0] and (list(sample_metadata_filtered_select_arranged_fixed_column_names["Age"])[i])  <= third_group[1]:
        sample_metadata_filtered_select_arranged_fixed_column_names["Age_Group"][i]=str(int(third_group[0])) + "-" +  str(int(third_group[1]))

    if (list(sample_metadata_filtered_select_arranged_fixed_column_names["Age"])[i])  >= fourth_group[0] and (list(sample_metadata_filtered_select_arranged_fixed_column_names["Age"])[i])  <= fourth_group[1]:
        sample_metadata_filtered_select_arranged_fixed_column_names["Age_Group"][i]=str(int(fourth_group[0])) + "-" +  str(int(fourth_group[1]))





### Similar to above, creating a group for smoking activity based on number of cigarettes smoked per day using percentiles

first_group_smoking=np.percentile(sample_metadata_filtered_select_arranged_fixed_column_names["Cigarettes Per Day"].dropna(), [0,25])
second_group_smoking=np.percentile(sample_metadata_filtered_select_arranged_fixed_column_names["Cigarettes Per Day"].dropna(), [26,50])
third_group_smoking=np.percentile(sample_metadata_filtered_select_arranged_fixed_column_names["Cigarettes Per Day"].dropna(), [51,75])
fourth_group_smoking=np.percentile(sample_metadata_filtered_select_arranged_fixed_column_names["Cigarettes Per Day"].dropna(), [76,100])


sample_metadata_filtered_select_arranged_fixed_column_names["# of Cigs per Day"]=""
sample_metadata_filtered_select_arranged_fixed_column_names.index=list(range(len(sample_metadata_filtered_select_arranged_fixed_column_names)))

for i in list(range(len(sample_metadata_filtered_select_arranged_fixed_column_names["# of Cigs per Day"]))):

    if  pd.isna(list(sample_metadata_filtered_select_arranged_fixed_column_names["Cigarettes Per Day"])[i]): # https://www.w3schools.com/python/pandas/ref_df_isna.asp
        continue

    if (list(sample_metadata_filtered_select_arranged_fixed_column_names["Cigarettes Per Day"])[i]) >= first_group_smoking[0] and (list(sample_metadata_filtered_select_arranged_fixed_column_names["Cigarettes Per Day"])[i])  <= first_group_smoking[1]:
        sample_metadata_filtered_select_arranged_fixed_column_names["# of Cigs per Day"][i]="(0,1]"

    if (list(sample_metadata_filtered_select_arranged_fixed_column_names["Cigarettes Per Day"])[i])  >= second_group_smoking[0] and (list(sample_metadata_filtered_select_arranged_fixed_column_names["Cigarettes Per Day"])[i])  <= second_group_smoking[1]:
        sample_metadata_filtered_select_arranged_fixed_column_names["# of Cigs per Day"][i]="(1,2]"

    if (list(sample_metadata_filtered_select_arranged_fixed_column_names["Cigarettes Per Day"])[i]) >= third_group_smoking[0] and (list(sample_metadata_filtered_select_arranged_fixed_column_names["Cigarettes Per Day"])[i])  <= third_group_smoking[1]:
        sample_metadata_filtered_select_arranged_fixed_column_names["# of Cigs per Day"][i]="(2,3]"

    if (list(sample_metadata_filtered_select_arranged_fixed_column_names["Cigarettes Per Day"])[i])  >= fourth_group_smoking[0] and (list(sample_metadata_filtered_select_arranged_fixed_column_names["Cigarettes Per Day"])[i])  <= fourth_group_smoking[1]:
        sample_metadata_filtered_select_arranged_fixed_column_names["# of Cigs per Day"][i]=">3"



sample_metadata_filtered_select_arranged_fixed_column_names=sample_metadata_filtered_select_arranged_fixed_column_names[["Patient", "Sample Subtype", "Sex", "Age_Group", "Race", "Tumor Stage", "# of Cigs per Day"]] ### Subsetting Sample Metadata


## Selecting Variable of Interest to show in heatmap and boxplot

var_of_int=st.selectbox("Select Variable of Interest", options=list(sample_metadata_filtered_select_arranged_fixed_column_names.columns)[1:], index=0)


### Removing NA's or unknown labels 


for i in list(range(len(sample_metadata_filtered_select_arranged_fixed_column_names[var_of_int]))):
    if sample_metadata_filtered_select_arranged_fixed_column_names[var_of_int][i] == "[unknown]":
        sample_metadata_filtered_select_arranged_fixed_column_names[var_of_int][i] = float("nan")
    
    if sample_metadata_filtered_select_arranged_fixed_column_names[var_of_int][i] == "":
        sample_metadata_filtered_select_arranged_fixed_column_names[var_of_int][i] = float("nan")

    if sample_metadata_filtered_select_arranged_fixed_column_names[var_of_int][i] == "[Not Available]":
        sample_metadata_filtered_select_arranged_fixed_column_names[var_of_int][i] = float("nan")


### Source:
### https://stackoverflow.com/questions/944700/how-can-i-check-for-nan-values


### After selecting varible of interest, drop samples that have NA for that variable of interest 
sample_metadata_filtered_select_arranged_fixed_column_names_drop_na = sample_metadata_filtered_select_arranged_fixed_column_names[sample_metadata_filtered_select_arranged_fixed_column_names[var_of_int].notna()]


### updating samples from above
count_matrix_filtered_normalized_subtype_top_1000_boxplot=count_matrix_filtered_normalized_subtype_top_1000.loc[:, count_matrix_filtered_normalized_subtype_top_1000.columns.isin(sample_metadata_filtered_select_arranged_fixed_column_names_drop_na["Patient"])]

### ordering column sample names to be same order as sample metadata
count_matrix_filtered_normalized_subtype_top_1000_boxplot=count_matrix_filtered_normalized_subtype_top_1000_boxplot[sample_metadata_filtered_select_arranged_fixed_column_names_drop_na["Patient"]]




########## Creating Heatmap ##########

heatmap_df=pd.DataFrame((count_matrix_filtered_normalized_subtype_top_1000_boxplot))
heatmap_df.index=genes_metadata_top_1000["gene_name"]

sample_metadata_filtered_select_arranged = sample_metadata_filtered_select_arranged_fixed_column_names_drop_na.sort_values(by=[var_of_int]) ### sorting values according to variable of interest 

# Source:
# https://stackoverflow.com/questions/13148429/how-to-change-the-order-of-dataframe-columns

heatmap_df=heatmap_df[sample_metadata_filtered_select_arranged["Patient"]] ### arranging samples to have same order as asmple metadata


heatmap_df_scaled = heatmap_df.apply(lambda o: (o-o.mean())/o.std(), axis = 1)  ### Scaling each gene across all samples (scaling by row)

#  Source: 
#https://www.python-graph-gallery.com/94-use-normalization-on-seaborn-heatmap



### Coloring each sample based on variable of interest (inspired by https://seaborn.pydata.org/generated/seaborn.clustermap.html)


variable_colors = ['blue', 'orange', 'green',    'red', 'purple', 'brown', 'pink', 'grey'] 

## Colors inspired bby:
 # https://stackoverflow.com/questions/54015895/altair-default-color-palette-colors-in-hex
 # # https://seaborn.pydata.org/tutorial/color_palettes.html


coloring_variable_of_int = dict(zip(set(sample_metadata_filtered_select_arranged[var_of_int]), np.random.choice(variable_colors, len(set(sample_metadata_filtered_select_arranged[var_of_int])), replace=False).tolist()))
coloring_variable_of_int_columns = pd.Series(sample_metadata_filtered_select_arranged[var_of_int]).map(coloring_variable_of_int) 
coloring_variable_of_int_columns.index=sample_metadata_filtered_select_arranged["Patient"]
coloring_variable_of_int_columns.name=var_of_int


### Selecting Option Whether or not to cluster samples (cluster columns in the case of the heatmap)
cluster_samples_yes_or_no=st.radio("Cluster Samples?", options=["No", "Yes"], index=0)

column_clustering=False
if cluster_samples_yes_or_no== "Yes":
    column_clustering=True


### plot heatmap with clustering 
import seaborn as sns
import matplotlib.pyplot as plt


heatmap=sns.clustermap(heatmap_df_scaled, cmap=sns.color_palette("coolwarm", as_cmap=True), vmin=-2, vmax=2, col_cluster=column_clustering, col_colors=coloring_variable_of_int_columns,dendrogram_ratio=0.05) ### https://stackoverflow.com/questions/34706845/change-xticklabels-fontsize-of-seaborn-heatmap
heatmap.fig.subplots_adjust(right=0.7) ### https://stackoverflow.com/questions/62882084/seaborn-clustermap-legend-overlap-with-figure 
heatmap.ax_cbar.set_position((1, .3, .03, .4)) ### https://stackoverflow.com/questions/62882084/seaborn-clustermap-legend-overlap-with-figure
heatmap.fig.suptitle('')  # https://stackoverflow.com/questions/49254337/how-do-i-add-a-title-to-a-seaborn-clustermap


ax = heatmap.ax_heatmap  # https://stackoverflow.com/questions/32868423/plot-on-top-of-seaborn-clustermap
ax = heatmap.ax_heatmap  # https://stackoverflow.com/questions/32868423/plot-on-top-of-seaborn-clustermap
ax.set(xlabel='', ylabel='Genes')
ax.set(xticklabels=[])   # https://stackoverflow.com/questions/58476654/how-to-remove-or-hide-x-axis-labels-from-a-seaborn-matplotlib-plot
ax.set(yticklabels=[])  
ax.tick_params(right=False, bottom=False) #https://www.tutorialspoint.com/how-to-remove-the-axis-tick-marks-on-a-seaborn-heatmap

ax=ax.figure


### Adding column legends to heatmap to indicate labels for samples based on variable of interest


from matplotlib.patches import Patch 
 
coloring_variable_of_int_1 = sorted(coloring_variable_of_int)
sorted_coloring_variable_of_int = {key:coloring_variable_of_int[key] for key in coloring_variable_of_int_1} # https://favtutor.com/blogs/python-sort-dictionary
 
handles = [Patch(facecolor=sorted_coloring_variable_of_int[i]) for i in coloring_variable_of_int_1]

plt.legend(handles,
          sorted_coloring_variable_of_int, 
          title= var_of_int,
           bbox_to_anchor=(1.15, 1), 
          bbox_transform=plt.gcf().transFigure,
          labelcolor = "black", 
          loc='upper right')


### Sources:
###https://stackoverflow.com/questions/62473426/display-legend-of-seaborn-clustermap-corresponding-to-the-row-colors
### https://stackoverflow.com/questions/59792534/how-can-i-label-the-clusters-in-sns-clustermap

st.pyplot(ax)




################## Creating Boxplot of 10 most variable genes across sample subtypes #####################


count_matrix_filtered_normalized_subtype_top_10_boxplot=count_matrix_filtered_normalized_subtype_top_1000_boxplot.loc[count_matrix_filtered_normalized_subtype_top_1000_boxplot.index.isin(genes_names_top_10_symbol),:] ## filtering count matrix for top 10 most variable genes

count_matrix_filtered_normalized_subtype_top_10_boxplot_transposed=count_matrix_filtered_normalized_subtype_top_10_boxplot.T ### transpoising count matrix


genes_metadata_top_10= genes_metadata_filtered_10_or_more.loc[genes_metadata_filtered_10_or_more.index.isin(genes_names_top_10_ensembl),:] ## updating gene metadata for top 10 genes


count_matrix_filtered_normalized_subtype_top_10_boxplot_transposed["Sample"] = count_matrix_filtered_normalized_subtype_top_10_boxplot_transposed.index ### Setting row names as sample ids

count_matrix_filtered_normalized_subtype_top_10_boxplot_transposed["Variable_of_Int"]=list(sample_metadata_filtered_select_arranged_fixed_column_names_drop_na[var_of_int]) #### adding column based on variable of interest


count_matrix_filtered_normalized_subtype_top_10_boxplot_transposed_melted=count_matrix_filtered_normalized_subtype_top_10_boxplot_transposed.melt(["Sample", "Variable_of_Int"]) ### melting dataframe for boxplot


### boxplot based on https://altair-viz.github.io/gallery/boxplot.html

#alt.themes.enable('opaque')

boxplot_altair=alt.Chart(count_matrix_filtered_normalized_subtype_top_10_boxplot_transposed_melted).mark_boxplot(extent='min-max').encode(
    x=alt.X("Variable_of_Int:N", title="",  axis=alt.Axis(labels=False, ticks=False), stack=None),
    y=alt.Y('value:Q', title="Normalized Expression"),
    color=alt.Color("Variable_of_Int:N", title=var_of_int),
    #column=alt.Column("gene_name:N", title="Top 10 Most Variable Genes from the Heatmap",  header=alt.Header(titleColor="white", labelColor="white")), #https://github.com/altair-viz/altair/issues/2197
    facet=alt.Column('gene_name:N', columns=5, title="Top 10 Most Variable Genes from the Heatmap")  #https://github.com/altair-viz/altair/issues/2197 and # https://stackoverflow.com/questions/50164001/multiple-column-row-facet-wrap-in-altair
).properties(
    width=200, height=300).resolve_scale(y='independent').configure_axisLeft(
  labelColor='black',
  titleColor='black'
).configure_axisRight(
  labelColor='black',
  titleColor='black'
).configure_axisTop(
  labelColor='black',
  titleColor='black'
).configure(background="white")

# Sources:
# https://stackoverflow.com/questions/66624408/shared-axis-labels-with-independent-scale
# https://stackoverflow.com/questions/50164001/multiple-column-row-facet-wrap-in-altair
  


st.altair_chart(boxplot_altair, use_container_width=False)
