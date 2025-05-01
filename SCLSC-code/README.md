# Supervised contrastive learning for single-cell annotation

In this work, we developed a novel modeling formalism for cell type annotation with a supervised contrastive learning method, named SCLSC (Supervised Contrastive Learning for Single Cell). 
Different from the previous usage of contrastive learning in single cell data analysis, we employed the contrastive learning for instance-type pairs instead of instance-instance pairs. 
Mores specifically, in the cell type annotation task, the contrastive learning is applied to learn cell and cell type representation that render cells of the same type to be clustered in the new embedding space. 
Through this approach, the knowledge derived from annotated cells is transferred to the feature representation for scRNA-seq data. 


![](figure/overall_pipeline.png)

## Enviroments and Package dependency

- python 3
- Pytorch 1.11 
- scanpy
- anndata

## Dataset

### Data preprocessing

The datasets underwent preprocessing to eliminate cells with high mitochondrial gene expression (more than 5 percents of the cell total count), cells with minimal gene expression (number of genes per cell < 200), and genes that were only detected in a small number of cells (number of cells that expressed the gene < 3). 
Subsequently, We selected 2000 highly variable genes (HGV) using analytic Pearson residuals implemented in Scanpy package. 
Following this, we normalized the count of each cell to 10,000 counts and applied a $log(x+1)$ transformation. The resulting dataset was then divided into training, validation, and test sets with a ratio of $8:1:1$. All of the preprocessing steps were performed using Scanpy package. 
The summary of the dataset, reference, and download link were provided as follows.

### Data download link

- **PBMC**: [Fresh 68k PBMCs Donor A](https://www.10xgenomics.com/resources/datasets/fresh-68-k-pbm-cs-donor-a-1-standard-1-1-0)
- **Pancreas**: [Figshare Pancreas Link](https://figshare.com/ndownloader/files/22891151)
- **Thymus**: [Thymus link](https://zenodo.org/record/5500511)
- **Lung**: [Figshare Lung Link](https://figshare.com/ndownloader/files/24539942)
- **CeNGEN**: [CeNGEN Data on GitHub](https://github.com/Munfred/wormcells-data/releases/download/taylor2020/taylor2020.h5ad)
- **Zebrafish**: [Zebrafish Dataset](https://ndownloader.figshare.com/files/24566651?private_link=e3921450ec1bd0587870)
- **Dengue dataset**: [EBI Dengue Dataset](https://www.ebi.ac.uk/gxa/sc/experiments/E-MTAB-9467/downloads)



## Learning cell and cell type embedding
```
ENCODER="MLP"
MARGIN=1.0
BATCH_SIZE=256
EPOCH=100
GENE_SET="hvg"

DATADIR=PATH_of_DATA_DIRECTORY
MODEL_SAVE_DIR=PATH_of_MODEL_SAVE_DIRECTORY
DATASET="zebrafish_all"
OUTPUT_DIR=PATH_of_OUTPUT_DIRECTORY

echo "Processing $DATASET"

python sc2l_main.py --encoder $ENCODER  --dataset_name $DATASET --data_dir $DATADIR --model_dir $MODEL_SAVE_DIR \
--gene_set ${GENE_SET} --margin $MARGIN  --output_dir $OUTPUT_DIR --epoch $EPOCH --batch_size $BATCH_SIZE

```

## Cell type annotation

```
ENCODER="MLP"
MARGIN=1.0
BATCH_SIZE=256
EPOCH=100
GENE_SET="hvg"



DATASET="pancreas_all"

DATADIR=PATH_of_DATA_DIRECTORY

MODEL_DIR=PATH_of_SAVED_MODEL_DIRECTORY
KNN_K=10
echo "Processing $DATASET"


python sc2l_test_knn.py --encoder $ENCODER  --dataset_name $DATASET --data_dir $DATADIR --model_dir $MODEL_DIR \
--gene_set ${GENE_SET} --margin $MARGIN  --knn_k $KNN_K --epoch $EPOCH
```

## Manuscript
https://www.biorxiv.org/content/10.1101/2023.08.08.552379v1



