# BASTS
This is the code repository for our ICPC 2021 paper "Improving Code Summarization with Block-wise Abstract Syntax Tree Splitting".

[Arxiv Preprint](https://arxiv.org/abs/2103.07845)


## Step through
1. Code splitting. As the limitation of LFS, the splitting code data set and experiment dataset can be downloaded from [Google Drive](https://drive.google.com/drive/folders/12N-pBzlHhoIgSku7onVPvQrc_JQJw1Q9?usp=sharing) or [阿里云盘](https://www.aliyundrive.com/s/pG3c2WQaQzy).

	See the readme of the `data_preprocess` folder for details.

	**Note**: You can skip this step, directly download our processed dataset `split_test/train/valid_ast.json` and proceed to the next step.

2. Tree-LSTM_pretrain. See the readme of the `Tree-LSTM_pretrain` folder for details.
   
3. Train BASTS model. See the readme of the `BASTS` folder for details.
