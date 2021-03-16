# Tree-LSTM pretrained
Tree-LSTM pretrain for BASTS

## Requirements
* Hardwares: a machine with two Intel(R) Xeon(R) CPU E5-2678 v3 @ 2.50GHz, 256 GB main memory and a GeForce RTX 2080 Ti graphics card
* OS: Ubuntu 18.04
* Packages:
    * python 3.7.4
    * tensorflow 2.0-gpu

## Step through
1. Construct sentence pairs and vocab of Tree-LSTM pre-training model
   
   `python 1.constructDataPair.py`     
2. Train the model
   
    `python 2.train.py`
3. Evaluation model
    
   `python 3.evaluate.py`
3. To construct the training set, valid set, and test set that need to be put into the pre-trained Tree-LSTM model.

   `python 4.constructData.py` 
4. Put the constructed data set into the pre-training model to get the overall AST hidden state corresponding to each code(continue to fine-tune in BASTS)
   
    `python 5.getTransformerAst.py`