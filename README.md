# Repository for the paper "Graph Denoising Networks: A Deep Learning Framework for Equity Portfolio Construction"



## Requirements
```
Python 3.9
torch
torch_geometric
```
## Overview
The ```model.py``` and ```train_test.py``` files provide the raw code needed to create and then train/test the GDN model.

The ```example.ipynb``` notebook provides a simple notebook that shows an implimentation of the GDN we use.

The provided datasets (YET TO BE UPLOADED) within the ```finance_data``` folder are in the form date_numDays_lbSize_correlationType_useOfSelfLoops. For this data we do not manually add self loops at any point as the GCNConv function does this automatically.

The ```GDN_outputs.py``` and ```GCN_outputs.py``` folders contain the respective model outputs $\gamma$ for each of the datasets. The files take the form model_numDays_featureSize_GDNNumEpochs_DDPMNumEpochs_gamma_delta_GDNLearningRate_DDPMLearningRate_DDPMlb_traininglb_dataName. The ```GDN_weights.py``` and ```GCN_weights.py``` folders contain the equivalent files but now for model weights at the end of the respective 100 day training period.

Finally, the ```analysis.ipynb``` notebook provides the code to reproduce Figure 4, Figure 5, the Sharpe Ratio tests and the t-tests.
