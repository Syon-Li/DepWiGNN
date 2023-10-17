# DepWiGNN

To reproduce the results in the paper, first download the datasets [stepgame](https://github.com/ZhengxiangShi/StepGame), [ReSQ](https://github.com/HLR/SpaRTUN/tree/main) and [SpaRTUN]([b.com/ZhengxiangShi/StepGame](https://github.com/HLR/SpaRTUN/tree/main)https://github.com/HLR/SpaRTUN/tree/main).

- For results in Table 1, set the **dataset_used='StepGame-main'** and **GNN_comparison=False** in **Train_stepgame_spartun.py** and run the script.
- To pretrain the model on SpaRTUN, set the **dataset_used='SPARTUN'** in **Train_stepgame_spartun.py** and run the script.
- To produce the results in Table 2, set the **pretrained_model_path** in **Train_resq.py** to the path where the SpaRTUN pretrained model was stored and change **use_pretrain_spartun** to True or False accordingly, and run the script.
- To produce comparison results of different GNNs, set GNN_comparison to True and change GNN_type in **Train_stepgame_spartun.py** correspondingly (e.g.'GCN').

