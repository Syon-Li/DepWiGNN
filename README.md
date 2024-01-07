# DepWiGNN

This is the repository for the paper **DepWiGNN: A Depth-wise Graph Neural Network for Multi-hop Spatial Reasoning in Text.**
To reproduce the results in the paper, first download the datasets [stepgame](https://github.com/ZhengxiangShi/StepGame), [ReSQ](https://github.com/HLR/SpaRTUN/tree/main) and [SpaRTUN]([b.com/ZhengxiangShi/StepGame](https://github.com/HLR/SpaRTUN/tree/main)https://github.com/HLR/SpaRTUN/tree/main).

- For results in Table 1, set the **dataset_used='StepGame-main'** and **GNN_comparison=False** in **Train_stepgame_spartun.py** and run the script.
- To pretrain the model on SpaRTUN, set the **dataset_used='SPARTUN'** in **Train_stepgame_spartun.py** and run the script.
- To produce the results in Table 2, set the **pretrained_model_path** in **Train_resq.py** to the path where the SpaRTUN pretrained model was stored and change **use_pretrain_spartun** to True or False accordingly, and run the script.
- To produce comparison results of different GNNs, set GNN_comparison to True and change GNN_type in **Train_stepgame_spartun.py** correspondingly (e.g.'GCN').



If you find our work useful, please cite our paper.

```
@inproceedings{li-etal-2023-depwignn,
    title = "{D}ep{W}i{GNN}: A Depth-wise Graph Neural Network for Multi-hop Spatial Reasoning in Text",
    author = "Li, Shuaiyi  and
      Deng, Yang  and
      Lam, Wai",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.428",
    doi = "10.18653/v1/2023.findings-emnlp.428",
    pages = "6459--6471",
    abstract = "Spatial reasoning in text plays a crucial role in various real-world applications. Existing approaches for spatial reasoning typically infer spatial relations from pure text, which overlook the gap between natural language and symbolic structures. Graph neural networks (GNNs) have showcased exceptional proficiency in inducing and aggregating symbolic structures. However, classical GNNs face challenges in handling multi-hop spatial reasoning due to the over-smoothing issue, i.e., the performance decreases substantially as the number of graph layers increases. To cope with these challenges, we propose a novel Depth-Wise Graph Neural Network (DepWiGNN). Specifically, we design a novel node memory scheme and aggregate the information over the depth dimension instead of the breadth dimension of the graph, which empowers the ability to collect long dependencies without stacking multiple layers. Experimental results on two challenging multi-hop spatial reasoning datasets show that DepWiGNN outperforms existing spatial reasoning methods. The comparisons with the other three GNNs further demonstrate its superiority in capturing long dependency in the graph.",
}
```
