<!-- # Crossmamba ([Data Mining and Knowledge Discovery](https://link.springer.com/journal/10618)) -->
# Crossmamba

## This is an offical implementation of Crossmamba: [Multivariate Time Series Forecasting Model for Cross-temporal and Cross-dimensional Dependencies with Mamba](). 

## Key Points

<p align="center">
<img src="./img/Model.jpg" alt="" align=center />
</p>

## Requirements

```
pip install torch causal-conv1d mamba-ssm numpy pandas einops tqdm
```

## Running

- You can download all datasets from https://github.com/thuml/Autoformer and put them in the `datasets/` directory.

- To get results of Crossmamba with $T=96,\tau=96$ on ETTh1 dataset, run:

```shell
python main_crossmamba.py --data ETTh1 --in_len 96 --out_len 96 --t_cycle 6 --d_model 32 --d_ff 32 --d_state 1
```
This is just an example, where t_cycle, d_model, d_ff, and d_state are adjustable hyper-parameters.

The model will be automatically trained and tested. The trained model will be saved in folder `checkpoints/` and evaluated metrics will be saved in folder `results/`.

- You can also evaluate a trained model by running:

  ```shell
  python eval_crossmamba.py --checkpoint_root ./checkpoints --setting_name Crossmamba_ETTh196__in96_seg6__dmodel-32_dstate-1_dff-32_dropout0.2_batch32___lrtype1_itr0
  ```

## Results

- Average results of multiple prediction steps with history step size $T=96$. 

- The closer the vertical axis is to 0, the higher the prediction accuracy.

- Red is our model.


<p align="center">
<img src="./img/Result.jpg" alt="" align=center />
</p>

## Acknowledgement

We appreciate the following github repo very much for the valuable code base and datasets:

https://github.com/zhouhaoyi/Informer2020

https://github.com/thuml/Autoformer

https://github.com/Thinklab-SJTU/Crossformer

https://github.com/yuqinie98/PatchTST

## Contact

If you have any questions or concerns, please contact us: YuhanLin4038@outlook.com or 2679146671@qq.com or submit an issue

## Citation

If you find this repo useful in your research, please consider citing our paper as follows:

```

```

