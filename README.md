# 利用規則
## 再現性の担保

## ユーティリティ
### sbatch_watch
sbatch を投げたジョブの状態とログ末尾を監視する簡易ウォッチャです。

```bash
./scripts/sbatch/utils/sbatch_watch.sh --gpu-ram 40 job.slurm
```

ただし、job.slurmの前に下記を追加すること
```bash
source /home/akasakam/projects/spatial_reasoning/projects/image_editing/experiments/analyse_vit/scripts/sbatch/utils/race_gurad.sh
```

オプションは `-h` で確認できます。
