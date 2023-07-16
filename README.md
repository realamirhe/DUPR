## Colab ready to use model
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]([https://colab.research.google.com/drive/1ICa2nh3mbflJCrJ4oKNUHWTZ9_vTJw8F?usp=sharing](https://colab.research.google.com/drive/1DnNr7nDGMad7fUVFzXuDjj-lr3_HnypO?usp=sharing))

## Instruction

### Download database

```bash
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
tar -xzf imagenette2-320.tgz
```

### Setup directories

```bash
mkdir -p ./trained_model
mkdir -p ./report
```

### Run model training

```bash
python dupr.py
```

### Local setup

- Make sure you have the following directories for local run
```bash
!mkdir -p $pascal_voc_conf.model.output
!mkdir -p $pascal_voc_conf.model_path
!mkdir -p $pretraining_conf.dataset.url
!mkdir -p $pretraining_conf.dataset.output
!mkdir -p $pretraining_conf.model_path
!mkdir -p $pretraining_conf.report_path
```
- Make sure to download [this checkpoint](https://drive.google.com/u/0/uc?id=1-c8ZJbhMX0w5FQR5-lshwK9yZAbhUGJm&export=download) and place it in `$pascal_voc_conf.model_path`
```bash
# download latest checkpoint
![ ! -f $pascal_voc_conf.model.checkpoint ] && gdown $pascal_voc_conf.model.download_url
!mkdir -p $pascal_voc_conf.model_path
!cp $pascal_voc_conf.model.checkpoint $pascal_voc_conf.model_path
```
- 