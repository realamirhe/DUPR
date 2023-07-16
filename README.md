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
