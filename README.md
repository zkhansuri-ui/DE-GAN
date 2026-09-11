# Enhanced DynGAN with Dynamic Parameters and Global Positional Encoding

## Architecture Overview

This implementation has two key improvements:

### 1. **Dynamic Parameter Tuning** (ConvDynamic)
- Uses learned weight combinations to adaptively process features
- In `DecoderDynamic`: 4 different convolution outputs are weighted by learned parameters
- Includes **MixStyle**: style augmentation for domain robustness
- Helps the model learn adaptive feature combinations for different image characteristics

### 2. **Global Positional Encoding** (CoordConv2d)
- Concatenates normalized (x, y) coordinate maps to feature tensors
- Provides explicit spatial information to the decoder
- Helps preserve spatial structure in generated images
- Particularly useful for medical image enhancement

---

## Architecture Components

### File Structure
```
models/
├── dyngan_generator_dynamic.py   # Generator
├── dyngan_discriminator.py       # Discriminator
networks/
├── unet.py                       # Contains ConvDDynamic, CoordConv2d
```

### Model Hierarchy

```
EnhGANGeneratorDynamic
├── CalibrationBlock 
│   └── Channel-wise attention
└── DynamicUNetBlock 
    ├── ConvDDynamic encoders (layers 1-3)
    │   ├── ConvDDynamic: Dynamic conv + MixStyle
    │   ├── MixStyle: Adaptive instance normalization
    │   └── Learned weight combinations
    ├── Standard Conv2d encoders (layers 4-5)
    └── Decoder with CoordConv2d
        └── CoordConv2d: Global positional encoding
```

---

## How to Use

### 1. # Prepare dataset

Create "data_adr.txt" file and determine requirement as bellow:

#############################################
<p>[data] </p>
<p>data_root             = datapath </p>
<p>data_names            = config/train_name_all.txt </p>
<p>modality_postfix      = [flair] </p>
<p>file_postfix          = nii.gz </p>
#############################################
<p> Put name of each Subject ID on "train_name_all.txt"  </p> 


### 2. **Train the Dynamic EnhGAN**

**Test mode (CPU, 1 epoch):**
```bash
python train_dyngan.py --device cpu --test
```

**Full training (GPU):**

```bash
python train_dyngan.py --epochs 400 --batch-size 8 --device cuda
```

### 3. **Use for Enhancement**

After training, use the model for enhancement:

```python
from models.dyngan_generator_dynamic import DynGANGeneratorDynamic
import torch

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DynGANGeneratorDynamic().to(device)
model.load_state_dict(torch.load('path/to/data'))
model.eval()

# Inference
with torch.no_grad():
    input_tensor = torch.randn(1, 1, 128, 128).to(device)
    output = model(input_tensor)
```

---

## Key Differences from Original EnhGAN

| Feature | Original EnhGAN | Enhanced DynGAN |
|---------|-----------------|-----------------|
| Encoder | Standard Conv2d | ConvDDynamic + MixStyle |
| Decoder | Standard Conv2d | CoordConv2d (spatial encoding) |
| Parameter Tuning | None | Dynamic weight combinations |
| Style Robustness | None | MixStyle augmentation |
| Spatial Awareness | Implicit | Explicit (coordinate maps) |

---

## Performance Considerations

**Advantages:**
- ✓ Adaptive feature processing
- ✓ Better spatial information awareness
- ✓ More robust to style variations
- ✓ Preserves original CalibrationBlock benefits

**Computational Cost:**
- ~10-15% slower than original (due to CoordConv2d + dynamic operations)
- Memory usage: slightly higher (~5-10%)

---

## Training Tips

1. **Start with test mode:**
   ```bash
   python train_dyngan.py --device cpu --test
   ```

2. **Monitor losses:**
   - Generator loss should decrease
   - Discriminator loss should stay stable
   - If D_loss → 0 too quickly, generator is winning (reduce G learning rate)

3. **Save checkpoints:**
   - Models saved every 50 epochs
   - Load best checkpoint for inference

   ```
---

## References

The dynamic components are based on:
- **ConvDDynamic**: Dynamic convolution with MixStyle augmentation
- **CoordConv2d**: Coordinate-based convolutions for spatial awareness
- **MixStyle**: Domain generalisation technique for medical imaging
