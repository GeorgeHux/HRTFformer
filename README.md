# HRTFformer: A Spatially-Aware Transformer for Individual HRTF Upsampling in Immersive Audio Rendering

[Arxiv](https://arxiv.org/abs/2510.01891)

## Project Structure

```text
configs/              Configuration objects for data, training, and model hyperparameters
data/                 HRTF dataset loaders, transforms, and preprocessing utilities
evaluation/           Objective evaluation scripts for LSD, localization, ILD, and ITD
model/                HRTFformer model components
trainer/              Training, testing, losses, metrics, and model factory utilities
main.py               Command-line entry point
```

## Model

The active model is created in `trainer/utils.py` through `get_model(config)`:

```python
AutoEncoder(Encoder, encoder_config, TransConvDecoder, decoder_config)
```

The encoder combines transformer blocks with downsampling layers. The decoder reconstructs high-resolution outputs with transformer-guided transposed-convolution blocks.

## Requirements

Install the Python dependencies needed by your data loader and evaluation workflow. The main training stack uses:

- Python 3.10+
- PyTorch
- NumPy
- SciPy
- Matplotlib
- pandas
- einops
- sofar
- netCDF4

Optional evaluation scripts may also require MATLAB Engine for Python, AMT, and `spatialaudiometrics`.

## Usage

Update paths and hyperparameters in `configs/config.py` before running. In particular, set the dataset directory, output directory, device, and HRTF loader for your machine.

The SONICOM HRTF dataset can be downloaded from [here](https://transfer.ic.ac.uk:9090/#/2022_SONICOM-HRTF-DATASET/).

Preprocess data:

```bash
python main.py preprocess -r True -d Sonicom
```

Train:

```bash
python main.py train -r True -d Sonicom
```

Test and evaluate:

```bash
python main.py test -r True -d Sonicom
```

## Outputs

Training writes logs, plots, and checkpoints under the configured output paths. Testing writes reconstructed HRTFs and evaluation artifacts next to the selected checkpoint.

## Notes

Large datasets, generated checkpoints, reconstructed HRTFs, SOFA files, and pickle artifacts are intentionally ignored by Git. Keep those files outside the repository or regenerate them from the configured data paths.

## Acknowledgements

Parts of the code are borrowed from the following repositories:

- [GEP-GAN](https://github.com/ahogg/HRTF-upsampling-with-a-generative-adversarial-network-using-a-gnomonic-equiangular-projection)

This study was made possible by support from [SONICOM](https://www.sonicom.eu/), a project that has received funding from the European Union's Horizon 2020 research and innovation program under grant agreement No. 101017743.

## Citation

If you find this code useful for your research, please consider citing the following paper:

```bibtex
@article{hu2025hrtfformer,
  title={HRTFformer: A Spatially-Aware Transformer for Personalized HRTF Upsampling in Immersive Audio Rendering},
  author={Hu, Xuyi and Li, Jian and Zhang, Shaojie and Goetz, Stefan and Picinali, Lorenzo and Akan, Ozgur B and Hogg, Aidan OT},
  journal={arXiv preprint arXiv:2510.01891},
  year={2025}
}
```
