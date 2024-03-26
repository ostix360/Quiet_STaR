# Quiet Star Algorithm Implementation

This is a fork of the [Quiet-STaR](https://github.com/ezelikman/quiet-star) project.

The goal is to improve the efficiency by adding first of all a head to choose weather to think or not.


## Table of Contents

- [Installation](#installation)
- [Working with the model](#working-with-the-model)
  - [Training](#training)
  - [Inference](#inference)
- [Contributing](#contributing)
- [License](#license)

## Installation

This project is written in Python and uses pip for package management.

Using a virtual environment is recommended.

First install the required packages:

```bash
pip install -r requirements.txt
```

Then install pytorch and unsloth:
```bash
pip3 install torch==2.2.0 --upgrade --force-reinstall --index-url https://download.pytorch.org/whl/cu121
pip install "unsloth[cu121-torch220] @ git+https://github.com/unslothai/unsloth.git"
```

## Working with the model

### Training

To train the model, because of hardware limitations, we use [unsloth](https://github.com/unslothai/unsloth) and [galore](https://github.com/jiaweizzhao/GaLore) to train the model.


The training script is in [quiet-star-train.py](quiet-star-train.py). 

Work in progress: Still doesn't work.


This section will be updated soon.

### Inference

This section will be updated soon.

## Contributing

To contribute please open a pull request. We will review it as soon as possible.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details but the original project is licensed under the Apache License 2.0 (see [LICENSE-APACHE](LICENSE-APACHE)).


