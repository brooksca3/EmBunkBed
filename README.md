
# EmBunkBed

## Introduction
embedding bunk beds for PLMs

## Setup and Installation

To get started with EmBunkBed, you need to set up a conda environment and install the necessary dependencies. Follow the steps below to prepare your environment.

### Prerequisites

- Anaconda3 (Version 2021.5)

### Creating Conda Environment

First, load Anaconda3 and create a new conda environment named `bunk_env` with Python 3.11 and the CUDA Toolkit 11.3:

```bash
module load anaconda3/2021.5
conda create -n bunk_env python=3.11 cudatoolkit=11.3
```

Activate the newly created environment:

```bash
conda activate bunk_env
```

### Installing Dependencies

Navigate to the `desformers` directory:

```bash
cd desformers
```

Before installing the required packages, you need to modify the `requirements.txt` file:

1. Comment out any lines starting with "nvidia".
2. Comment out the line `triton==2.1.0`.

Now, you can install the remaining requirements using pip:

```bash
pip install -r requirements.txt
```

## Getting Started

to-do: add to this section

---
