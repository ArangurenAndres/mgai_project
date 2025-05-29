# MGAI Project

This project focuses on using modern game AI algorithms to generate and play Super Mario levels. It involves implementing GAN and Diffusion models for level generation and training an agent to autonomously play the game.

## Setup

You can set up the Python environment and install dependencies using the following commands:

1. For Unix/Linux/MacOS:
    ```sh
    chmod +x setup.sh
    ./setup.sh
    ```

2. For Windows:
    Either double-click the `setup.bat` file or run it from the command prompt:
    ```cmd
    setup.bat
    ```

Alternatively, you can run these commands manually:
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you are a collaborator in this project, please create a new branch and checkout before making any modifications. Merci!

## Project Overview

The project is divided into two main tasks:

1. **Level Generation**: Implement GAN and Diffusion models to create realistic Super Mario levels.
2. **Autonomous Agent**: Train an agent to autonomously play the game without prior knowledge.

## Level Generation

### Models

1. **MLP Model**: Use a simple MLP model to generate Super Mario levels. 
    - `mlp_model.py`: The model architecture for the MLP.
    - `mlp_train.py`: Simply run this training file in order to generate MLP .txt levels.

2. **GAN Model**: Uses a DCGAN model to generate Super Mario levels.
    - `dcgan_model.py`: The model architecture for the DCGAN.
    - `dcgan_train.py`: Simply run this training file in order to generate DCGAN .txt levels. 
    - `dcgan_ablation.py`: This is the file used to run the ablation study. 

    The generated .txt files are stored under the `/generated_levels` folder.
    The ablation results are stored under the `/ablation_results` folder.

- **Diffusion Models**: Implement diffusion-based models for level generation (future work).

### Level Rendering

1. **generate_levels.py**: This file renders the .txt levels into .png files and stores them under the `/generated_levels` folder. 

### Utils Folder

1. **process_data.py**: Contains forward and backward functions for data transformations.
    - Forward: `symbolic.txt` → identity (integers) → embedding
    - Backward: embedding → identity (integers) → `symbolic.txt`

    These functions are mainly used in model training with patches.

## Agent for Autonomous Playing

The agent is designed to play both real and generated levels autonomously. The training involves reinforcement learning techniques to improve the agent's performance over time.

## Requirements

To generate a `requirements.txt` file, use the following commands from the root of your project directory:
```sh
pip install pipreqs
pipreqs . --force
```

## Contributing

We welcome contributions to this project. Please ensure that you follow the setup instructions and create a new branch for your changes. For major changes, please open an issue first to discuss what you would like to change.


