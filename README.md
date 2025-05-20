# MGAI project


## Create requirements.txt file 




# From the root of your project directory:



```sh
pip install pipreqs

pipreqs . --force



```



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

2. If you are a collaborator in this project before doing any modifications create a new branch and checkout :) merci 


## Project

The following project is divided in two main taks:

1. Implement GAN and Diffusion models to create realistic Super Mario levels 

2. Train agent to autonomously play the game without previous knowledge


## Level generation

### utils folder

1. process_data.py  (Forward and backward functions to apply the following data transformations)

- (Forward) symbolic.txt --> identity (integers) --> embedding 
- (Backward) embedding --> identity (integers) --> symbolic.txt 

**These functions will be mainly implemented in model training using patch**

2. render_level.py (Use this function if you want to map a txt to rendered png using the corresponding tiles (An))

```sh
run_render(processor,symb_file=symb_file, symb_name=symb_test, save_folder=save_folder)

```




## Agent for autonomous playing both real and generated levels