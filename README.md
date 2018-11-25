# Tennis

## Environment

The environment used for this project is a variant of the  [Tennis Environment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) agent by Unity.

![image ](tennis.png)

The environment consists of two agents controlling tennis rackets. The aim for each agent is to hit the ball over the net, on to the other agents side of the court, and colaborate with each other to prevent the ball from hitting the floor or going out of bounds for as long as possible.

The available *state-space* for each agent consists of a vector with 24 variables. This consists of 8 values encoding racket and ball information for the current step, and previous 2 timesteps. Notice that there is actually some duplicate racket information values. Having information over multiple timesteps allows the agent to have information about the movement of objects in the environment.

![image state](states.jpg)

The *action-space* for each agent are two continuous variables:

1. Amount of movement in x axis (towards/away from net)
2. Amount to jump by

A reward of +0.1 is given to an agent if it hits the ball over the net, and within bounds of the court. It receives a reward of -0.01 if it lets the ball hit the ground or hits the ball out of bounds.

For each episode, the rewards for each agent are added up (without discounting), and the maximum of the two is given as the score of the episode. The task is considered solved if the rolling average score over the past 100 timesteps is at least +0.5.



## Setup Libraries

You will need to have `Python >= 3.5` and `pip` installed, plus some aditional libraries such as:

- matplotlib
- numpy>=1.11.0
- torch==0.4.0
- unityagents

Before continuing, it is recomended that you create a new [python virtualenv](https://virtualenv.pypa.io/en/latest/) before continuing. You can install the dependencies by cloning this repository and running the following  command lines on Linux (once you are in the desired virtualenv):

```sh
# Clone repo and move into installation directory
git clone https://github.com/ronrest/rlnd_p3
cd rlnd_p3/python

# install dependencies
pip install .

# Go back to root directory of repo
cd ../
```


## Setup environment

You will need to download the environment binary files for your operating system.

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Linux Headless: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip)
    - For training on the cloud (eg Amazon AWS), or training locally without a GUI.
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)


If you want to train on the cloud, you can either use the Linux Headless version, or you can [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)).

Make sure that you unzip the downloaded file and take note of where it is stored. The following command lines can be used in Linux for downloading the headless (No GUI) version environment and unzipping it.

```sh
# Get the headless Tennis environment.
wget -c https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip
unzip Tennis_Linux_NoVis.zip
rm Tennis_Linux_NoVis.zip
```


## Train

All the training was done in the [train.ipynb](train.ipynb) notebook. Make sure you update the following variable values in the `settings` section of the notebook:

- `ENV_FILE`: update this to the file path to the environment file you donwloaded
- `MODEL_NAME` This will store all the saved snapshots, log files, and training curve plots associated with your model in a separate subdirectory. These things will be stored in `models/{MODEL_NAME}/`
- Feel free to experiment with the other settings as well to see what results you get.

## Run a Trained Model

To see the trained agent play in the environment, then you can run the `play.py` script. But before doing so, you should modify the following variables in the `settings` section of the file.

- `ENV_FILE`: update this to the file path to the environment file you donwloaded. You should use an environment that is *not* headless if you want to view it visually.
- `MODEL_NAME` The name of the model you assigned when training. Leave this unchanged if you want to view the agent that was packaged with this repository.
