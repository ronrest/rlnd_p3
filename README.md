# Tennis

## Environment

The environment used for this project is a variant of the  [Tennis Environment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) agent by Unity.

![image ](tennis.png)

The environment consists of two agents controlling tennis rackets. The aim for each agent is to hit the ball over the net, on to the other agents side of the court, and colaborate with each other to prevent the ball from hitting the floor or going out of bounds for as long as possible.

The available *state-space* for each agent consists of a vector with 24 variables. This consists of 8 values encoding racket and ball information for the current step, and previous 2 timesteps.


TODO: Add image of state space and its components
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
