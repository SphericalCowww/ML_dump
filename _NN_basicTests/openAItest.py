import sys, os, time
import numpy as np
import pandas as pd
import tensorflow as tf
import gym
import matplotlib.pyplot as plt



######################################################################################################
if __name__ == "__main__":
    envName = "CartPole-v1"
    env = gym.make(envName, render_mode="rgb_array")
    env.reset()
    while True:
        plt.imshow(env.render())
        action = env.action_space.sample()
        obs, rew, done, info, _ = env.step(action)
        if done: env.reset()
        time.sleep(0.02)
    env.close()


 
