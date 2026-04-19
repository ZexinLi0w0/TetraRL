# W10 dynamic preference-switching summary

- n_episodes: 100
- switch_episode: 50
- pre-switch omega: [0.500,0.500,0.000,0.000]
- post-switch omega: [0.000,0.000,0.200,0.800]
- adjustment window (skipped after switch): 10 episodes

## Reward windows

- pre-switch mean reward (last 10 eps before switch): 9.3000
- post-switch mean reward (eps 60..69): 9.1000

## Acceptance

- collapse criterion: post < 0.5 * pre -> 4.6500
- reward_collapse: false
