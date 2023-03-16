Question:

- DQN Fast convergence on small data sets ----BCQ implementation has problems

- Use of reward.cumsum()

- offline rl discrete problem is not difficult to deal with?



TODO:

- Collect loss changes during training (complete)

- The log name automatically reflects the experiment Settings

- Need to count the cumulative rewards per turn

- Use hydra

- Understand multithreaded data collection code





The observed phenomenon:

- env does not use seed, so it is run differently each time. Choosing a score based on SB score is not exactly the same as Configure



Data set directory: "/ home/chengpeng/chengpeng_BCQ_co/chengpeng_BCQ_co/data_generation"

- buffers_100: LP iteration number as reward, 100 data, generally used to test whether the code is feasible;

- buffers_100300: LP iteration times as reward, 100,300 data;

- nodebuffers_10000: node as reward, 1W data.

- timebuffers_10000: solvingtime as reward, 1W data;

- nodemixedbuffers_10000: node as reward, 50% expert data, 50% weak expert data, 1W data;

 
