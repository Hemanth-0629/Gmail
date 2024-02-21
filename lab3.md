G.HEMANTH
CS21B2011

```
MDP = createMDP(8,["left";"right"])
MDP.States
MDP.Actions
MDP.T
MDP.R
MDP.TerminalStates = ["s1";"s8"];
nS = numel(MDP.States)
nS = numel(MDP.States);
nA = numel(MDP.Actions);
MDP.R = -1*ones(nS,nS,nA);
MDP.TerminalStates = ["s1";"s8"]
nS = numel(MDP.States)
nA = numel(MDP.Actions)
MDP.R = -1*ones(nS,nS,nA)
MDP.R(:,state2idx(MDP,MDP.TerminalStates),:) = 10
MDP.T(1,1,1) = 1
MDP.T(1,2,2) = 1
MDP.T(1,1,1) = 1
MDP.T(1,2,2) = 1
MDP.T(2,1,1) = 1
MDP.T(2,3,2) = 1
MDP.T(3,2,1) = 1
MDP.T(3,4,2) = 1
MDP.T(4,3,1) = 1
MDP.T(4,5,2) = 1
MDP.T(5,4,1) = 1
MDP.T(5,6,2) = 1
MDP.T(6,5,1) = 1
MDP.T(6,7,2) = 1
MDP.T(7,6,1) = 1
MDP.T(7,8,2) = 1
MDP.T(8,7,1) = 1
MDP.T(8,8,2) = 1
MDP.T
MDP.R
env = rlMDPEnv(MDP)
state_information = getObservationInfo(env)
action_information = getActionInfo(env)
qTable = rlTable(state_information,action_information)
qTable.Table
qTable.Table = ones(size(qTable.Table))*5
qTable.Table
qRepresentation = rlQValueRepresentation(qTable,state_information,action_information)
qRepresentation.Options
qRepresentation.Options.L2RegularizationFactor = 0
qRepresentation.Options.LearnRate = 0.01
agentOpts = rlQAgentOptions
agentOpts.EpsilonGreedyExploration
agentOpts.EpsilonGreedyExploration.EpsilonDecay = 0.01
qAgent = rlQAgent(qRepresentation,agentOpts)
trainOpts = rlTrainingOptions
trainOpts.MaxStepsPerEpisode = 10
trainOpts.MaxEpisodes = 100
trainOpts.StopTrainingCriteria = "AverageReward"
trainOpts.StopTrainingValue = 13
trainOpts.ScoreAveragingWindowLength = 30
QTable0 = getLearnableParameters(getCritic(qAgent))
disp(QTable0{1})
doTraining = true
if doTraining
trainingStats = train(qAgent,env,trainOpts);
else
load('genericMDPQAgent.mat','qAgent');
end
Data = sim(qAgent,env)
cummulativeReward = sum(Data.Reward)
QTable = getLearnableParameters(getCritic(qAgent))
QTable{1}


```

1. MDP Setup:
   - An MDP with 8 states and two possible actions ("left" and "right") is created.
   - Terminal states are defined as "s1" and "s8".
   - Transition probabilities (`MDP.T`) and rewards (`MDP.R`) are defined for each state-action pair.

2. Environment Setup:
   - An RL environment (`env`) is created based on the defined MDP.

3. Q-Value Table Setup:
   - Observation and action information are retrieved from the environment.
   - An initial Q-table (`qTable`) is created with all values initialized to 5.

4. Q-Value Representation Setup:
   - The Q-table is used to initialize a Q-value representation (`qRepresentation`) object.

5. Q-Agent Setup:
   - Q-learning agent options are set (`agentOpts`), including exploration parameters.
   - A Q-agent (`qAgent`) is created using the Q-value representation and agent options.

6. Training Options Setup:
   - Training options (`trainOpts`) are configured, specifying parameters such as the maximum number of steps per episode, maximum episodes, and stopping criteria based on average reward.

7. Training:
   - If `doTraining` is set to true, the agent is trained using the specified environment and training options. Training statistics are stored in `trainingStats`.
   - If `doTraining` is false, a pre-trained agent is loaded from a file.

8. Simulation and Evaluation:
   - The trained or loaded agent is simulated in the environment (`env`) to collect data (`Data`).
   - Cumulative reward is calculated from the collected data.
   - The learned Q-table (`QTable`) from the critic of the agent is retrieved and displayed.

9. Results:
   - Finally, the cumulative reward and the learned Q-table are displayed.

![Screenshot 2024-02-21 164835](https://github.com/Hemanth-0629/Gmail/assets/112465874/87b182ff-c062-46cf-a1b0-e1097afcc0e3)
