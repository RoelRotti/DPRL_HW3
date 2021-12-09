import numpy as np
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt
from deer.base_classes import Environment


class Chain(Environment):
    def __init__(self, size):
        self.size = size

        self.reward = np.zeros(size)
        self.reward[-1] = 1
        self.reward[0] = 0.2

        self._mode = -1  # train
        self.current_state = 0  # initial state (from assignment)
        self._actions = [0, 1]  # 0 --> left, 1 --> right
        self._last_ponctual_observation = [0, 0]

    def reset(self, mode):
        self.current_state = 0
        self._mode = mode
        return [0]

    def act(self, action):
        """Applies the agent action [action] on the environment.

        Parameters
        -----------
        action : int
            The action selected by the agent to operate on the environment. Should be an identifier
            included between 0 included and nActions() excluded.
        """
        self._last_ponctual_observation = [self.current_state, action]

        if action == 0 and self.current_state != 0:  # move step to the left in the  chain
            self.current_state = self.current_state - 1
        if action == 0 and self.current_state == 0:  # stay left in the chain
            self.current_state = self.current_state
        if action == 1 and self.current_state != self.size-1:  # move step to the right in the  chain
            self.current_state = self.current_state + 1
        if action == 1 and self.current_state == self.size-1:  # stay right in the chain
            self.current_state = self.current_state
        reward = self.reward[self.current_state]
        return reward

    def inputDimensions(self):
        """Gets the shape of the input space for this environment.
        - () or (1,) means each observation at a given time step is a single scalar and the history size is 1 (= only current
        observation)
        """
        return [(1,)]

    def nActions(self):
        """Gets the number of different actions that can be taken on this environment.
        It can be either an integer in the case of a finite discrete number of actions
        or it can be a list of couples [min_action_value,max_action_value] for a continuous action space"""
        return len(self._actions)

    def inTerminalState(self):
        """
        There are no terminal states (according to the assignment).
        """
        return False

    def observe(self):
        """
        Shows last observation.
        """
        return np.array(self._last_ponctual_observation)

    def summarizePerformance(self, test_data_set, *args, **kwargs):
        """Optional hook that can be used to show a summary of the performance of the agent on the
        environment in the current mode.

        Parameters
        -----------
        test_data_set : agent.DataSet
            The dataset maintained by the agent in the current mode, which contains
            observations, actions taken and rewards obtained, as well as whether each transition was terminal or
            not. Refer to the documentation of agent.DataSet for more information.
        """

        observations = test_data_set.observations()

        chain_positions = observations[100:200]

        steps = np.arange(len(chain_positions))

        host = host_subplot(111, axes_class=AA.Axes)
        plt.subplots_adjust(right=0.9, left=0.1)
        par1 = host.twinx()
        host.set_xlabel("Time")
        host.set_ylabel("Position")

        p1, = host.plot(steps, np.repeat(chain_positions, 10), lw=3, c='b', alpha=0.8, ls='-', label='Position')

        par1.set_ylim(-0.09, 9.09)

        host.axis["left"].label.set_color(p1.get_color())

        plt.savefig("plot.png")
        print("A plot of the policy obtained has been saved under the name plot.png")

    def end(self):
        """Optional hook called at the end of all epochs
        """
        print("Done")
