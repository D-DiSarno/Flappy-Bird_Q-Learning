import numpy as np
  # pillow
import sys
from PIL import Image
import pygame
from .games.base.pygamewrapper import PyGameWrapper

class PLE(object):

    def __init__(self,
                 game, fps=30, frame_skip=1, num_steps=1,
                 reward_values=None, force_fps=True, display_screen=False,
                 add_noop_action=True, state_preprocessor=None, rng=24):

        if reward_values is None:
            reward_values = {}
        self.game = game
        self.fps = fps
        self.frame_skip = frame_skip
        self.NOOP = None
        self.num_steps = num_steps
        self.force_fps = force_fps
        self.display_screen = display_screen
        self.add_noop_action = add_noop_action

        self.last_action = []
        self.action = []
        self.previous_score = 0
        self.frame_count = 0


        if reward_values:
            self.game.adjustRewards(reward_values)


        if isinstance(self.game, PyGameWrapper):
            if isinstance(rng, np.random.RandomState):
                self.rng = rng
            else:
                self.rng = np.random.RandomState(rng)

            # some pygame games preload the images
            # to speed resetting and inits up.
            pygame.display.set_mode((1, 1), pygame.NOFRAME)
        else:
            # in order to use doom, install following https://github.com/openai/doom-py
            from .games.base.doomwrapper import DoomWrapper
            if isinstance(self.game, DoomWrapper):
                self.rng = rng
        
        self.game.setRNG(self.rng)
        self.init()

        self.state_preprocessor = state_preprocessor
        self.state_dim = None

        if self.state_preprocessor is not None:
            self.state_dim = self.game.getGameState()

            if self.state_dim is None:
                raise ValueError(
                    "Asked to return non-visual state on game that does not support it!")
            else:
                self.state_dim = self.state_preprocessor(self.state_dim).shape

        if game.allowed_fps is not None and self.fps != game.allowed_fps:
            raise ValueError("Game requires %dfps, was given %d." %
                             (game.allowed_fps, game.allowed_fps))

    def _tick(self):
        """
        Calculates the elapsed time between frames or ticks.
        """
        if self.force_fps:
            return 1000.0 / self.fps
        else:
            return self.game.tick(self.fps)

    def init(self):

        self.game._setup()
        self.game.init() #this is the games setup/init

    def getActionSet(self):

        actions = self.game.actions

        if (sys.version_info > (3, 0)): #python ver. 3
            if isinstance(actions, dict) or isinstance(actions, dict_values):
                actions = actions.values()
        else:
            if isinstance(actions, dict):
                actions = actions.values()

        actions = list(actions) #.values()
        #print (actions)
        #assert isinstance(actions, list), "actions is not a list"

        if self.add_noop_action:
            actions.append(self.NOOP)

        return actions

    def getFrameNumber(self):


        return self.frame_count

    def game_over(self):


        return self.game.game_over()

    def score(self):


        return self.game.getScore()

    def lives(self):


        return self.game.lives

    def reset_game(self):

        self.last_action = []
        self.action = []
        self.previous_score = 0.0
        self.game.reset()

    def getScreenRGB(self):


        return self.game.getScreenRGB()

    def getScreenGrayscale(self):

        frame = self.getScreenRGB()
        frame = 0.21 * frame[:, :, 0] + 0.72 * \
            frame[:, :, 1] + 0.07 * frame[:, :, 2]
        frame = np.round(frame).astype(np.uint8)

        return frame

    def saveScreen(self, filename):

        frame = Image.fromarray(self.getScreenRGB())
        frame.save(filename)

    def getScreenDims(self):

        return self.game.getScreenDims()

    def getGameStateDims(self):

        return self.state_dim

    def getGameState(self):

        state = self.game.getGameState()
        if state is not None:
            if self.state_preprocessor is not None:
                return self.state_preprocessor(state)
            return state
        else:
            raise ValueError(
                "Was asked to return state vector for game that does not support it!")

    def act(self, action):

        return sum(self._oneStepAct(action) for i in range(self.frame_skip))

    def _draw_frame(self):


        self.game._draw_frame(self.display_screen)

    def _oneStepAct(self, action):

        if self.game_over():
            return 0.0

        if action not in self.getActionSet():
            action = self.NOOP

        self._setAction(action)
        for i in range(self.num_steps):
            time_elapsed = self._tick()
            self.game.step(time_elapsed)
            self._draw_frame()

        self.frame_count += self.num_steps

        return self._getReward()

    def _setAction(self, action):


        if action is not None:
            self.game._setAction(action, self.last_action)

        self.last_action = action

    def _getReward(self):

        reward = self.game.getScore() - self.previous_score
        self.previous_score = self.game.getScore()

        return reward
