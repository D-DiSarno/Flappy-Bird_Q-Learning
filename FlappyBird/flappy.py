from itertools import cycle
import random
import sys
import pygame
from pygame.locals import *

xrange = range


class gameManager:   
    def __init__(self, game, fpsCount = 1): #fps Count is number of frames between when the reward is returned after the action
        self.movement = None
        self.score_check = 0
        self.fpsCount = fpsCount
        self.game = game
        self.keyEventUp = pygame.event.Event(KEYDOWN, {'key': self.game.getInputs()[0]})
        self.topCount = 0
        self.bottomCount = 0
        self.upperDistOffset = 10  #How far away from the pipe the bird has to be in order for certain reward 
        self.lowerDistOffset = 20  

    def executeAction(self):
        if self.movement == [1,0]:
           pygame.event.post(self.keyEventUp)
           self.game.assignAction()
        else:
            self.game.assignWait()
            

    def action(self, action, score):
        self.movement = action
        self.score_check = score
        self.executeAction()

    def determinePosReward(self): #Increases reward as it gets 
        return 8.88889*(self.upperDistOffset)+355.556

    def getReward(self, crashInfo, score):
        reward = 0
        done = False
        self.distanceTop = self.game.playery - (self.game.IMAGES['pipe'][0].get_height()+self.game.upperPipes[-2]['y'])
        self.distanceBottom = self.game.lowerPipes[-2]['y'] - self.game.playery
        if self.distanceTop < self.upperDistOffset  or self.distanceBottom < self.lowerDistOffset :
            reward += -200
        else:
            reward += self.determinePosReward()
        if crashInfo != None: # Did crash
            done = True
            reward += -200
            return reward, done, score
        elif score > self.score_check:
            reward += 200
            return reward, done, score
        else:
            return reward, done, score

    def actionSequence(self,action):
        self.action(action, self.game.score)
        crashInfo = None
        for i in range(self.fpsCount):
            crashInfo = self.game.levelLoop()
        return self.getReward(crashInfo, self.game.score)
    
    def reset(self):
        self.game.initLevel()
        pygame.event.post(self.keyEventUp)

    def getState(self):
        try:
            distanceTop = self.game.playery - (self.game.IMAGES['pipe'][0].get_height()+self.game.upperPipes[-2]['y'])
            distanceBottom = self.game.lowerPipes[-2]['y'] - self.game.playery
            return [
                self.game.lowerPipes[-2]['x'],
                distanceTop,
                distanceBottom,
                self.game.playerVelY,
            ]
        except AttributeError: #when getting the length during setup of model of the getState when some variables have not been made yet
            return (0,0,0,0)
    
    def play(self):
        while True:
            self.game.initLevel()
            crashInfo = None
            while crashInfo == None:
                crashInfo = self.game.levelLoop()
            self.game.showGameOverScreen(crashInfo)
    
    def setOutputs(self, predict):#predict is a tensor
        self.game.output1 = predict[0].item()
        self.game.output2 = predict[1].item()


class game:
    def __init__(self, file_dir = ''):
        self.SCREENWIDTH  = 288
        self.SCREENHEIGHT = 512
        self.PIPEGAPSIZE  = 100 # gap between upper and lower part of pipe
        self.BASEY        = self.SCREENHEIGHT * 0.79
        # image, sound and hitmask  dicts
        self.IMAGES, self.SOUNDS, self.HITMASKS = {}, {}, {}

        # list of all possible players (tuple of 3 positions of flap)
        self.PLAYERS_LIST = (
            # red bird
            (
                file_dir + '/FlappyBird/assets/sprites/redbird-upflap.png',
                file_dir + '/FlappyBird/assets/sprites/redbird-midflap.png',
                file_dir + '/FlappyBird/assets/sprites/redbird-downflap.png',
            ),
            # blue bird
            (
                file_dir + '/FlappyBird/assets/sprites/bluebird-upflap.png',
                file_dir + '/FlappyBird/assets/sprites/bluebird-midflap.png',
                file_dir + '/FlappyBird/assets/sprites/bluebird-downflap.png',
            ),
            # yellow bird
            (
                file_dir + '/FlappyBird/assets/sprites/yellowbird-upflap.png',
                file_dir + '/FlappyBird/assets/sprites/yellowbird-midflap.png',
                file_dir + '/FlappyBird/assets/sprites/yellowbird-downflap.png',
            ),
        )

        # list of backgrounds
        self.BACKGROUNDS_LIST = (
            file_dir + '/FlappyBird/assets/sprites/background-day.png',
            file_dir + '/FlappyBird/assets/sprites/background-night.png',
        )

        # list of pipes
        self.PIPES_LIST = (
            file_dir + '/FlappyBird/assets/sprites/pipe-green.png',
            file_dir + '/FlappyBird/assets/sprites/pipe-red.png',
        )


        
        self.xrange = range


        pygame.init()
        self.FPSCLOCK = pygame.time.Clock()
        self.SCREEN = pygame.display.set_mode((self.SCREENWIDTH, self.SCREENHEIGHT))
        pygame.display.set_caption('Flappy Bird')

        # numbers sprites for score display
        self.IMAGES['numbers'] = (
            pygame.image.load(file_dir + '/FlappyBird/assets/sprites/0.png').convert_alpha(),
            pygame.image.load(file_dir + '/FlappyBird/assets/sprites/1.png').convert_alpha(),
            pygame.image.load(file_dir + '/FlappyBird/assets/sprites/2.png').convert_alpha(),
            pygame.image.load(file_dir + '/FlappyBird/assets/sprites/3.png').convert_alpha(),
            pygame.image.load(file_dir + '/FlappyBird/assets/sprites/4.png').convert_alpha(),
            pygame.image.load(file_dir + '/FlappyBird/assets/sprites/5.png').convert_alpha(),
            pygame.image.load(file_dir + '/FlappyBird/assets/sprites/6.png').convert_alpha(),
            pygame.image.load(file_dir + '/FlappyBird/assets/sprites/7.png').convert_alpha(),
            pygame.image.load(file_dir + '/FlappyBird/assets/sprites/8.png').convert_alpha(),
            pygame.image.load(file_dir + '/FlappyBird/assets/sprites/9.png').convert_alpha()
        )

        # game over sprite
        self.IMAGES['gameover'] = pygame.image.load(file_dir + '/FlappyBird/assets/sprites/gameover.png').convert_alpha()
        # message sprite for welcome screen
        self.IMAGES['message'] = pygame.image.load(file_dir + '/FlappyBird/assets/sprites/message.png').convert_alpha()
        # base (ground) sprite
        self.IMAGES['base'] = pygame.image.load(file_dir + '/FlappyBird/assets/sprites/base.png').convert_alpha()

        # sounds
        if 'win' in sys.platform:
            self.soundExt = '.wav'
        else:
            self.soundExt = '.ogg'

        self.SOUNDS['die']    = pygame.mixer.Sound(file_dir + '/FlappyBird/assets/audio/die' + self.soundExt)
        self.SOUNDS['hit']    = pygame.mixer.Sound(file_dir + '/FlappyBird/assets/audio/hit' + self.soundExt)
        self.SOUNDS['point']  = pygame.mixer.Sound(file_dir + '/FlappyBird/assets/audio/point' + self.soundExt)
        self.SOUNDS['swoosh'] = pygame.mixer.Sound(file_dir + '/FlappyBird/assets/audio/swoosh' + self.soundExt)
        self.SOUNDS['wing']   = pygame.mixer.Sound(file_dir + '/FlappyBird/assets/audio/wing' + self.soundExt)



        
    def initLevel(self):
        '''used to start initilize the level'''
        # select random background sprites
        self.randBg = random.randint(0, len(self.BACKGROUNDS_LIST) - 1)
        self.IMAGES['background'] = pygame.image.load(self.BACKGROUNDS_LIST[self.randBg]).convert()

        # select random player sprites
        self.randPlayer = random.randint(0, len(self.PLAYERS_LIST) - 1)
        self.IMAGES['player'] = (
            pygame.image.load(self.PLAYERS_LIST[self.randPlayer][0]).convert_alpha(),
            pygame.image.load(self.PLAYERS_LIST[self.randPlayer][1]).convert_alpha(),
            pygame.image.load(self.PLAYERS_LIST[self.randPlayer][2]).convert_alpha(),
        )

        # select random pipe sprites
        self.pipeindex = random.randint(0, len(self.PIPES_LIST) - 1)
        self.IMAGES['pipe'] = (
            pygame.transform.flip(
                pygame.image.load(self.PIPES_LIST[self.pipeindex]).convert_alpha(), False, True),
            pygame.image.load(self.PIPES_LIST[self.pipeindex]).convert_alpha(),
        )

        # hitmask for pipes
        self.HITMASKS['pipe'] = (
            self.getHitmask(self.IMAGES['pipe'][0]),
            self.getHitmask(self.IMAGES['pipe'][1]),
        )

        # hitmask for player
        self.HITMASKS['player'] = (
            self.getHitmask(self.IMAGES['player'][0]),
            self.getHitmask(self.IMAGES['player'][1]),
            self.getHitmask(self.IMAGES['player'][2]),
        )
        """Shows welcome screen animation of flappy bird"""
        # index of player to blit on screen
        self.playerIndex = 0
        self.playerIndexGen = cycle([0, 1, 2, 1])
        # iterator used to change playerIndex after every 5th iteration
        self.loopIter = 0

        self.playerx = int(self.SCREENWIDTH * 0.2)
        self.playery = int((self.SCREENHEIGHT - self.IMAGES['player'][0].get_height()) / 2)

        self.messagex = int((self.SCREENWIDTH - self.IMAGES['message'].get_width()) / 2)
        self.messagey = int(self.SCREENHEIGHT * 0.12)

        self.basex = 0
        # amount by which base can maximum shift to left
        self.baseShift = self.IMAGES['base'].get_width() - self.IMAGES['background'].get_width()

        # player shm for up-down motion on welcome screen
        self.playerShmVals = {'val': 0, 'dir': 1}
        self.movementInfo = self.showWelcomeAnimation()

        self.score = self.playerIndex = self.loopIter = 0
        self.playerIndexGen = self.movementInfo['playerIndexGen']
        self.playerx, self.playery = int(self.SCREENWIDTH * 0.2), self.movementInfo['playery']
        self.basex = self.movementInfo['basex']
        self.baseShift = self.IMAGES['base'].get_width() - self.IMAGES['background'].get_width()

        # get 2 new pipes to add to upperPipes lowerPipes list
        self.newPipe1 = self.getRandomPipe()
        self.newPipe2 = self.getRandomPipe()

        # list of upper pipes
        self.upperPipes = [
            {'x': self.SCREENWIDTH + 200, 'y': self.newPipe1[0]['y']},
            {'x': self.SCREENWIDTH + 200 + (self.SCREENWIDTH / 2), 'y': self.newPipe2[0]['y']},
        ]

        # list of lowerpipe
        self.lowerPipes = [
            {'x': self.SCREENWIDTH + 200, 'y': self.newPipe1[1]['y']},
            {'x': self.SCREENWIDTH + 200 + (self.SCREENWIDTH / 2), 'y': self.newPipe2[1]['y']},
        ]

        self.pipeVelX = -4 

        # player velocity, max velocity, downward acceleration, acceleration on flap
        self.playerVelY    =  -9   # player's velocity along Y, default same as playerFlapped
        self.playerMaxVelY =  10   # max vel along Y, max descend speed
        self.playerMinVelY =  -8   # min vel along Y, max ascend speed
        self.playerAccY    =   1   # players downward acceleration
        self.playerRot     =  45   # player's rotation
        self.playerVelRot  =   3   # angular speed
        self.playerRotThr  =  20   # rotation threshold
        self.playerFlapAcc =  -9   # players speed on flapping
        self.playerFlapped = False # True when player flaps


    def showWelcomeAnimation(self):
        '''shows the welcome animation'''
        while True:

            values = self.introLooper()

            # adjust playery, playerIndex, basex
            if (self.loopIter + 1) % 5 == 0:
                self.playerIndex = next(self.playerIndexGen)
            self.loopIter = (self.loopIter + 1) % 30
            self.basex = -((-self.basex + 4) % self.baseShift)
            self.playerShm(self.playerShmVals)

            # draw sprites
            self.SCREEN.blit(self.IMAGES['background'], (0,0))
            self.SCREEN.blit(self.IMAGES['player'][self.playerIndex],
                        (self.playerx, self.playery + self.playerShmVals['val']))
            self.SCREEN.blit(self.IMAGES['message'], (self.messagex, self.messagey))
            self.SCREEN.blit(self.IMAGES['base'], (self.basex, self.BASEY))

            pygame.display.update()
            self.FPSCLOCK.tick(self.getFPS())

            if values != None:
                return values

    def levelLoop(self):
        '''completes one game loop'''
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and (event.key in self.getInputs()):
                if self.playery > -2 * self.IMAGES['player'][0].get_height():
                    self.playerVelY = self.playerFlapAcc
                    self.playerFlapped = True
                    self.wingSound()
        
        # check for crash here
        self.crashTest = self.checkCrash({'x': self.playerx, 'y': self.playery, 'index': self.playerIndex},
                               self.upperPipes, self.lowerPipes)
        if self.crashTest[0]:
            return {
                'y': self.playery,
                'groundCrash': self.crashTest[1],
                'basex': self.basex,
                'upperPipes': self.upperPipes,
                'lowerPipes': self.lowerPipes,
                'score': self.score,
                'playerVelY': self.playerVelY,
                'playerRot': self.playerRot
            }

        # check for score
        self.playerMidPos = self.playerx + self.IMAGES['player'][0].get_width() / 2
        for pipe in self.upperPipes:
            self.pipeMidPos = pipe['x'] + self.IMAGES['pipe'][0].get_width() / 2
            if self.pipeMidPos <= self.playerMidPos < self.pipeMidPos + 4:
                self.score += 1
                self.pointSound()

        # playerIndex basex change
        if (self.loopIter + 1) % 3 == 0:
            self.playerIndex = next(self.playerIndexGen)
        self.loopIter = (self.loopIter + 1) % 30
        self.basex = -((-self.basex + 100) % self.baseShift)

        # rotate the player
        if self.playerRot > -90:
            self.playerRot -= self.playerVelRot

        # player's movement
        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
            self.playerVelY += self.playerAccY
        if self.playerFlapped:
            self.playerFlapped = False

            # more rotation to cover the threshold (calculated in visible rotation)
            self.playerRot = 45

        self.playerHeight = self.IMAGES['player'][self.playerIndex].get_height()
        self.playery += min(self.playerVelY, self.BASEY - self.playery - self.playerHeight)

        # move pipes to left
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 3 > len(self.upperPipes) > 0 and 0 < self.upperPipes[0]['x'] < 5:  #at faster frame rate sometimes wont generate pipe
            self.newPipe = self.getRandomPipe()
            self.upperPipes.append(self.newPipe[0])
            self.lowerPipes.append(self.newPipe[1])

        # remove first pipe if its out of the screen
        if len(self.upperPipes) > 0 and self.upperPipes[0]['x'] < -self.IMAGES['pipe'][0].get_width():
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)

        # draw sprites
        self.SCREEN.blit(self.IMAGES['background'], (0,0))

        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            self.SCREEN.blit(self.IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            self.SCREEN.blit(self.IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        self.SCREEN.blit(self.IMAGES['base'], (self.basex, self.BASEY))
        # print score so player overlaps the score
        self.showScore(self.score)

        # Player rotation has a threshold
        self.visibleRot = self.playerRotThr
        if self.playerRot <= self.playerRotThr:
            self.visibleRot = self.playerRot
        
        self.playerSurface = pygame.transform.rotate(self.IMAGES['player'][self.playerIndex], self.visibleRot)
        self.SCREEN.blit(self.playerSurface, (self.playerx, self.playery))
        self.showInfo()
        pygame.display.update()
        self.FPSCLOCK.tick(self.getFPS())

    def checkCrash(self, player, upperPipes, lowerPipes):
        """returns True if player collides with base or pipes."""
        pi = player['index']
        player['w'] = self.IMAGES['player'][0].get_width()
        player['h'] = self.IMAGES['player'][0].get_height()

        # if player crashes into ground
        if player['y'] + player['h'] >= self.BASEY - 1:
            return [True, True]
        else:

            playerRect = pygame.Rect(player['x'], player['y'],
                        player['w'], player['h'])
            pipeW = self.IMAGES['pipe'][0].get_width()
            pipeH = self.IMAGES['pipe'][0].get_height()

            for uPipe, lPipe in zip(upperPipes, lowerPipes):
                # upper and lower pipe rects
                uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
                lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

                # player and upper/lower pipe hitmasks
                pHitMask = self.HITMASKS['player'][pi]
                uHitmask = self.HITMASKS['pipe'][0]
                lHitmask = self.HITMASKS['pipe'][1]

                # if bird collided with upipe or lpipe
                uCollide = self.pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
                lCollide = self.pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

                if uCollide or lCollide:
                    return [True, False]

        return [False, False]

    def showGameOverScreen(self,crashInfo):
        """crashes the player down and shows gameover image"""
        score = crashInfo['score']
        playerx = self.SCREENWIDTH * 0.2
        playery = crashInfo['y']
        playerHeight = self.IMAGES['player'][0].get_height()
        playerVelY = crashInfo['playerVelY']
        playerAccY = 2
        playerRot = crashInfo['playerRot']
        playerVelRot = 7

        self.basex = crashInfo['basex']

        upperPipes, lowerPipes = crashInfo['upperPipes'], crashInfo['lowerPipes']

        # play hit and die sounds
        self.hitSound()
        if not crashInfo['groundCrash']:
            pass
            self.dieSound()

        while True:
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE): 
                    pygame.quit()
                    sys.exit()
                if event.type == KEYDOWN and (event.key in self.getInputs()):
                    if playery + playerHeight >= self.BASEY - 1:
                        return

            # player y shift
            if playery + playerHeight < self.BASEY - 1:
                playery += min(playerVelY, self.BASEY - playery - playerHeight)

            # player velocity change
            if playerVelY < 15:
                playerVelY += playerAccY

            # rotate only when it's a pipe crash
            if not crashInfo['groundCrash']:
                if playerRot > -90:
                    playerRot -= playerVelRot

            # draw sprites
            self.SCREEN.blit(self.IMAGES['background'], (0,0))

            for uPipe, lPipe in zip(upperPipes, lowerPipes):
                self.SCREEN.blit(self.IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
                self.SCREEN.blit(self.IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

            self.SCREEN.blit(self.IMAGES['base'], (self.basex, self.BASEY))
            self.showScore(score)
            playerSurface = pygame.transform.rotate(self.IMAGES['player'][1], playerRot)
            self.SCREEN.blit(playerSurface, (playerx,playery))
            self.SCREEN.blit(self.IMAGES['gameover'], (50, 180))
            self.FPSCLOCK.tick(self.getFPS())
            pygame.display.update()

    def showScore(self,score):
        """displays score in center of screen"""
        scoreDigits = [int(x) for x in list(str(score))]
        totalWidth = 0 # total width of all numbers to be printed

        for digit in scoreDigits:
            totalWidth += self.IMAGES['numbers'][digit].get_width()

        Xoffset = (self.SCREENWIDTH - totalWidth) / 2

        for digit in scoreDigits:
            self.SCREEN.blit(self.IMAGES['numbers'][digit], (Xoffset, self.SCREENHEIGHT * 0.1))
            Xoffset += self.IMAGES['numbers'][digit].get_width()

    def getRandomPipe(self):
        """returns a randomly generated pipe"""
        # y of gap between upper and lower pipe
        gapY = random.randrange(0, int(self.BASEY * 0.6 - self.PIPEGAPSIZE))
        gapY += int(self.BASEY * 0.2)
        pipeHeight = self.IMAGES['pipe'][0].get_height()
        pipeX = self.SCREENWIDTH + 10

        return [
            {'x': pipeX, 'y': gapY-pipeHeight},  # upper pipe
            {'x': pipeX, 'y': gapY+self.PIPEGAPSIZE}, # lower pipe
        ]

    def playerShm(self, playerShm):
        """oscillates the value of playerShm['val'] between 8 and -8"""
        if abs(playerShm['val']) == 8:
            playerShm['dir'] *= -1

        if playerShm['dir'] == 1:
            playerShm['val'] += 1
        else:
            playerShm['val'] -= 1

    def pixelCollision(self, rect1, rect2, hitmask1, hitmask2):
        """Checks if two objects collide and not just their rects"""
        rect = rect1.clip(rect2)

        if rect.width == 0 or rect.height == 0:
            return False

        x1, y1 = rect.x - rect1.x, rect.y - rect1.y
        x2, y2 = rect.x - rect2.x, rect.y - rect2.y

        for x in xrange(rect.width):
            for y in xrange(rect.height):
                if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                    return True
        return False
        
    def getHitmask(self, image):
        """returns a hitmask using an image's alpha."""
        mask = []
        for x in xrange(image.get_width()):
            mask.append([])
            for y in xrange(image.get_height()):
                mask[x].append(bool(image.get_at((x,y))[3]))
        return mask
    
    def showInfo(self):
        pass

    def getFPS(self):
        """Returns the FPS for current game"""
        return 30
    
    def getInputs(self):
        """Returns the correct input for current game"""
        return [K_SPACE, K_UP]
    
    def introLooper(self):
        """First game loop in intro screen"""
        # May need to be differnt for training and evaulating 
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and (event.key in self.getInputs()):
                self.wingSound()
                return {
                    'playery': self.playery + self.playerShmVals['val'],
                    'basex': self.basex,
                    'playerIndexGen': self.playerIndexGen,
                    }
    def wingSound(self):
        self.SOUNDS['wing'].play()
    
    def pointSound(self):
        self.SOUNDS['point'].play()
    
    def hitSound(self):
        self.SOUNDS['hit'].play()
    
    def dieSound(self):
        self.SOUNDS['die'].play()

    def assignAction(self):
        pass

    def assignWait(self):
        pass


class PlayGame(game):
    def __init__(self, file_dir = ''):
        game.__init__(self, file_dir = file_dir)
       # self.type = 'PLAY'
    def getFps(self):
        return 30
    def getInputs(self):
        return [K_SPACE, K_UP]
    def introLooper(self):
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and (event.key in self.getInputs()):
                self.wingSound()
                return {
                'playery': self.playery + self.playerShmVals['val'],
                'basex': self.basex,
                'playerIndexGen': self.playerIndexGen,
                    }
    def wingSound(self):
        self.SOUNDS['wing'].play()
    
    def pointSound(self):
        self.SOUNDS['point'].play()
    
    def hitSound(self):
        self.SOUNDS['hit'].play()
    
    def dieSound(self):
        self.SOUNDS['die'].play()

class TrainGame(game):
    def __init__(self, file_dir = ''):
        game.__init__(self, file_dir)

    def getFPS(self):
        return 3840

    def getInputs(self):
        return [K_AC_BACK] #Using Andriod Backspace Key because not on PC keyboard

    def introLooper(self):
        return {
        'playery': self.playery + self.playerShmVals['val'],
        'basex': self.basex,
        'playerIndexGen': self.playerIndexGen,
                }
    def wingSound(self):
        pass
    
    def pointSound(self):
        pass
    
    def hitSound(self):
        pass
    
    def dieSound(self):
        pass

class EvaluateGame(game):
    def __init__(self, file_dir = '', fps = 3840):
        game.__init__(self, file_dir)
        self.fps = fps
        self.font = pygame.font.SysFont('Courier New', 30) #Name of font then size
        self.IMAGES['wait'] = pygame.image.load(file_dir + '/FlappyBird/assets/sprites/added/Wait.png').convert_alpha()
        self.IMAGES['flap'] = pygame.image.load(file_dir + '/FlappyBird/assets/sprites/added/Flap.png').convert_alpha()
        self.output1 = None
        self.output2 = None
        self.flapCount = 0 #Used to see how many frames have passed since the flap is shown
        self.imageShown = None

    def getFPS(self):
        return self.fps

    def getInputs(self):
        return [K_AC_BACK] #Using Andriod Backspace Key because not on PC keyboard
        
    def introLooper(self):
        return {
        'playery': self.playery + self.playerShmVals['val'],
        'basex': self.basex,
        'playerIndexGen': self.playerIndexGen,
            }

    def assignAction(self):
        self.imageShown = self.IMAGES['flap']
        self.flapCount = 1
    
    def assignWait(self):
        if self.flapCount > 2:
            self.flapCount = 0
            self.imageShown = self.IMAGES['wait']
        else:
            self.flapCount += 1

    def showInfo(self):
        if self.imageShown != None:
            self.SCREEN.blit(self.imageShown, (150,425))
            tOutput1 = self.font.render(str(int(self.output1)), True, (0,0,0))
            tOutput2 = self.font.render(str(int(self.output2)), True, (0,0,0))
            self.SCREEN.blit(tOutput1, (10,435))
            self.SCREEN.blit(tOutput2, (10,475))

    def showInfoCord(self):
        pass

    def wingSound(self):
        pass
    
    def pointSound(self):
        pass
    
    def hitSound(self):
        pass
    
    def dieSound(self):
        pass
