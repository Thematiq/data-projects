import numpy as np
from typing import Optional
from numba import jit
from random import random


class SimulatedAnnealing:
    def __init__(self):
        self._story = None
        self._best_story = None
        self.model = self.exponential_model

    def init(self):
        raise NotImplementedError()

    def cost(self, solution):
        raise NotImplementedError()

    def get_neighbour(self, solution):
        raise NotImplementedError()

    def fit(self, initial_temp: float, alpha: float, save_story: bool = True, verbose: bool = True,
            max_iterations: Optional[int] = None, max_stagnation: Optional[int] = None, initial_state=None,
            epsilon: float = 1e-3, break_point: Optional[float] = None, use_reheating: bool = False):
        """
        Find function's minimum using simulated annealing
        Basic temperature equation is:
        T_n = T_n * alpha
        :param initial_temp:    Initial temperature of the system
        :param alpha:           Temperature's drop coefficient
        :param save_story:      Flag informing whether model should save the story (score and solution at every iter)
        :param max_iterations:  Maximum number of epochs
        :param max_stagnation:  Maximum number of epochs without new best solution
        :param initial_state:   Initial state. If not provided function self.init() is used
        :param epsilon:         Temperature's break point. Optimizer stops when temperature < epsilon
        :param break_point:     Score's break point. Optimizer stops when best_score < break_point
        :param verbose:         If true, method will print current temperature and solution score
        :param use_reheating:   If true, model will reheat instead of breaking when reached epsilon
        :return:                Tuple containing solution and it's score
        """
        if initial_state is None:
            current_sol = self.init()
        else:
            current_sol = initial_state

        if save_story:
            self._story, self._best_story = [], []
        else:
            self._story, self._best_story = None, None

        current_score = self.cost(current_sol)
        best_score, best_sol = current_score, current_sol
        temp = initial_temp
        epochs, stagnated_epochs = 0, 0

        while temp > epsilon or use_reheating:
            if max_iterations is not None and epochs > max_iterations:
                if verbose:
                    print('Number of epochs reached max_iterations')
                break
            if max_stagnation is not None and stagnated_epochs > max_stagnation:
                if verbose:
                    print('Number of epochs without improvement reached max_stagnation')
                break
            if break_point is not None and best_score <= break_point:
                if verbose:
                    print('Best score reached break_point')
                break
            if use_reheating and temp < epsilon:
                temp = initial_temp

            new_sol = self.get_neighbour(current_sol)
            new_score = self.cost(new_sol)
            delta = new_score - current_score

            if self.__metropolis_acceptance(delta, temp):
                current_sol, current_score = new_sol, new_score
            if current_score < best_score:
                best_sol, best_score = current_sol, current_score
                stagnated_epochs = 0
            else:
                stagnated_epochs += 1

            if save_story:
                self._story.append((current_sol, current_score))
                self._best_story.append((best_sol, best_score))
            if verbose:
                print('\rCurrent temperature: {} \t Current solution: {} \t Best solution: {} \t\t'.format(temp,
                                                                                                           current_score,
                                                                                                           best_score), end='')
            temp = self.model(temp, alpha)
            epochs += 1
        else:
            if verbose:
                print('Temperature dropped below epsilon')

        if verbose:
            print(f'\nNumber of epochs: {epochs}')
        return best_sol, best_score

    @staticmethod
    def exponential_model(temp, alpha):
        return temp * alpha

    @staticmethod
    def linear_model(temp, alpha):
        return temp - alpha

    @staticmethod
    def __metropolis_acceptance(delta: float, current_temp: float, threshold: float = 5) -> bool:
        state = delta / current_temp
        if state < -threshold:
            return True
        criterion = np.exp(-state)
        return criterion > random()

    @property
    def story(self):
        if self._story is None:
            raise AttributeError('Last fit had disabled saving!')
        return self._story

    @property
    def best_story(self):
        if self._best_story is None:
            raise AttributeError('Last fit had disabled saving!')
        return self._best_story

    @staticmethod
    def __draw_progress_bar(current: int, total: int, current_score: float,
                            best_score: float, bar_width: int = 50) -> None:
        FILL, BLANK = '█', '.'
        a_fill = int(current * bar_width // total)
        percent = "%.2f" % (100 * current / total)
        curr = "%.2f" % current_score
        best = "%.2f" % best_score
        a = FILL * a_fill
        b = BLANK * (bar_width - a_fill)
        print(f'\rProgress: |{a}{b}| {percent}% current: {curr} best: {best}', end='')
