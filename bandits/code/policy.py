import numpy as np
import scipy.linalg as linalg

from collections import namedtuple

LinStat = namedtuple('LinStat', ['mean', 'cov', 'inv_cov', 'response'])

class LinUCB(object):

    def __init__(self, alpha):
        self.alpha = alpha

    def predict(self, context_dict, model):

        ucb_scores = {}

        for action in context_dict:
            stats = model.stat_for_action(action)
            context = context_dict[action]

            ucb_scores[action] = context.dot(stats.mean) + self.alpha * np.sqrt(context.T.dot(stats.inv_cov.dot(context)))

        return max(ucb_scores, key=ucb_scores.get)

class LinearModel(object):

    def __init__(self, dimension, reg_param, err_sigma=1.0):
        self.dim = dimension
        self.lam = reg_param
        self.err_normalization = 1.0 / (err_sigma * err_sigma)
        self.stat_dict = {}

    def stat_for_action(self, action):
        if self.stat_dict.get(action) is None:
            mean = np.zeros(self.dim)
            cov = self.lam * np.identity(self.dim)
            inv_cov = linalg.inv(cov)
            response = np.zeros(self.dim)

            self.stat_dict[action] = LinStat(mean, cov, inv_cov, response)

        return self.stat_dict[action]

    def update(self, data_dict):

        for action in data_dict:
            stat = self.stat_for_action(action)
            data = data_dict[action]

            response = stat.response
            cov = stat.cov

            for (y, context) in data:
                cov += self.err_normalization * np.outer(context, context)
                response += y * context

            inv_cov = linalg.inv(cov)
            mean = self.err_normalization * inv_cov.dot(response)

            self.stat_dict[action] = LinStat(mean, cov, inv_cov, response)


class Adaptor(object):

    def __init__(self, bandit, model):
        self.bandit = bandit
        self.model = model
        self.last_recommendation = None
        self.last_context = None

    def recommend(self, context, articles):
        c = np.array(context)
        context_dict = {}
        for article in articles:
            context_dict[article] = c

        recommendation = self.bandit.predict(context_dict, self.model)

        self.last_recommendation = recommendation
        self.last_context = c

        return recommendation

    def update(self, reward):
        if reward == 0:
            y = -1
        elif reward == 1:
            y = 1
        else:
            return

        data = {}
        data[self.last_recommendation] = [(y, self.last_context)]

        self.model.update(data)


ADAPTOR = Adaptor(LinUCB(0.5), LinearModel(6, 1.0))

def set_articles(articles):
    pass

def update(reward):
    ADAPTOR.update(reward)

def reccomend(time, user_features, articles):
    out = ADAPTOR.recommend(user_features, articles)
    return out

