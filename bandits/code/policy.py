import numpy.random
import numpy as np

articles = {}
coefficients = {} # article_id -> M, inv(M), b, w
ALPHA = 1.8
last_recommended = None
last_user = None


def set_articles(some_articles):
    # for article in some_articles:
    #    articles[article[0]] = article[1:]
    pass


def update(reward):
    m, m_inv, b, w = coefficients[last_recommended]
    m += np.dot(last_user, np.transpose(last_user))
    b += np.dot(reward, last_user)
    m_inv = np.linalg.inv(m)
    coefficients[last_recommended] = m, m_inv, b, np.dot(m_inv, b)


def reccomend(time, user_features, some_articles):
    ucbs = {}
    dimension = len(user_features)
    for article_id in some_articles:
        if not coefficients.__contains__(article_id):
            coefficients[article_id] = np.identity(dimension), np.identity(dimension), np.zeros(dimension), np.zeros(dimension)
        m_inv = coefficients[article_id][1]
        w = coefficients[article_id][3]
        ucbs[article_id] = np.dot(w, user_features) + \
                           ALPHA*np.sqrt(np.dot(np.dot(np.transpose(user_features), m_inv), user_features))
    recommended = None, -np.inf
    for article_id in ucbs.keys():
        if ucbs[article_id] > recommended[1]:
            recommended = article_id, ucbs[article_id]
    global last_recommended
    last_recommended = recommended[0]
    global last_user
    last_user = user_features
    return recommended[0]
