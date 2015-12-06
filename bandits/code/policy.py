import numpy.random
import numpy as np

articles = {}
coefficients = {}
ALPHA = 1.8
last_recommended = None
last_user = None


def set_articles(some_articles):
    # for article in some_articles:
    #    articles[article[0]] = article[1:]
    pass


def update(reward):
    m, b = coefficients[last_recommended]
    m += np.dot(last_user, np.transpose(last_user))
    b += np.dot(reward, last_user)
    coefficients[last_recommended] = m, b


def reccomend(time, user_features, articles):
    ucbs = {}
    dimension = len(user_features)
    for article_id in articles:
        if not coefficients.__contains__(article_id):
            coefficients[article_id] = np.identity(dimension), np.zeros(dimension)
        m = coefficients[article_id][0]
        w = np.dot(np.linalg.inv(m), coefficients[article_id][1])
        ucbs[article_id] = np.dot(w, user_features) + \
                           ALPHA*np.sqrt(np.dot(np.dot(np.transpose(user_features), m), user_features))
    recommended = None, -np.inf
    for article_id in ucbs.keys():
        if ucbs[article_id] > recommended[1]:
            recommended = article_id, ucbs[article_id]
    global last_recommended
    last_recommended = recommended[0]
    global last_user
    last_user = user_features
    return recommended[0]
