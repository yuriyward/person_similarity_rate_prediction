import pandas
from sklearn.neighbors import KNeighborsClassifier

names = ['id', 'user_id', 'film_id', 'rate']


def load_data():
    global train
    global task
    task = pandas.read_csv(
        'data/task.csv',
        header=None,
        names=names,
        delimiter=';',
        index_col='id',
    )
    train = pandas.read_csv(
        'data/train.csv',
        header=None,
        names=names,
        delimiter=';',
        index_col='id',
    )


def predict_rates():
    nbrs = KNeighborsClassifier(n_neighbors=105)  # 30 was also ok
    rates = train['rate']
    films = train.drop(['rate', 'film_id'], axis=1)
    nbrs.fit(films, rates)

    task_films = task.drop(['film_id', 'rate'], axis=1)
    for index, row in task_films.iterrows():
        print(index)
        prediction = nbrs.predict(task_films.loc[[index]])
        task.at[index, 'rate'] = prediction[0]


def save_as_submission():
    task.rate = task.rate.astype(int)
    print(task.head())
    task.to_csv('submission.csv', sep=';', header=None, )


def start_prediction():
    load_data()
    predict_rates()
    save_as_submission()


def start():
    start_prediction()


if __name__ == "__main__":
    start()
