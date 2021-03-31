import pickle


def save_pickle(a, filepath):
    with open(filepath, "wb") as handle:
        pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filepath):
    with open(filepath, "rb") as handle:
        b = pickle.load(handle)

    return b