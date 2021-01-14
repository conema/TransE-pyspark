import pickle
from pyspark import SparkContext


def load_dataset(sc, path):
    ds = sc.textFile(path).map(lambda x: (x.split('\t')))

    head = ds.map(lambda x: x[0])
    label = ds.map(lambda x: x[1])
    tail = ds.map(lambda x: x[2])

    # union of heads and tails
    entities = head.union(tail).distinct()

    label = label.distinct()

    # add unique id to entities and labels
    entities_to_id = entities.zipWithIndex()
    label_to_id = label.zipWithIndex()

    # collect entities and labels in a map (entity -> id and label -> id)
    entities_to_id_map = entities_to_id.collectAsMap()
    label_to_id_map = label_to_id.collectAsMap()

    # substitute entities and labels with their ids
    ds_to_id = ds.map(lambda x: (entities_to_id_map[x[0]], label_to_id_map[x[1]], entities_to_id_map[x[2]]))

    return ds_to_id.collect(), entities_to_id_map, label_to_id_map


def get_id_by_value(value_map, value):
    return [k for k, v in value_map.items() if v == value]


def build_dict(ds):
    ds_dict = {}

    for e1, r, e2 in ds:
        ds_dict[(e1, r, e2)] = None

    return ds_dict


def backup(entity_embedding, label_embedding, id, path):
    with open(path + '/entity_embedding_' + str(id) + '.pkl', 'wb') as output:
        pickle.dump(entity_embedding, output, pickle.HIGHEST_PROTOCOL)

    with open(path + '/label_embedding_' + str(id) + '.pkl', 'wb') as output:
        pickle.dump(label_embedding, output, pickle.HIGHEST_PROTOCOL)


def restore(entity_path, label_path):
    with open(entity_path, 'rb') as input:
        entity_embedding = pickle.load(input)

    with open(label_path, 'rb') as input:
        label_embedding = pickle.load(input)

    return entity_embedding, label_embedding
