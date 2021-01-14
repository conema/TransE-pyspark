import numpy as np
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from TransEmodule import utils


def check_entities(x, map):
    if x in map:
        return map[x]
    else:
        return None


def calculate_rankings(rank_list):
    flat = rank_list.map(lambda x: x[0]).persist()
    prepare_mean = flat.map(lambda x: (x, 1))
    prepare_hits = flat.map(lambda x: (1 if x <= 10 else 0, 1))

    x = prepare_mean.reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]))
    mean = x[0]/x[1]

    x = prepare_hits.reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]))
    hits = x[0]/x[1]

    return mean, hits


def testing(partition, test_entities_to_id, test_labels_to_id,
            entities_to_id_map, label_to_id_map, entity_embedding,
            label_embedding):
    rank_list = []
    i = 0

    for (h, l, t) in partition:
        # get train ids from testset ids
        h_train = check_entities(utils.get_id_by_value(test_entities_to_id.value, h)[0],
                                 entities_to_id_map.value)

        l_train = check_entities(utils.get_id_by_value(test_labels_to_id.value, l)[0],
                                 label_to_id_map.value)

        t_train = check_entities(utils.get_id_by_value(test_entities_to_id.value, t)[0],
                                 entities_to_id_map.value)

        if h_train is None or l_train is None or t_train is None:
            continue

        # head
        corrupted_entities = entity_embedding.value.vector + label_embedding.value.vector[l_train]
        distances = np.apply_along_axis(lambda x: np.sum(np.abs(x - entity_embedding.value.vector[t_train])), 1, corrupted_entities)
        indices = np.argsort(distances)
        rank = np.where(indices == h_train)

        rank_list.append(rank[0])

        # tail
        corrupted_entities = entity_embedding.value.vector[h_train] + label_embedding.value.vector[l_train]
        distances = np.apply_along_axis(lambda x: np.sum(np.abs(corrupted_entities - x)), 1, entity_embedding.value.vector)
        indices = np.argsort(distances)
        rank = np.where(indices == t_train)

        rank_list.append(rank[0])

        if i % 1000 == 0:
            rank_list_baby = np.concatenate(rank_list, axis=0)
            print("Mean: " + str(np.mean(rank_list_baby)))
            print("Hit: " + str(np.mean(rank_list_baby <= 10)*100))
            print(i)
        i += 1

    return rank_list


def test(testset, test_entities_to_id, test_labels_to_id,
         entities_to_id_map, label_to_id_map, entity_embedding,
         label_embedding):
    testset_rdd = sc.parallelize(testset).persist()

    test_entities_BC = sc.broadcast(test_entities_to_id)
    test_labels_BC = sc.broadcast(test_labels_to_id)
    entities_embedding_BC = sc.broadcast(entity_embedding)
    labels_embedding_BC = sc.broadcast(label_embedding)
    entities_map_BC = sc.broadcast(entities_to_id_map)
    labels_map_BC = sc.broadcast(label_to_id_map)

    rank_list = testset_rdd.mapPartitions(lambda x: testing(x,
                                                            test_entities_BC,
                                                            test_labels_BC,
                                                            entities_map_BC,
                                                            labels_map_BC,
                                                            entities_embedding_BC,
                                                            labels_embedding_BC)
                                          )

    mean, hits = calculate_rankings(rank_list)

    return mean, hits


if __name__ == "__main__":
    # change the paths if you are not using
    # our terraform project!

    # create the session
    conf = SparkConf().setAll([("spark.worker.cleanup.enabled", True),
                               ("spark.serializer",
                                "org.apache.spark.serializer.KryoSerializer"),
                               ("spark.kryo.registrationRequired", "false"),
                               ("spark.master", "spark://s01:7077")])

    sc = SparkContext(conf=conf).getOrCreate()

    sc.addPyFile('TransEmodule.zip')

    entity_embedding, label_embedding = utils.restore('/home/ubuntu/transe/entity_embedding_999.pkl',
                                                      '/home/ubuntu/transe/label_embedding_999.pkl')

    ds_to_id, entities_to_id_map, label_to_id_map = utils.load_dataset(sc, "hdfs://s01:9000/train2.tsv")

    testset, test_entities_to_id, test_labels_to_id = utils.load_dataset(sc, "hdfs://s01:9000/test2.tsv")

    mean, hits = test(testset, test_entities_to_id, test_labels_to_id,
                      entities_to_id_map, label_to_id_map,
                      entity_embedding, label_embedding)

    print("Mean: " + str(mean) + "\nHits@10: " + str(hits))
