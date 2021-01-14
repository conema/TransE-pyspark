import findspark
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from TransEmodule import utils
from TransEmodule.TransE import TransE

findspark.init()

if __name__ == "__main__":
    # Change spark.master and the paths if your not using
    # our terraform project!

    # create the session
    conf = SparkConf().setAll([
                                ("spark.worker.cleanup.enabled", True),
                                ("spark.serializer", "org.apache.spark.serializer.KryoSerializer"),
                                ("spark.kryo.registrationRequired", "false"),
                                ("spark.master", "spark://s01:7077")
                            ])

    sc = SparkContext(conf=conf).getOrCreate()

    sc.setLogLevel("ERROR")

    sc.addPyFile('TransEmodule.zip')

    ds_to_id, entities_to_id_map, label_to_id_map = utils.load_dataset(sc, 'hdfs://s01:9000/train2.tsv')

    transe = TransE(spark_context=sc,
                    n_epochs=1000,
                    n_batches=2,
                    gamma_margin=1,
                    learning_rate=0.01,
                    latent_dimension=50,
                    path="/home/ubuntu",
                    distance="L1")

    entity_embedding, label_embedding = transe.fit(ds_to_id,
                                                   entities_to_id_map,
                                                   label_to_id_map)

    print(entity_embedding)
    print(label_embedding)
