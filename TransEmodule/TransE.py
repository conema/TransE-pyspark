import numpy as np
from math import sqrt, ceil
import random
import time
from TransEmodule.Embedding import Embedding
from TransEmodule import utils


class TransE:
    """
      Translating Embeddings (TransE) for Modeling Multi-relational Data

      spark_context: the spark context to distribuite the computation
      n_epochs: number of epochs to run
      n_batches: number of batches
      gamma_margin: TransE margin hyperparameter (>0, gamma)
      learning_rate: learning rate of the stochastic gradient descent (lambda)
      latent_dimension: dimension of the embeddings (k)
      path: folder where embeddings are saved
      distance: 'L1' if L1 distance or 'L2' if L2 distance
      entity_embedding: starting embedding for training (random if none)
      label_embedding: starting embedding for training (random if none)
    """

    def __init__(self, spark_context, n_epochs,
                 n_batches, gamma_margin, learning_rate,
                 latent_dimension, path=None, distance="L2",
                 entity_embedding=None, label_embedding=None):
        self._spark_context = spark_context
        self._n_epochs = n_epochs
        self._n_batches = n_batches
        self._gamma_margin = gamma_margin
        self._learning_rate = learning_rate
        self._latent_dimension = latent_dimension
        self._distance = distance
        self._entity_embedding = entity_embedding
        self._label_embedding = label_embedding
        self._path = path

    def _initialize(self):
        """
          Initialization process
        """
        min_in = -(6/sqrt(self._latent_dimension))
        max_in = +(6/sqrt(self._latent_dimension))

        # initialize embedding with random uniform distribution
        # according to min and max
        self._entity_embedding = Embedding(min=min_in,
                                           max=max_in,
                                           shape=(self._entity_size,
                                                  self._latent_dimension))
        self._label_embedding = Embedding(min=min_in,
                                          max=max_in,
                                          shape=(self._label_size,
                                                 self._latent_dimension))

        # normalize label embedding
        self._label_embedding.normalize()
        assert ceil(np.linalg.norm(
            self._label_embedding.vector[0])) == float(1)

    def fit(self, trainset, entities_to_id_map, label_to_id_map):
        """
          Start the learning phase
        """
        self._entity_size = len(entities_to_id_map)
        self._label_size = len(label_to_id_map)
        self._trainset_size = len(trainset)

        # initalization step
        if self._entity_embedding is None or self._label_embedding is None:
            self._initialize()
        else:
            print("Embedding loaded from file")

        # start stochastic gradient descent
        self._sgd(trainset)

        return self._entity_embedding, self._label_embedding

    def _sgd(self, trainset):
        """
          Stochastic gradient descent in mini-batch mode
        """

        # cache the trainset in memory accross operations
        trainset_RDD = self._spark_context.parallelize(trainset).persist()

        # build a dict {relation_id: [(head, tail)]} to easy the computations
        trainset_dict = utils.build_dict(trainset)
        trainset_dict_BC = self._spark_context.broadcast(trainset_dict)

        self._entity_embedding.normalize()
        b = (self._trainset_size/self._n_batches)/self._trainset_size

        tot_time = 0

        for epoch in range(self._n_epochs):
            # initialize margin based ranking criterion to minimize
            L_AC = self._spark_context.accumulator(0)

            start = time.time()

            if epoch % 50 == 0 and self._path is not None:
                utils.backup(self._entity_embedding,
                             self._label_embedding,
                             epoch, self._path)

            for _ in range(self._n_batches):
                # add each new batch, broadcast the updated embeddings

                entity_embedding_BC = self._spark_context.broadcast(self._entity_embedding)
                label_embedding_BC = self._spark_context.broadcast(self._label_embedding)

                # sample a new batch of size b
                batch_RDD = trainset_RDD.sample(withReplacement=True,
                                                fraction=b)

                distance = self._distance
                learning_rate = self._learning_rate
                gamma_margin = self._gamma_margin
                entity_size = self._entity_size

                # execute in each partition the minimize step
                new_embeddings = batch_RDD.mapPartitions(
                    lambda x: TransE.corrupt_minimize(trainset_dict_BC,
                                                      x,
                                                      distance,
                                                      entity_embedding_BC,
                                                      label_embedding_BC,
                                                      learning_rate,
                                                      gamma_margin,
                                                      L_AC,
                                                      entity_size)
                )

                # update the old embedding with the new embedding just computed
                self.update_embeddings(new_embeddings)

                # clean some spark stuff to free hdd/ram space
                new_embeddings.unpersist()
                entity_embedding_BC.destroy()
                label_embedding_BC.destroy()
                batch_RDD.unpersist()

            end = (time.time() - start)
            tot_time += end

            print("End of epoch " + str(epoch) + " with cost " +
                  str(L_AC.value) + " in " + str(end) + " seconds")

        print("tot time" + str(tot_time))

    def update_embeddings(self, new_embeddings):
        """
          Update and merge the embeddings returned by the mapPartitions
        """
        new_embeddings_collected = new_embeddings.collect()

        for embedding_vector in new_embeddings_collected:
            for e_key, e_embedding in embedding_vector[0].items():
                self._entity_embedding.vector[e_key] = e_embedding.vector

            for l_key, l_embedding in embedding_vector[1].items():
                self._label_embedding.vector[l_key] = l_embedding.vector

    @staticmethod
    def corrupt_minimize(trainset_dict_BC, partition, distance,
                         entity_embedding_BC, label_embedding_BC,
                         learning_rate, gamma_margin, L_AC, entity_size):
        """
          Create the corrupted triplet and do the minimize step
        """
        # we need local embedding since we can't update the broadcast variable
        # use dict because the embedding will be added one per iteration and
        # doing it with numpy is inefficient
        entity_embedding_local = {}
        label_embedding_local = {}

        for triplet in partition:
            # add local embeddings only if they are not in the dict
            # if they are, they were computed in a previosly iteration
            head, label, tail = TransE.open_triplet(triplet)
            if head not in entity_embedding_local:
                entity_embedding_local[head] = Embedding(
                    vector=entity_embedding_BC.value.vector[head])
            if label not in label_embedding_local:
                label_embedding_local[label] = Embedding(
                    vector=label_embedding_BC.value.vector[label])
            if tail not in entity_embedding_local:
                entity_embedding_local[tail] = Embedding(
                    vector=entity_embedding_BC.value.vector[tail])

            # create a corrupted triplet
            triplet_corrupted, corrupted = TransE.corrupt_triplet(
                trainset_dict_BC, triplet, entity_size)
            if corrupted not in entity_embedding_local:
                entity_embedding_local[corrupted] = Embedding(
                    vector=entity_embedding_BC.value.vector[corrupted])

            # minimize the cost function and take the new embeddings
            # (one for the naive triplet and one for the corrupted one)
            TransE.minimize_L(triplet, triplet_corrupted, distance,
                              entity_embedding_local, label_embedding_local,
                              learning_rate, gamma_margin, L_AC)

            entity_embedding_local[head].normalize()
            # label_embedding_local[label].normalize()
            entity_embedding_local[tail].normalize()
            entity_embedding_local[corrupted].normalize()
        yield (entity_embedding_local, label_embedding_local)

    @staticmethod
    def corrupt(element, random_entity, entity_to_corrupt):
        """
            Corrupts the current triplet
        """
        if entity_to_corrupt == 0:
            return (random_entity, element[1], element[2])
        elif entity_to_corrupt == 2:
            return (element[0], element[1], random_entity)

    @staticmethod
    def generate_corrupted_triplet(trainset_dict_BC, current_element,
                                   entity_to_corrupt, entity_size):
        """
            Generates a corrupted triplet
        """

        while True:
            random_entity = random.randint(0, entity_size-1)

            corrupted = TransE.corrupt(
                current_element, random_entity, entity_to_corrupt)

            if corrupted not in trainset_dict_BC.value:
                return corrupted, random_entity

    @staticmethod
    def corrupt_triplet(trainset_dict_BC, element, entity_size):
        """
            Returns a corrupt triplet, it may corrup
            t the first o second entities (random one)
        """
        entity_to_corrupt = random.randrange(0, 3, 2)
        corrupted_triplet, new_entity = TransE.generate_corrupted_triplet(
            trainset_dict_BC, element, entity_to_corrupt, entity_size)

        return corrupted_triplet, new_entity

    @staticmethod
    def minimize_L(triplet_naive, triplet_corrupted, distance,
                   entity_embedding_local, label_embedding_local,
                   learning_rate, gamma_margin, L_AC):
        """
          Minimize the cost function L
        """
        distance_naive = TransE.calculate_distance(
            triplet_naive, entity_embedding_local,
            label_embedding_local, distance
        )

        distance_corrupted = TransE.calculate_distance(
            triplet_corrupted, entity_embedding_local,
            label_embedding_local, distance
        )

        # do the gradient step only with the positive part
        if gamma_margin + distance_naive - distance_corrupted > 0:

            # update the loss value
            L_AC += gamma_margin + distance_naive - distance_corrupted

            # do gradient descend step
            TransE.gradient_descend(triplet_naive, triplet_corrupted, distance,
                                    entity_embedding_local,
                                    label_embedding_local,
                                    learning_rate)

    @staticmethod
    def calculate_distance(triple, entity_embedding_local,
                           label_embedding_local, distance):
        """
          Calculate distance of two vectors
        """
        head, label, tail = TransE.open_triplet(triple)
        head_vector = entity_embedding_local[head].vector
        relation_vector = label_embedding_local[label].vector
        tail_vector = entity_embedding_local[tail].vector

        if distance == 'L1':
            return np.sum(
                np.absolute(
                    head_vector + relation_vector - tail_vector
                )
            )
        elif distance == 'L2':
            return np.sum(
                np.square(
                    head_vector + relation_vector - tail_vector
                )
            )
        else:
            raise Exception('Distance must be L1 or L2')

    @staticmethod
    def gradient_descend(triplet_naive, triplet_corrupted, distance,
                         entity_embedding_local, label_embedding_local,
                         learning_rate):
        """
          Calculate the gradient of the cost function and do the descend

          The cost function is divided in two, so we compute the gradient of
          the naive triplet and separately the gradient of the corrupted one.
          The sign of the corrupted triplet is minus since in the cost
          function we have "-D(...)
        """

        # gradient for the naive triplet
        head, label, tail = TransE.open_triplet(triplet_naive)

        gradient = TransE.gradient(
            head, label, tail,
            distance,
            entity_embedding_local,
            label_embedding_local)

        entity_embedding_local[head].vector += (learning_rate * gradient)
        label_embedding_local[label].vector += (learning_rate * gradient)
        entity_embedding_local[tail].vector -= (learning_rate * gradient)

        # gradient for the corrupted triplet
        head_corrupted, label_corrupted, tail_corrupted = TransE.open_triplet(triplet_corrupted)

        # note the minus!
        gradient = -TransE.gradient(head_corrupted, label_corrupted,
                                    tail_corrupted, distance,
                                    entity_embedding_local,
                                    label_embedding_local)

        entity_embedding_local[head_corrupted].vector += (learning_rate * gradient)
        label_embedding_local[label_corrupted].vector += (learning_rate * gradient)
        entity_embedding_local[tail_corrupted].vector -= (learning_rate * gradient)

    @staticmethod
    def gradient(head, label, tail,
                 distance, entity_embedding_local, label_embedding_local):
        """
          Calculate the gradient of the distance of the cost function
        """
        # this is the derivative of distance in the cost function
        gradient = 2*(entity_embedding_local[tail].vector -
                      entity_embedding_local[head].vector -
                      label_embedding_local[label].vector)

        if distance == "L1":
            gradient[gradient > 0] = 1
            gradient[gradient < 0] = -1
        elif distance != "L2":
            raise Exception('Distance must be L1 or L2')

        return gradient

    @staticmethod
    def open_triplet(triplet):
        return triplet[0], triplet[1], triplet[2]
