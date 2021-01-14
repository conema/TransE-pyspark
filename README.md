# TransE PySpark

1. [Introduction](#introduction)
2. [Project structure](#Project-structure)
3. [Implementation](#Implementation)
4. [How to (manual)](#How-to-manual)
5. [How to (with our terraform project)](#How-to-with-our-terraform-project)
6. [Experimental results](#Experimental-results)
7. [Contributors](#Contributors)
8. [References](#References])

## Introduction
This project implements the Translating Embeddings for Modeling Multi-relational Data (TransE) \[1\] in Spark \[2\] via the python API (pyspark). TransE is a model for prediction of relationships in knowledge graphs, which given an head, a label and a tail, of the form *(h, l, t)* tries to modifies the embedding to make the invariant *h+l ≈ t* true. The relationships are represented like translations, thus the invariant, if true, means that if h+l is near t, then t must be related to h and t. For example in *bears + like ≈ honey*, t is related to h and t, since bears like honey, the same is not true for *bears + like ≈ stones*. 

The final objective of the method is to learn the embeddings from a training set, by minimizing a margin-based ranking criteria via stochastic gradient descent in mini-batch mode. Since the trainset contains a lot of tuple (h,l, t), it makes sense to distributed this computation in a cluster.

## Project structure
* dataset: dataset used for the experiments (FB15k-237)
* TransEmodule: package which groups TransE and all the helper class/methods
    * TransE.py: TransE implementation
    * Embedding.py: a class to represent word Embedding
    * utils.py: just some utils functions (load dataset, save embeddings, ....)
* example.py: complete example to run the training phase
* test.py: complete example of training (in PySpark)

**Note:** example.py and test.py are ready to be run in a AWS EC2 cluster which is created with our terraform project \[3\]. If you want to run TransE you will need to use the terraform project or you will need to modify the spark.master and the various path in the example.py and test.py files. 

## Implementation

This projects tries to follow the structure and the naming convention of the original paper \[1\], but some modification were needed to make it work with Spark. The idea behind the project is that the training phase can be run in a distributed manner, hence we parallelize the entire dataset, we sample the dataset *n_batches* times and each partition compute the gradient descent of the part of data which it have. For example, if the sample contains \[1, 2, 3, 4\] and it's evenly distributed between 4 worker nodes, the first worker node will compute the gradient of *1*, the second worker will compute the gradient of *2* and so on. Note that more workers we have, less data each worker needs to compute and consequently the learning will be faster. After each batch (sample) computation it is needed to update the embedding vectors to make a new gradient step, hence we need to collect the new embeddings calculated from the workers to the master, so the master node need to have enough RAM to store it. We used various spark (e.g. broadcast) and numpy methods to make the code as efficiently as possible. Note also that we are using a sample of the dataset, the original paper takes all the dataset in batches, but the results are comparable and the sampling procedure is faster than taking batches in spark.

The testing phase can also be distributed with the same fashion of the training phase.

## How to (manual)

* Training: open the example.py file and modify the TransE parameters (iterations, n_batches, etc...) if needed, remember also to modify the Spark master url and the paths if you are not using our terraform project to create the cluster \[3\]. Submit the file to your master.

* Testing: open the test.py remember also to modify the Spark master url and the paths if you are not using our terraform project to create the cluster \[3\]. Submit the file to your master.

**Note:** the modules are passed to spark as a zip file, so you need to create a zip file of the TransEmodule folder and put it in the root folder (at the same level of the example.py and test.py files)

## How to (with our terraform project)
0. Download and install Terraform
1. Download the terraform project from [3] and unzip it
2. Open the terraform project folder "spark-terraform/"
3. Create a file named "terraform.tfvars" and paste this:
```
access_key="<YOUR AWS ACCESS KEY>"
secret_key="<YOUR AWS SECRET KEY>"
token="<YOUR AWS TOKEN>"
aws_key_name="TransE"
amz_key_path="TransE.pem"
```
**Note:** without setting the other variables (you can find it on variables.tf), terraform will create a cluster on region "us-east-1", with 1 namenode, 3 datanode and with an instance type of m5.xlarge.

3. Download the files from this repository
4. Put the files of this repository into the "app" terraform project folder (e.g. example.py should be in spark-terraform/app/example.py and so on for all the other files)
5. Create a zip archive of the TransEmodule and put it on spark-terraform/app/TransEmodule.zip
6. Open a terminal and generate a new ssh-key
```
ssh-keygen -f <PATH_TO_SPARK_TERRAFORM>/spark-terraform/localkey
```
Where `<PATH_TO_SPARK_TERRAFORM>` is the path to the /spark-terraform/ folder (e.g. /home/user/)

7. Login to AWS and create a key pairs named **TransE** in **PEM** file format. Follow the guide on [AWS DOCS](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-key-pairs.html#having-ec2-create-your-key-pair). Download the key and put it in the spark-terraform/ folder.

8. Open a terminal and go to the spark-terraform/ folder, execute the command
 ```
 terraform init
 terraform apply
 ```
 After a while (wait!) it should print some public DNS in a green color, these are the public dns of your instances.

9. Connect via ssh to all your instances via
 ```
ssh -i <PATH_TO_SPARK_TERRAFORM>/spark-terraform/TransE.pem ubuntu@<PUBLIC DNS>
 ```

10. (first) execute on the master (one by one):
 ```
$HADOOP_HOME/sbin/start-dfs.sh
$HADOOP_HOME/sbin/start-yarn.sh
$HADOOP_HOME/sbin/mr-jobhistory-daemon.sh start historyserver' > /home/ubuntu/hadoop-start-master.sh
$SPARK_HOME/sbin/start-master.sh
hdfs dfs -put /home/ubuntu/dataset/train2.tsv /train2.tsv
hdfs dfs -put /home/ubuntu/dataset/test2.tsv /test2.tsv
 ```
And (after) execute on the slaves:
```
$SPARK_HOME/sbin/start-slave.sh spark://s01:7077
```

11. You are ready to execute TransE! Execute this comand on the master
```
/opt/spark-3.0.1-bin-hadoop2.7/bin/spark-submit --deploy-mode cluster --master yarn --executor-cores 4 --executor-memory 16g example.py
```

12. Remember to do `terraform destroy` to delete your EC2 instances

**Note:** The steps from 0 to 7 (included) are needed only on the first execution ever


## Experimental results
TODO

## Contributors
[<img alt="conema" src="https://avatars3.githubusercontent.com/u/12801153?v=4&s=117" width="117">](https://github.com/conema)|[<img alt="fbacci" src="https://avatars3.githubusercontent.com/u/17594819?v=4&s=117" width="117">](https://github.com/fbacci)|
:---:|:---:|
[conema](https://github.com/conema)|[fbacci](https://github.com/fbacci)|

## References
\[1\] https://proceedings.neurips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf

\[2\] https://spark.apache.org/

\[3\] https://github.com/conema/spark-terraform