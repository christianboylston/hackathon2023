from metaflow import FlowSpec, step, current, conda, batch, environment, Parameter
from dotenv import load_dotenv
from custom_decorators import enable_decorator, pip
import os

try:
    load_dotenv(verbose=True, dotenv_path='.env')
except:
    print("No dotenv package")


class TrainFlow(FlowSpec):
    """
    A flow where Metaflow prints 'Hi'.
    Run this flow to validate that Metaflow is installed correctly.
    """
    train = Parameter("train", default=False)
    data_table = Parameter("table", default="intent_dataset")
    model = Parameter("model", default="joaobarroca/distilbert-base-uncased-finetuned-massive-intent-detection-english")
    model_file = Parameter("model_file", default="model")
    delete_enpoint = Parameter("delete_endpoint", default=True)
    endpoint_name = Parameter("endpoint_name", default=f"intent-endpoint")
    
    @environment(vars={
        "SF_USER":os.environ["SF_USER"],
        "SF_PWD":os.environ["SF_PWD"],
        "SF_ACCOUNT":os.environ["SF_ACCOUNT"],
        "SF_DB":os.environ["SF_DB"],
        "SF_SCHEMA":os.environ["SF_SCHEMA"]
    })
    @step
    def start(self):
        """
        This is the 'start' step. We're going to us it to make sure that we have all
        of our credentials
        """
        # print out some debug info
        print("flow name: %s" % current.flow_name)
        print("run id: %s" % current.run_id)
        print("username: %s" % current.username)
        if os.environ.get('EN_BATCH', '0') == '1':
            print("ATTENTION: AWS BATCH ENABLED!") 
        # check variables and db connections are working fine
        assert os.environ['SF_SCHEMA']
        assert os.environ['SF_TABLE']
        assert os.environ['SF_DB']
        assert os.environ['SF_USER']
        assert os.environ['SF_PWD']
        assert os.environ['SF_ACCOUNT']
        self.next(self.load_data)
    
    #@batch(image="public.ecr.aws/outerbounds/dotenv:latest")
    #@pip(libraries={"snowflake-connector-python": "3.0.3", "python-dotenv":"1.0.0",
    #                "pandas":"2.0.1"})
    @environment(vars={
        "SF_USER":os.environ["SF_USER"],
        "SF_PWD":os.environ["SF_PWD"],
        "SF_ACCOUNT":os.environ["SF_ACCOUNT"],
        "SF_DB":os.environ["SF_DB"],
        "SF_SCHEMA":os.environ["SF_SCHEMA"]
    })
    @step
    def load_data(self):
        import snowflake.connector
        import pandas as pd
        conn = snowflake.connector.connect(
                user=os.environ["SF_USER"],
                password=os.environ["SF_PWD"],
                account=os.environ["SF_ACCOUNT"],
                database=os.environ["SF_DB"],
                schema=os.environ["SF_SCHEMA"])
        
        # query the data from Snowflake and create a Pandas dataframe
        query = 'SELECT * FROM intent_dataset;'
        self.data = pd.read_sql(query, conn)
        conn.close()
        self.next(self.process_data)
    
    @step
    def process_data(self):
        from transformers import AutoTokenizer
        from datasets import Features, Value, Dataset, ClassLabel
        # load the pre-trained BERT tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model)
        # Tokenize the dataset
        def tokenize_function(examples):
            return tokenizer(examples['TEXT'], padding="max_length", truncation=True)
        
        label_list = self.data['CATEGORY'].unique().tolist()

        # instantiate a ClassLabel object with the number of classes and the names of the labels
        self.num_classes = len(label_list)
        # create Hugging Face dataset
        features = Features({"TEXT": Value("string"), 
                            "CATEGORY": ClassLabel(num_classes=self.num_classes, names=label_list),
                            "SOURCE": Value("string"),})
        dataset = Dataset.from_pandas(self.data, features=features)
        # Add label column
        dataset = dataset.rename_column("CATEGORY", "label")
        
        # split dataset into train and test sets based on the value of "SOURCE"
        train_dataset = dataset.filter(lambda example: example["SOURCE"] == "train")
        test_dataset = dataset.filter(lambda example: example["SOURCE"] == "test")
        
        self.train_dataset = train_dataset.map(tokenize_function, batched=True)
        self.test_dataset = test_dataset.map(tokenize_function, batched=True)
        self.next(self.train_model)
    
    @step
    def train_model(self):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
        import os
        import tarfile
        import boto3
        # load the pre-trained BERT tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model)
        # load the pre-trained small BERT model for sequence classification
        model = AutoModelForSequenceClassification.from_pretrained(self.model)
        
        if self.train:
            model = AutoModelForSequenceClassification.from_pretrained(self.model, 
                                                                    num_labels=self.num_classes)
            # set up the training arguments
            training_args = TrainingArguments(
                output_dir='./results',          # output directory
                evaluation_strategy='epoch',     # evaluate model after every epoch
                learning_rate=2e-4,              # learning rate
                per_device_train_batch_size=16,  # batch size for training
                per_device_eval_batch_size=64,   # batch size for evaluation
                num_train_epochs=1,              # number of training epochs
                weight_decay=0.01,               # weight decay
                push_to_hub=False,               # whether to upload the model checkpoint to the Hub
                logging_dir='./logs',            # directory for storing logs
                logging_steps=10,
            )

            # create the Trainer object
            trainer = Trainer(
                model=model,                         # the instantiated model to be trained
                args=training_args,                  # training arguments
                train_dataset=self.train_dataset,         # the training dataset
                eval_dataset=self.test_dataset            # the evaluation dataset
            )
            
            trainer.train()
        # Save the model and tokenizer to the specified directory
        directory = "./model"

        if not os.path.exists(directory):
            os.mkdir(directory)
            
        model.save_pretrained(directory)
        tokenizer.save_pretrained(directory)
        # compress model files
        with tarfile.open("model.tar.gz", mode='w:gz') as archive:
            archive.add(directory, arcname='.')
            
        # Set up the S3 client and bucket name
        s3 = boto3.client('s3')
        self.bucket_name = 'loyalhackathon'
        # Set the path and name of the input tar file
        self.input_file = 'model.tar.gz'
        self.object_key = 'model_mf.tar.gz'
        
        # Upload the file to the S3 bucket
        s3.upload_file(self.input_file, self.bucket_name, self.object_key)
        self.next(self.deploy)
        
    @step
    def deploy(self):
        from sagemaker.huggingface.model import HuggingFaceModel
        from sagemaker.serverless import ServerlessInferenceConfig
        import sagemaker
        
        sess = sagemaker.Session()
        role = "SageMakerRole"
        sagemaker_session_bucket="loyalhackathon"
        sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)
     
        # create Hugging Face Model Class
        huggingface_model = HuggingFaceModel(
        model_data=f"s3://{self.bucket_name}/{self.object_key}",
        role="SageMakerRole",         # iam role with permissions to create an Endpoint
        transformers_version="4.12",  # transformers version used
        pytorch_version="1.9",        # pytorch version used
        py_version='py38',            # python version used
        )
        
        # Specify MemorySizeInMB and MaxConcurrency in the serverless config object
        serverless_config = ServerlessInferenceConfig(
            memory_size_in_mb=3072, max_concurrency=1,
        )
        
        predictor = huggingface_model.deploy(
            serverless_inference_config=serverless_config,
            endpoint_name=self.endpoint_name)
            
        data = {
        "inputs": "I love this song"
        }

        res = predictor.predict(data=data)
        print(res)
        if self.delete_enpoint:
            predictor.delete_endpoint()
        self.next(self.end)
        

    @step
    def end(self):
        """
        This is the 'end' step. All flows must have an 'end' step, which is the
        last step in the flow.
        """
        print("all done")


if __name__ == "__main__":
    load_dotenv()
    TrainFlow()