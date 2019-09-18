from cdqa.utils.download import *
from cdqa.reader.bertqa_sklearn import convert_examples_to_features, read_squad_examples
from pytorch_pretrained_bert.tokenization import BertTokenizer


def test_processor_functions():

    download_squad(dir="./data")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    max_seq_length = 256
    max_query_length = 32
    doc_stride = 128
    is_training = False
    verbose = False
    n_jobs = -1

    examples = read_squad_examples(
        "./data/SQuAD_1.1/dev-v1.1.json",
        is_training=is_training,
        version_2_with_negative=False,
        n_jobs=n_jobs,
    )
    assert len(examples) == 10570

    features = convert_examples_to_features(
        examples,
        tokenizer,
        max_seq_length,
        doc_stride,
        max_query_length,
        is_training,
        verbose,
        n_jobs=n_jobs,
    )
    assert len(features) == 12006
