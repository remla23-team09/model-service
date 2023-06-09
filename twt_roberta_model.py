import click
import pandas as pd
import pickle
from transformers import AutoModelForSequenceClassification


pd.set_option('display.max_colwidth', None)


@click.command()
@click.argument('model_output_filepath', type=click.Path())
def main(model_output_filepath):
    # Get the preprocessed data, and split it into test and training data.
    classifier = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

    # Store classifier
    pickle.dump(classifier, open(model_output_filepath, 'wb'))


if __name__ == "__main__":
    main()
