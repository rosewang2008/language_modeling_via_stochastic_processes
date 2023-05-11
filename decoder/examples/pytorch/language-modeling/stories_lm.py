"""TODO(wikitext): Add a description here."""


import os

import datasets


# TODO(wikitext): BibTeX citation
_CITATION = """\
"""

# TODO(wikitext):
_DESCRIPTION = """\
 The WikiText language modeling dataset is a collection of over 100 million tokens extracted from the set of verified
 Good and Featured articles on Wikipedia. The dataset is available under the Creative Commons Attribution-ShareAlike License.
"""
_URL = "https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/"
_DATA_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext"


class StoriesConfig(datasets.BuilderConfig):
    """BuilderConfig for GLUE."""

    def __init__(self, **kwargs):
        """BuilderConfig for Wikitext
        Args:
          data_url: `string`, url to the dataset (word or raw level)
          **kwargs: keyword arguments forwarded to super.
        """

        super(StoriesConfig, self).__init__(
            version=datasets.Version(
                "1.0.0",
            ),
            **kwargs,
        )


class Stories(datasets.GeneratorBasedBuilder):
    """TODO(wikitext_103): Short description of my dataset."""

    # TODO(wikitext_103): Set up version.
    VERSION = datasets.Version("0.1.0")
    BUILDER_CONFIGS = [
        StoriesConfig(
            name="stories-v1",
            description="word level dataset. No processing is needed other than replacing newlines with <eos> tokens.",
        ),
    ]

    def _info(self):
        # TODO(wikitext): Specifies the datasets.DatasetInfo object
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datasets.Features(
                {
                    "text": datasets.Value("string")
                    # These are the features of your dataset like images, labels ...
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage=_URL,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO(wikitext): Downloads the data and defines the splits
        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"data_file":
                            "/nlp/scr/rewang/transformers/examples/pytorch/language-modeling/stories_data/stories_unannotated/test.txt", "split": "test"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data_file": "/nlp/scr/rewang/transformers/examples/pytorch/language-modeling/stories_data/stories_unannotated/train.txt", "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"data_file": "/nlp/scr/rewang/transformers/examples/pytorch/language-modeling/stories_data/stories_unannotated/test.txt", "split": "valid"},
            ),
        ]
    def _generate_examples(self, data_file, split):

        """Yields examples."""
        # TODO(wikitext): Yields (key, example) tuples from the dataset
        with open(data_file, encoding="utf-8") as f:
            for idx, row in enumerate(f.readlines()):
                if row.strip():
                    yield idx, {"text": row}
                else:
                    yield idx, {"text": ""}
