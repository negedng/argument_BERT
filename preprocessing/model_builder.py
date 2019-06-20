from argument_BERT.preprocessing import feature_extractor
import timeit

def argue_feature_build(dataset):
    """Build feature to match ArguE features
    """
    propositionSet, parsedPropositions = feature_extractor.get_propositions(dataset)

    print("-----1. Feature: EXTRACTING WORD VECTORS-----")
    start_time = timeit.default_timer()
    dataset = feature_extractor.word_vector_feature(dataset, propositionSet, parsedPropositions)
    elapsed = timeit.default_timer() - start_time
    print(elapsed)

    return dataset
