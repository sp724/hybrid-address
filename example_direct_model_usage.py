"""
Example: How to call the underlying model directly without using AddressStructuringPipeline.

This demonstrates how to access individual runners (CRF, Fuzzy Matching, Postcode Matching, 
Post-Processing) directly for more granular control over the address structuring pipeline.
"""

from data_structuring.components.database import Database
from data_structuring.components.runners import (
    RunnerCRF, 
    RunnerFuzzyMatch, 
    RunnerPostProcessing
)
from data_structuring.components.runners.runner_postcode_match import RunnerPostcodeMatch
from data_structuring.components.data_provider.normalization import decode_and_clean_str
from data_structuring.config import (
    CRFConfig,
    FuzzyMatchConfig,
    PostProcessingConfig,
    DatabaseConfig,
    PostProcessingCountryWeightsConfig,
    PostProcessingTownWeightsConfig
)


def direct_model_inference(address_text: str, batch_size: int = 1):
    """
    Call the underlying model directly without the AddressStructuringPipeline wrapper.
    
    Args:
        address_text: Single address or multi-line address string
        batch_size: Number of addresses to process at once
        
    Returns:
        Dictionary containing results from each stage of the pipeline
    """
    
    # Step 1: Initialize configurations (use defaults or customize)
    crf_config = CRFConfig()
    fuzzy_match_config = FuzzyMatchConfig()
    postcode_config = FuzzyMatchConfig()  # Postcode runner uses same config
    post_processing_config = PostProcessingConfig()
    town_weights_config = PostProcessingTownWeightsConfig()
    country_weights_config = PostProcessingCountryWeightsConfig()
    database_config = DatabaseConfig()
    
    # Step 2: Initialize the database (loads all reference data)
    print("Loading database...")
    database = Database(config=database_config)
    
    # Step 3: Initialize individual runners
    print("Initializing runners...")
    crf_runner = RunnerCRF(config=crf_config, database=database)
    fuzzy_runner = RunnerFuzzyMatch(config=fuzzy_match_config, database=database)
    postcode_runner = RunnerPostcodeMatch(config=postcode_config, database=database)
    post_processing_runner = RunnerPostProcessing(
        config=post_processing_config,
        town_weights=town_weights_config,
        country_weights=country_weights_config,
        database=database
    )
    
    # Step 4: Clean and prepare the address
    print(f"\nProcessing address:\n{address_text}\n")
    cleaned_address = decode_and_clean_str(
        address_text.replace("\\n", "\n").replace("\r", "").upper()
    )
    
    # Validate address length
    if len(cleaned_address) > crf_config.max_sequence_length:
        raise ValueError(
            f"Address too long: {len(cleaned_address)} characters. "
            f"Maximum: {crf_config.max_sequence_length}"
        )
    
    # Step 5: Run each stage of the pipeline individually
    print("=" * 60)
    print("STAGE 1: CRF Tagging")
    print("=" * 60)
    crf_results = list(crf_runner.tag([cleaned_address]))
    crf_result = crf_results[0]
    print(f"CRF predictions per tag: {list(crf_result.predictions_per_tag.keys())}")
    for tag, predictions in crf_result.predictions_per_tag.items():
        print(f"___  {tag}: {[p.prediction for p in predictions]}")
    
    print("\n" + "=" * 60)
    print("STAGE 2: Fuzzy Matching")
    print("=" * 60)
    fuzzy_results = list(fuzzy_runner.match([cleaned_address]))
    fuzzy_result = fuzzy_results[0]
    print(f"Country matches: {len(fuzzy_result.country_matches)}")
    for country_match in fuzzy_result.country_matches:
        print(f"  - {country_match.matched} (distance: {country_match.dist}, possibility: {country_match.possibility})")
    print(f"Town matches: {len(fuzzy_result.town_matches)}")
    for town_match in fuzzy_result.town_matches:
        print(f"  - {town_match.matched} (distance: {town_match.dist}, possibility: {town_match.possibility})")
    
    print("\n" + "=" * 60)
    print("STAGE 3: Postcode Matching")
    print("=" * 60)
    postcode_results = list(postcode_runner.match([cleaned_address]))
    postcode_result = postcode_results[0]
    print(f"Postcode matches found: {len(postcode_result.postcode_matches)}")
    for postcode_match in postcode_result.postcode_matches:
        print(f"  - Matched: {postcode_match.matched}, Town: {postcode_match.possibility}, Country: {postcode_match.origin}")
    
    print("\n" + "=" * 60)
    print("STAGE 4: Post-Processing")
    print("=" * 60)
    post_processing_results = list(post_processing_runner.run(
        crf_results,
        fuzzy_results,
        postcode_results
    ))
    final_result = post_processing_results[0]
    print(f"Best country match: {final_result.i_th_best_match_country(0)}")
    print(f"Best town match: {final_result.i_th_best_match_town(0)}")
    
    # Step 6: Return comprehensive results
    return {
        "cleaned_address": cleaned_address,
        "crf_result": crf_result,
        "fuzzy_result": fuzzy_result,
        "postcode_result": postcode_result,
        "final_result": final_result
    }


def direct_crf_only(address_text: str):
    """
    Call ONLY the CRF model directly without any other stages.
    Useful when you only want NER/tagging results.
    """
    
    crf_config = CRFConfig()
    database_config = DatabaseConfig()
    
    print("Loading database...")
    database = Database(config=database_config)
    
    print("Initializing CRF runner...")
    crf_runner = RunnerCRF(config=crf_config, database=database)
    
    cleaned_address = decode_and_clean_str(address_text.upper())
    
    print(f"\nRunning CRF on: {cleaned_address}\n")
    crf_results = list(crf_runner.tag([cleaned_address]))

    crf_result = crf_results[0]

    print(f"CRF predictions per tag: {list(crf_result.predictions_per_tag.keys())}")
    for tag, predictions in crf_result.predictions_per_tag.items():
        print(f"___  {tag}: {[p.prediction for p in predictions]}")
    
    return crf_result


def direct_fuzzy_matching_only(address_text: str):
    """
    Call ONLY the fuzzy matching runner.
    Useful when you want to test fuzzy matching independently.
    """
    
    fuzzy_config = FuzzyMatchConfig()
    database_config = DatabaseConfig()
    
    print("Loading database...")
    database = Database(config=database_config)
    
    print("Initializing fuzzy matcher...")
    fuzzy_runner = RunnerFuzzyMatch(config=fuzzy_config, database=database)
    
    cleaned_address = decode_and_clean_str(address_text.upper())
    
    print(f"\nRunning fuzzy matching on: {cleaned_address}\n")
    fuzzy_results = list(fuzzy_runner.match([cleaned_address]))
    
    return fuzzy_results[0]


if __name__ == "__main__":
    # Example address
    test_address_list = ["""1600 Pennsylvania Ave NW Washington, DC 20500, USA"""]
    
    for test_address in [str(addr) for addr in test_address_list]:

        # Option 1: Run full pipeline directly
        print("\n" + "="*60)
        print("OPTION 1: Full Direct Model Usage")
        print("="*60)
        results = direct_model_inference(test_address)
        
        # Option 2: Run CRF only
        print("\n\n" + "="*60)
        print("OPTION 2: CRF Only")
        print("="*60)
        crf_only = direct_crf_only(test_address)
        print(f"CRF Tags: {list(crf_only.predictions_per_tag.keys())}")
        
        # Option 3: Run fuzzy matching only
        print("\n\n" + "="*60)
        print("OPTION 3: Fuzzy Matching Only")
        print("="*60)
        fuzzy_only = direct_fuzzy_matching_only(test_address)
        print(f"Countries found: {len(fuzzy_only.country_matches)}")
        print(f"Towns found: {len(fuzzy_only.town_matches)}")
