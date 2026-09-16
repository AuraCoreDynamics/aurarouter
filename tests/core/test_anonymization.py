"""Tests for the data anonymization pipeline."""

import pytest
from aurarouter.sovereignty.privacy import PrivacyMatch
from aurarouter.sovereignty.anonymization import AnonymizationPipeline, RegexAnonymizer

def test_regex_anonymizer_anonymize_deanonymize():
    anonymizer = RegexAnonymizer()
    prompt = "Contact user@example.com with SSN 123-45-6789 and api_key=sk-1234."
    
    matches = [
        PrivacyMatch(pattern_name="Email Address", severity="medium", matched_text="user@example.com", position=8, category="PII"),
        PrivacyMatch(pattern_name="SSN", severity="high", matched_text="123-45-6789", position=34, category="PII"),
        PrivacyMatch(pattern_name="API Key", severity="high", matched_text="sk-1234", position=54, category="CREDENTIALS")
    ]
    
    masked_prompt, mapping = anonymizer.anonymize(prompt, matches)
    
    # Check that original text is gone and replaced by placeholders
    assert "user@example.com" not in masked_prompt
    assert "123-45-6789" not in masked_prompt
    assert "sk-1234" not in masked_prompt
    
    # There should be 3 placeholders in the mapping
    assert len(mapping) == 3
    
    # Now deanonymize
    # Pretend the model echoed the placeholders
    model_response = f"I noted the email {list(mapping.keys())[0]}, the SSN {list(mapping.keys())[1]}, and the key {list(mapping.keys())[2]}."
    deanonymized = anonymizer.deanonymize(model_response, mapping)
    
    assert "user@example.com" in deanonymized
    assert "123-45-6789" in deanonymized
    assert "sk-1234" in deanonymized

def test_anonymization_pipeline_orchestration():
    pipeline = AnonymizationPipeline(plugins=[RegexAnonymizer()])
    prompt = "Here is my ip 192.168.1.100."
    matches = [
        PrivacyMatch(pattern_name="Private IP", severity="low", matched_text="192.168.1.100", position=15, category="INFRA")
    ]
    
    masked_prompt, mapping = pipeline.anonymize(prompt, matches)
    assert "192.168.1.100" not in masked_prompt
    assert len(mapping) == 1
    
    placeholder = list(mapping.keys())[0]
    deanonymized = pipeline.deanonymize(f"Connecting to {placeholder}", mapping)
    assert "Connecting to 192.168.1.100" == deanonymized
