import re
import string

def clean_reference(ref):
    """
    Clean and preprocess reference text for machine learning.
    
    Args:
        ref (str): Raw reference text
        
    Returns:
        str: Cleaned reference text
    """
    if not isinstance(ref, str):
        return ""
    
    # Convert to lowercase
    ref = ref.lower()
    
    # Strip leading/trailing whitespace
    ref = ref.strip()
    
    # Remove newline characters
    ref = ref.replace('\n', ' ').replace('\r', ' ')
    
    # Remove extra whitespace
    ref = re.sub(r'\s+', ' ', ref)
    
    # Keep only alphanumeric characters and basic punctuation
    # Allow: letters, numbers, spaces, periods, commas, colons, semicolons, hyphens
    ref = re.sub(r'[^a-z0-9\s\.,;:\-]', '', ref)
    
    # Remove extra spaces again after punctuation removal
    ref = re.sub(r'\s+', ' ', ref).strip()
    
    return ref