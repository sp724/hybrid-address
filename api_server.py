"""
REST API for Address Structuring using FastAPI.

This API accepts postal addresses in JSON format and returns the detected
country and town with confidence scores.

Example request:
{
    "address": "1600 Pennsylvania Ave NW\nWashington, DC 20500\nUSA"
}

Example response:
{
    "success": true,
    "address": "1600 Pennsylvania Ave NW\nWashington, DC 20500\nUSA",
    "country": {
        "name": "US",
        "confidence": 0.9857,
        "iso_code": "USA"
    },
    "town": {
        "name": "WASHINGTON",
        "confidence": 0.9791
    }
}
"""

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional
import polars as pl
import uvicorn

from data_structuring.pipeline import AddressStructuringPipeline
from data_structuring.components.readers.dataframe_reader import DataFrameReader

# Initialize FastAPI app
app = FastAPI(
    title="Address Structuring API",
    description="API for structuring postal addresses and extracting country and town information",
    version="1.0.0"
)

# Initialize the pipeline once at startup
print("Initializing AddressStructuringPipeline...")
pipeline = AddressStructuringPipeline()
print("Pipeline initialized successfully!")


# ============================================================================
# Request/Response Models
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    service: str


class CountryMatch(BaseModel):
    """Country match details."""
    name: str = Field(..., description="Country name or code")
    confidence: Optional[float] = Field(..., description="Confidence score between 0 and 1")
    iso_code: Optional[str] = Field(..., description="ISO country code")


class TownMatch(BaseModel):
    """Town match details."""
    name: str = Field(..., description="Town name")
    confidence: Optional[float] = Field(..., description="Confidence score between 0 and 1")


class PostalCodeMatch(BaseModel):
    """Postal code/ZIP code match details."""
    code: str = Field(..., description="Postal code or ZIP code")
    town: Optional[str] = Field(None, description="Town associated with the postal code")
    country: Optional[str] = Field(None, description="Country associated with the postal code")


class SingleAddressRequest(BaseModel):
    """Request model for single address structuring."""
    address: str = Field(..., description="Postal address string", min_length=1)


class SingleAddressResponse(BaseModel):
    """Response model for single address structuring."""
    success: bool
    address: str
    country: CountryMatch
    town: TownMatch
    postal_code: Optional[PostalCodeMatch] = Field(None, description="Postal code/ZIP code if detected")


class BatchAddressRequest(BaseModel):
    """Request model for batch address structuring."""
    addresses: List[str] = Field(..., description="List of postal addresses", min_items=1)


class BatchAddressResult(BaseModel):
    """Result for a single address in batch response."""
    address: str
    country: CountryMatch
    town: TownMatch
    postal_code: Optional[PostalCodeMatch] = Field(None, description="Postal code/ZIP code if detected")


class BatchAddressResponse(BaseModel):
    """Response model for batch address structuring."""
    success: bool
    count: int = Field(..., description="Number of addresses processed")
    results: List[BatchAddressResult]




# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify the API is running.
    
    Returns:
        HealthResponse: Service status information
    """
    return HealthResponse(
        status="healthy",
        service="Address Structuring API"
    )


@app.post("/api/structure-address", response_model=SingleAddressResponse)
async def structure_address(request: SingleAddressRequest):
    """
    Structure a single postal address and return country and town.
    
    Args:
        request: SingleAddressRequest containing the address string
        
    Returns:
        SingleAddressResponse: Structured address with country and town matches
        
    Raises:
        HTTPException: If address processing fails
    """
    
    try:
        address = request.address.strip()
        
        if not address:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Address cannot be empty"
            )
        
        # Create DataFrame with the address
        df = pl.DataFrame({"addresses": [address]})
        
        # Run through pipeline
        results = pipeline.run(
            DataFrameReader(df, "addresses"),
            batch_size=1024
        )
        
        if not results:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No results returned from pipeline"
            )
        
        result = results[0]
        
        # Extract country
        country_name, country_confidence, country_iso = result.i_th_best_match_country(
            0, value_if_none="UNKNOWN"
        )
        
        # Extract town
        town_name, town_confidence, _ = result.i_th_best_match_town(
            0, value_if_none="UNKNOWN"
        )
        
        # Extract postal code if available
        postal_code_match = None
        if hasattr(result, 'postcode_matches') and result.postcode_matches:
            for postcode_match in result.postcode_matches:
                postal_code_match = PostalCodeMatch(
                    code=postcode_match.matched,
                    town=postcode_match.possibility,
                    country=postcode_match.origin
                )
                break  # Get first match
        
        # Build response
        return SingleAddressResponse(
            success=True,
            address=address,
            country=CountryMatch(
                name=country_name,
                confidence=float(country_confidence) if country_confidence else None,
                iso_code=country_iso
            ),
            town=TownMatch(
                name=town_name,
                confidence=float(town_confidence) if town_confidence else None
            ),
            postal_code=postal_code_match
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Server error: {str(e)}"
        )


@app.post("/api/structure-addresses-batch", response_model=BatchAddressResponse)
async def structure_addresses_batch(request: BatchAddressRequest):
    """
    Structure multiple postal addresses in batch.
    
    Args:
        request: BatchAddressRequest containing a list of address strings
        
    Returns:
        BatchAddressResponse: Structured addresses with country and town matches
        
    Raises:
        HTTPException: If batch processing fails
    """
    
    try:
        addresses = [addr.strip() for addr in request.addresses]
        
        # Validate all addresses are non-empty
        if not all(addresses):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="All addresses must be non-empty strings"
            )
        
        # Create DataFrame with addresses
        df = pl.DataFrame({"addresses": addresses})
        
        # Run through pipeline
        results = pipeline.run(
            DataFrameReader(df, "addresses"),
            batch_size=1024
        )
        
        if not results:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No results returned from pipeline"
            )
        
        # Build response
        structured_results = []
        for i, result in enumerate(results):
            country_name, country_confidence, country_iso = result.i_th_best_match_country(
                0, value_if_none="UNKNOWN"
            )
            town_name, town_confidence, _ = result.i_th_best_match_town(
                0, value_if_none="UNKNOWN"
            )
            
            # Extract postal code if available
            postal_code_match = None
            if hasattr(result, 'postcode_matches') and result.postcode_matches:
                for postcode_match in result.postcode_matches:
                    postal_code_match = PostalCodeMatch(
                        code=postcode_match.matched,
                        town=postcode_match.possibility,
                        country=postcode_match.origin
                    )
                    break  # Get first match
            
            structured_results.append(
                BatchAddressResult(
                    address=addresses[i],
                    country=CountryMatch(
                        name=country_name,
                        confidence=float(country_confidence) if country_confidence else None,
                        iso_code=country_iso
                    ),
                    town=TownMatch(
                        name=town_name,
                        confidence=float(town_confidence) if town_confidence else None
                    ),
                    postal_code=postal_code_match
                )
            )
        
        return BatchAddressResponse(
            success=True,
            count=len(structured_results),
            results=structured_results
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Server error: {str(e)}"
        )


# ============================================================================
# Run the API
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="127.0.0.1",
        port=5000,
        reload=False,
        log_level="info"
    )
