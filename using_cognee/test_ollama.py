import asyncio
import cognee
from cognee.shared.logging_utils import get_logger, ERROR
from cognee.api.v1.search import SearchType
 
async def main():
    # Create a clean slate for cognee -- reset data and system state
    cognee.config.use_instructor = False
    cognee.config.enable_summarization = False
    cognee.config.enable_classification = False
 
    # cognee knowledge graph will be created based on this text
    text = """
    Natural language processing (NLP) is an interdisciplinary
    subfield of computer science and information retrieval.
    """
 
    # Add the text, and make it available for cognify
    await cognee.add(text)
 
    # Run cognify and build the knowledge graph using the added text
    await cognee.cognify()
 
    # Query cognee for insights on the added text
    query_text = "Tell me about NLP"
    search_results = await cognee.search(
        query_type=SearchType.INSIGHTS, 
        query_text=query_text
    )
    
    for result_text in search_results:
        print(result_text)
 
if __name__ == "__main__":
    logger = get_logger(level=ERROR)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())