import cognee
import asyncio
from cognee.api.v1.search import SearchType

async def main():
    query = "Who was Albert Einstein?"
    results = await cognee.search(query_type=SearchType.RAG_COMPLETION, query_text=query)

    if results:
        print("✅ Data is stored and searchable.")
        for res in results:
            print(res)
            print("\n---\n")
    else:
        print("❌ No data found or search did not return any results.")

asyncio.run(main())