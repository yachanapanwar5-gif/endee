from app import search

print("🔍 AI CRM Semantic Search Demo")
print("--------------------------------")

while True:
    query = input("\nEnter your search (or type 'exit'): ")
    
    if query.lower() == 'exit':
        print("Goodbye!")
        break
    
    result = search(query)
    print(f"Best match: {result}")
