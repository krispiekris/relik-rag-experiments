from ollama import chat

response = chat(
    model="openbmb/minicpm5",
    messages=[
        {
            "role": "user",
            "content": (
                "What is retrieval-augmented generation? "
                "Answer in two sentences."
            ),
        }
    ],
    think=False,
)

print(response.message.content)