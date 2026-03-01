# import os
# from groq import Groq
# from dotenv import load_dotenv

# load_dotenv()
# quary = input("Enter what you want ?")
# client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# response = client.chat.completions.create(
#     model="llama-3.1-8b-instant",
#     messages=[{"role": "user", "content" : "quary"}]
# )

# print(response.choices[0].message.content)