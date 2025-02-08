# -*- coding: utf-8 -*-

!pip install --quiet langchain-google-genai

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro", temperature=0.1, google_api_key="AIzaSyBkFf0VJ8IxIMlv7lBfF6CSKwFvry7aPA0")

#test to confirm if packages are installed and working correctly
response = llm.invoke(
    "Improve this description : In this notebok we'll explore advanced techniques, and building reAct agents using LangChain and Gemini-1.0"
)

print(response.content)

question="""Q:Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
Each can has 3 tennis balls. How many tennis balls does he have now?
A:The answer is 11.
Q : The cafeteria had 23 apples.
If they used 20 to make lunch and bought 6 more, how many apples do they have?"""
response = llm.invoke(question)
print(response.content)

question="""Q:Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
Each can has 3 tennis balls. How many tennis balls does he have now?
A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls.
5 + 6 = 11. That's how many tennis balls he has now.
Q : The cafeteria had 23 apples.
If they used 20 to make lunch and bought 6 more, how many apples do they have?"""
response = llm.invoke(question)
print(response.content)

question="""Q:Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
Each can has 3 tennis balls. How many tennis balls does he have now?
A:The answer is 11.

Q : The cafeteria had 23 apples.
If they used 20 to make lunch and bought 6 more, how many apples do they have?
A : Let's think step by step"""

response = llm.invoke(question)
print(response.content)

question="""A car travels 80 km at a speed of v km/h and then travels 120km at a speed of (v-10) km/h.
          The total time for the journey is 4 hours. Solve for v
          find the value of v and show your step-by-step reasoning\n\n
          Step 1 : Write the time equation: 80/v +120/(v-10) = 4.\n
          Step 2 : Multiply both the sides by v(v-10) to eliminate fractions:\n
           80(v-10) + 120v = 4v(v-10).\n
          Step 3 : Expand and simplify the eqution to obtain a quadratic equation:\n
          Step 4 : Solve the quadratic equation for v:\n
          Now, Solve the problem : \n
          A car travels 100 km at a speed of v km/h and then travels 150km at a speed of (v-20) km/h.
          The total time for the journey is 5 hours
          Find the value of v and show your step-by-step reasoning"""

response = llm.invoke(question)
print(response.content)

