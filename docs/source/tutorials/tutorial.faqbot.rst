===========================
FAQ Bot in 20 Lines of Code
===========================

Turn your company FAQ documents into an intelligent chatbot that can answer questions in natural language. This tutorial shows you how to build a working FAQ bot with just 20 lines of Python code.

What You'll Build
=================

A smart FAQ bot that:
- Loads your FAQ documents automatically
- Understands questions asked in different ways
- Returns the most relevant answers
- Works instantly with any FAQ content

The Complete Bot (20 Lines)
===========================

Create a file called ``faq_bot.py``:

.. code-block:: python

   from localvectordb import LocalVectorDB

   # Create database and load FAQ documents
   print("Loading FAQ bot...")
   db = LocalVectorDB(name="faq_bot", embedding_provider="ollama", embedding_model="nomic-embed-text")

   # Sample FAQ content (replace with your own)
   faqs = [
       "Q: What are your business hours? A: We're open Monday-Friday 9 AM to 6 PM EST, closed weekends and holidays.",
       "Q: How do I reset my password? A: Click 'Forgot Password' on the login page, enter your email, and follow the link sent to you.",
       "Q: What payment methods do you accept? A: We accept all major credit cards, PayPal, bank transfers, and Apple Pay.",
       "Q: How do I cancel my subscription? A: Go to Account Settings > Billing > Cancel Subscription, or contact support@company.com.",
       "Q: Do you offer refunds? A: Yes, we offer full refunds within 30 days of purchase, no questions asked.",
       "Q: How do I contact customer support? A: Email support@company.com, call 1-800-HELP-NOW, or use live chat on our website.",
       "Q: What's your shipping policy? A: Free shipping on orders over $50, standard delivery 3-5 business days, express 1-2 days.",
       "Q: Can I change my account email? A: Yes, go to Account Settings > Profile > Email Address and verify the new email."
   ]

   # Add FAQs to database (only do this once)
   if db.get_stats()['documents'] == 0:
       print(f"Adding {len(faqs)} FAQ entries...")
       db.upsert(faqs)

   # Interactive chat loop
   print("FAQ Bot ready! Ask me anything (type 'quit' to exit)")
   while True:
       question = input("\nYour question: ").strip()
       if question.lower() in ['quit', 'exit', 'bye']:
           break
       
       results = db.query(question, k=1)
       if results and results[0].score > 0.3:
           print(f"Answer: {results[0].content}")
       else:
           print(":~| Sorry, I don't have information about that. Try rephrasing your question.")

That's it! Run it with:

.. code-block:: bash

   python faq_bot.py


Example Conversation
====================

.. code-block:: text

   Loading FAQ bot...
   Adding 8 FAQ entries...
   FAQ Bot ready! Ask me anything (type 'quit' to exit)

   Your question: when are you open?
   Answer: Q: What are your business hours? A: We're open Monday-Friday 9 AM to 6 PM EST, closed weekends and holidays.

   Your question: i forgot my password
   Answer: Q: How do I reset my password? A: Click 'Forgot Password' on the login page, enter your email, and follow the link sent to you.

   Your question: can i pay with credit card?
   Answer: Q: What payment methods do you accept? A: We accept all major credit cards, PayPal, bank transfers, and Apple Pay.

   Your question: how much does shipping cost?
   Answer: Q: What's your shipping policy? A: Free shipping on orders over $50, standard delivery 3-5 business days, express 1-2 days.

   Your question: quit
   Goodbye!

Why This Works So Well
======================

**Smart Matching**: The bot finds relevant answers even when you ask questions differently:
- "when are you open?" → finds "business hours"
- "i forgot my password" → finds "reset my password" 
- "can i pay with credit card?" → finds "payment methods"

**Semantic Understanding**: Uses AI embeddings to understand meaning, not just keywords.

**Confidence Scoring**: Only answers when confident (score > 0.3), otherwise asks for clarification.

Customize for Your Business
===========================

Replace the sample FAQs with your own content:

**Load from a File**

.. code-block:: python

   # Read FAQs from a text file
   with open('company_faqs.txt', 'r') as f:
       faqs = [line.strip() for line in f if line.strip()]

**Load from CSV**

.. code-block:: python

   import csv
   
   faqs = []
   with open('faqs.csv', 'r') as f:
       reader = csv.DictReader(f)
       for row in reader:
           faqs.append(f"Q: {row['Question']} A: {row['Answer']}")

**Different FAQ Format**

.. code-block:: python

   # If your FAQs are just question-answer pairs
   faq_pairs = [
       ("What are your hours?", "Monday-Friday 9 AM to 6 PM EST"),
       ("How do I reset password?", "Click 'Forgot Password' on login page"),
       # ... more pairs
   ]
   
   faqs = [f"Q: {q} A: {a}" for q, a in faq_pairs]

Enhanced Version (30 Lines)
===========================

Want a slightly more sophisticated bot? Here's an enhanced version:

.. code-block:: python

   from localvectordb import LocalVectorDB
   import json

   class FAQBot:
       def __init__(self, faq_data):
           self.db = LocalVectorDB(name="enhanced_faq", embedding_provider="ollama", embedding_model="nomic-embed-text")
           self.load_faqs(faq_data)
       
       def load_faqs(self, faqs):
           if self.db.get_stats()['documents'] == 0:
               print(f"Loading {len(faqs)} FAQ entries...")
               # Add with metadata for better organization
               documents = []
               metadata = []
               for i, faq in enumerate(faqs):
                   documents.append(faq)
                   metadata.append({"faq_id": i, "category": "general"})
               self.db.upsert(documents, metadata=metadata)
       
       def ask(self, question):
           results = self.db.query(question, k=1, score_threshold=0.3)
           if results:
               return results[0].content
           return "I don't have information about that. Could you rephrase your question?"
       
       def chat(self):
           print("Enhanced FAQ Bot ready! (type 'quit' to exit)")
           while True:
               question = input("\nYour question: ").strip()
               if question.lower() in ['quit', 'exit']:
                   break
               print(f"{self.ask(question)}")

   # Usage
   faqs = [
       "Q: What are your business hours? A: We're open Monday-Friday 9 AM to 6 PM EST.",
       # ... your FAQ content
   ]

   bot = FAQBot(faqs)
   bot.chat()

Real-World Applications
=======================

This simple pattern works great for:

**Customer Support**
- Company policy questions
- Product information
- Troubleshooting steps
- Account management

**Internal Knowledge Base**
- Employee handbook questions
- IT support procedures
- Company process documentation
- Training materials

**Product Documentation**
- API usage questions
- Feature explanations
- Integration guides
- Best practices

Integration Ideas
=================

**Web Interface**: Add Flask/FastAPI to create a web-based chat interface

**Slack Bot**: Integrate with Slack API for team-wide FAQ access

**WhatsApp/SMS**: Connect to messaging platforms for customer support

**Website Widget**: Embed in your website as a help chat widget

Advanced Features to Add
========================

**Categories and Filtering**

.. code-block:: python

   # Search within specific categories
   results = db.query(question, filters={"category": "billing"})

**Conversation Memory**

.. code-block:: python

   # Track conversation context for follow-up questions
   conversation_history = []

**Analytics**

.. code-block:: python

   # Track what questions people ask most
   question_analytics = {}

**Auto-Improvement**

.. code-block:: python

   # Identify questions with no good answers
   # Add new FAQs based on common unanswered questions

Next Steps
==========

Your FAQ bot is ready to use! To make it even better:

1. **Add more FAQs**: The more content, the better the answers
2. **Test with real questions**: See how users actually ask questions
3. **Monitor performance**: Track which questions get good answers
4. **Iterate**: Add new FAQs based on real user needs

**Want to go further?** Try these tutorials:
- **Index Your Downloads Folder**: Search through actual documents
- **Building a RAG Chat Application**: Add conversation memory and context

Congratulations!
================

You've built a working FAQ bot that:

- Understands natural language questions
- Finds relevant answers intelligently
- Handles variations in how people ask questions
- Can be customized for any business or use case

This same 20-line pattern can handle hundreds of FAQs and thousands of questions!