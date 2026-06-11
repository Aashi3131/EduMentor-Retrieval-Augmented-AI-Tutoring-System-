# EduMentor: Semantic RAG Tutoring System
EduMentor is an interactive, vector-based Retrieval-Augmented Generation (RAG) tutoring system. It enables students to upload slide decks, lecture notes, and ask questions. The system returns answers strictly grounded in the document context, accompanied by automated dual-source fact-checking badges.

What to expect when you run this project:

# 1. Use cmd: streamlit run app.py 
and the following will open:
<img width="401" height="282" alt="image" src="https://github.com/user-attachments/assets/a633bb0e-8c4d-4fe4-b121-4219d9336659" />


# 2. Go to OpenRouter
Get your API Key & insert it.

# 3. The following will open:
Choose your tutor model,
then choose if you want to verify the response generated through pdf & web or not (If yes, choose verifier model as well)
<img width="309" height="325" alt="image" src="https://github.com/user-attachments/assets/6aa99a4d-70d2-4bb0-bc61-fd6520e74dee" />


# 4. Browse through your files
Upload the PDFs, and get the generated response.
<img width="742" height="553" alt="image" src="https://github.com/user-attachments/assets/009f80f0-8848-4701-bc45-bced5fdb1003" />

You can also click on "Generate Chat History" in the sidebar and then download the chat in pretty HTML document.
<img width="283" height="154" alt="image" src="https://github.com/user-attachments/assets/a80387cf-b80a-4265-a46d-993d2b58710d" />

This is how it will look like:
<img width="555" height="316" alt="image" src="https://github.com/user-attachments/assets/5f9f91e5-917d-4e19-bb00-034c1f85878f" />

This project showcases concept of Retrieval-Augmented Generation (RAG), a way to make AI answers more reliable by combining searching for relevant information and then generating a response instead of guessing based only on old training data. It finds useful data from external sources (like documents) and then uses it to give a better answer. 

Concepts this project covers:
1. External Knowledge Source: Stores domain specific or general information like documents.
2. Text Chunking and Preprocessing: Breaks large text into smaller, manageable chunks and cleans it for consistency.
3. Embedding Model: Converts text into numerical vectors that capture semantic meaning.
4. Vector Database: Stores embeddings and enables similarity search for fast information retrieval.
5. Query Encoder: Transforms the user’s query into a vector for comparison with stored embeddings.
6. Retriever: Finds and returns the most relevant chunks from the database based on query similarity.
7. Prompt Augmentation Layer: Combines retrieved chunks with the user’s query to provide context to the LLM.
8. LLM (Generator): Generates a grounded response using both the query and retrieved knowledge.
9. Verifier (Optional): Checks if response generated is valid according to the PDF and Web.
