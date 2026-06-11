# EduMentor: Semantic RAG Tutoring System
EduMentor is an interactive, vector-based Retrieval-Augmented Generation (RAG) tutoring system. It enables students to upload slide decks, lecture notes, and ask questions. The system returns answers strictly grounded in the document context, accompanied by automated dual-source fact-checking badges.

What to expect when you run this project:

# 1. Use cmd: streamlit run app.py 
and the following will open:
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/1bd86a01-a1fc-4bc9-bdca-7efecd3a3dc2" />

# 2. Go to OpenRouter
Get your API Key & insert it.

#3. The following will open:
Choose your tutor model,
then choose if you want to verify the response generated through pdf & web or not (If yes, choose verifier model as well)
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/7081e882-3f4c-4705-8e81-3aff420127a8" />

# 4. Browse through your files
Upload the PDFs, and get the generated response.
You can also click on "Generate Chat History" in the sidebar and then download the chat in pretty HTML document.
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/ff00fd5d-b131-4b4b-9f05-696826fc22a9" />

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

Here's the ingesetion flow:
<img width="338" height="183" alt="image" src="https://github.com/user-attachments/assets/6716bc83-94f5-4391-9765-bc9e55a9635c" />

Here's the Query Flow:
<img width="664" height="529" alt="image" src="https://github.com/user-attachments/assets/b4402802-4764-47f8-a201-621821952113" />

# Note: Individual pages of a PDF are chunked seperately, causing page boundary context loss. Can be improved by chunking the whole PDF Text at once.
