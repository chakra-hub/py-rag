from groq import Groq
from pydantic import BaseModel
from config import settings

# Store conversation history per session
conversations = {}

class ChatRequest(BaseModel):
    session_id: str
    question: str
    context: str = None

class AskLLM:
    def __init__(self):
        self.client = Groq(api_key=settings.groq_api_key)

    async def chat_with_groq(self, session_id: str, question: str, context: str = None):
        """
        Chat with Groq, maintaining conversation history per session
        Args:
            session_id: Unique identifier for the conversation
            question: User's question
            context: Optional context from RAG
        """
        # Initialize conversation history if new session
        if session_id not in conversations:
            conversations[session_id] = [
                {"role": "system", "content": "You are a helpful assistant."}
            ]
        
        # Build the user message with context if provided
        user_message = question
        if context:
            user_message = f"Context: {context}\n\nQuestion: {question}"
        
        # Add user message to history
        conversations[session_id].append({"role": "user", "content": user_message})
        
        try:
            # Send entire conversation history to maintain context
            chat_completion = self.client.chat.completions.create(
                messages=conversations[session_id],
                model="llama-3.3-70b-versatile"
            )
            
            assistant_reply = chat_completion.choices[0].message.content
            
            # Add assistant response to history
            conversations[session_id].append({"role": "assistant", "content": assistant_reply})
            
            return {"response": assistant_reply, "session_id": session_id}
        
        except Exception as e:
            # Remove the last user message if there was an error
            conversations[session_id].pop()
            raise Exception(f"LLM error: {str(e)}") 